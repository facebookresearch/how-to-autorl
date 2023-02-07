# All of this is copied/lightly adapted from the original BG-PBT code: https://github.com/xingchenwan/bgpbt

import os
import sys
import math
import torch
import numpy as np
import logging
from copy import deepcopy
from typing import Callable, List
from abc import ABC, abstractmethod

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.util import deactivate_inactive_hyperparameters

from hydra_plugins.utils.lazy_imports import lazy_import

ss = lazy_import("scipy.stats")
gpytorch = lazy_import("gpytorch")

MAX_CHOLESKY_SIZE = 2000
MIN_CUDA = 1024
DEVICE = "cpu"


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


def normalize(data, wrt):
    """Normalize data to be in range (0,1), with respect to (wrt) boundaries,
    which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (np.max(wrt, axis=0) - np.min(wrt, axis=0))


def grad_search(
    cs: CS.ConfigurationSpace,
    x_center,
    f: Callable,
    n_restart: int = 1,
    step: int = 40,
    batch_size: int = 1,
    dtype=torch.float,
    fixed_dims=None,
    verbose: bool = True,
):
    """Vanilla gradient-based search"""
    num_fixed_dims = x_center.shape[0] - len(cs) if x_center.shape[0] > len(cs) else 0
    if num_fixed_dims > 0:
        fixed_dims = list(range(len(cs), x_center.shape[0]))
    else:
        fixed_dims = None

    x0s = []
    for _ in range(n_restart):
        p = cs.sample_configuration().get_array()
        if fixed_dims is not None:
            p = np.concatenate((p, x_center[fixed_dims]))
        x0s.append(p)
    x0 = np.array(x0s).astype(np.float)

    def _grad_search(x0):
        lb, ub = np.zeros(x0.shape[0]), np.ones(x0.shape[0])
        n_step = 0
        x = deepcopy(x0)
        acq_x = f(x).detach().numpy()
        x_tensor = torch.tensor(x, dtype=dtype).requires_grad_(True)
        optimizer = torch.optim.Adam([{"params": x_tensor}], lr=0.1)

        while n_step <= step:
            optimizer.zero_grad()
            acq = f(x_tensor)
            acq.backward()
            if num_fixed_dims:
                x_tensor.grad[fixed_dims] = 0.0
            if verbose and n_step % 20 == 0:
                logging.info(f"Acquisition optimisation: Step={n_step}: Value={x_tensor}. Acq={acq_x}.")
            optimizer.step()
            with torch.no_grad():
                x_tensor = torch.maximum(
                    torch.minimum(x_tensor, torch.tensor(ub).to(x_tensor.dtype)), torch.tensor(lb).to(x_tensor.dtype)
                )
            n_step += 1
        x = x_tensor.detach().numpy().astype(x0.dtype)
        acq_x = f(x).detach().numpy()
        del x_tensor
        return x, acq_x

    X, fX = [], []
    for i in range(n_restart):
        res = _grad_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])
    top_idices = np.argpartition(np.array(fX).flatten(), batch_size)[:batch_size]
    return (
        np.array([x for i, x in enumerate(X) if i in top_idices]).astype(np.float),
        np.array(fX).astype(np.float).flatten()[top_idices],
    )


def get_start_point(cs: CS.ConfigurationSpace, x_center, frozen_dims: List[int] = None, return_config=False):
    # get a perturbed starting point from x_center
    new_config_array = deepcopy(x_center)

    perturbation_factor = [0.8, 1.2]  # <- taken from PB2
    for i in range(new_config_array.shape[0]):
        # print(param_name)
        if np.isnan(new_config_array[i]) or (frozen_dims is not None and i in frozen_dims):
            continue
        param_name = cs.get_hyperparameter_by_idx(i)
        if type(cs[param_name]) == CSH.CategoricalHyperparameter:
            new_config_array[i] = np.random.choice(range(len(cs[param_name].choices)))
        elif (
            type(cs[param_name]) in [CSH.UniformIntegerHyperparameter, CSH.NormalIntegerHyperparameter]
            and cs[param_name].lower == 0
            and cs[param_name].upper == 1
        ):  # binary
            new_config_array[i] = np.random.choice([0, 1])
        else:
            perturbation = np.random.uniform(*perturbation_factor)
            new_config_array[i] = np.clip(x_center[i] * perturbation, 0.0, 1.0)
    # whether to change the network depth
    config = CS.Configuration(cs, vector=new_config_array)
    config = deactivate_inactive_hyperparameters(config, cs)

    try:
        cs.check_configuration(config)
    except ValueError:
        config = CS.Configuration(cs, config.get_dictionary())
    new_config_array = config.get_array()
    if return_config:
        return new_config_array, config
    return new_config_array


def construct_bounding_box(
    cs: CS.ConfigurationSpace,
    x,
    tr_length,
    weights=None,
):
    """Construct a bounding box around x_cont with tr_length being the k-dimensional trust region size.
    The weights should be the learnt lengthscales of the GP surrogate model.
    """
    if weights is None:
        weights = 1.0 / len(x.shape[0]) * np.ones(x.shape[1])
    # non-ard lengthscales passed -- this must be a scalar input
    elif len(weights) != x.shape[0]:
        weights = weights * np.ones(x.shape[0])
    weights = weights / weights.mean()
    # We now have weights.prod() = 1
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    lb, ub = np.zeros_like(x), np.ones_like(x)
    for i, dim in enumerate(x):
        if np.isnan(x[i]) or i >= len(cs):
            lb[i], ub[i] = 0.0, 1.0
        else:
            hp = cs[cs.get_hyperparameter_by_idx(i)]
            if type(hp) == CSH.CategoricalHyperparameter:
                lb[i], ub[i] = 0, len(hp.choices)
            else:
                lb[i] = np.clip(x[i] - weights[i] * tr_length / 2.0, 0.0, 1.0)
                ub[i] = np.clip(x[i] + weights[i] * tr_length / 2.0, 0.0, 1.0)
                if type(hp) in [
                    CSH.UniformIntegerHyperparameter,
                    CSH.NormalIntegerHyperparameter,
                    CSH.NormalFloatHyperparameter,
                    CSH.UniformFloatHyperparameter,
                ]:
                    lb[i] = max(hp._inverse_transform(hp.lower), lb[i])
                    ub[i] = min(hp._inverse_transform(hp.upper), ub[i])
    return lb, ub


def get_dim_info(cs: CS.ConfigurationSpace, x, return_indices=False):
    """Return the information on the categorical, integer and continuous spaces"""
    if len(cs) != len(x):
        # this is because x is longer than cs -- the final dimensions are the contextual info presented as fixed dimensions.
        x = deepcopy(x)[: len(cs)]
    cat_dims, cont_dims, int_dims = [], [], []
    cat_dims_idx, cont_dims_idx, int_dims_idx = [], [], []
    for i, variable in enumerate(range(len(x))):
        # do not sample an inactivated hyperparameter (such a hyperparameter has nan value imputed)
        if x[variable] != x[variable]:
            continue
        if type(cs[cs.get_hyperparameter_by_idx(variable)]) == CSH.CategoricalHyperparameter:
            cat_dims.append(cs.get_hyperparameter_by_idx(variable))
            cat_dims_idx.append(i)
        elif type(cs[cs.get_hyperparameter_by_idx(variable)]) in [
            CSH.UniformIntegerHyperparameter,
            CSH.NormalIntegerHyperparameter,
        ]:
            int_dims.append(cs.get_hyperparameter_by_idx(variable))
            int_dims_idx.append(i)
        elif type(cs[cs.get_hyperparameter_by_idx(variable)]) in [
            CSH.UniformFloatHyperparameter,
            CSH.NormalFloatHyperparameter,
        ]:
            cont_dims.append(cs.get_hyperparameter_by_idx(variable))
            cont_dims_idx.append(i)
    if return_indices:
        return cat_dims_idx, cont_dims_idx, int_dims_idx
    return cat_dims, cont_dims, int_dims


def sample_discrete_neighbour(cs: CS.ConfigurationSpace, x, frozen_dims: List[int] = None):
    """Sample a neighbour from x in one of the active hyperparameter.
    select type:
    frozen_dims: the frozen dimensions where neighbours that differ from them will be rejected.
    """
    # note that for acquisition function optimisation (which this def is likely used), integer-type variables are treated
    # as discrete.
    assert len(x) >= len(cs)
    if len(x) > len(cs):
        # this is because x is longer than cs -- the final dimensions are the contextual info presented as fixed dimensions.
        fixed_dims = x[len(cs) :]
        x = deepcopy(x)[: len(cs)]
    else:
        fixed_dims = None
    cat_dims, _, int_dims = get_dim_info(cs, x)
    config = CS.Configuration(cs, vector=x.detach().numpy() if isinstance(x, torch.Tensor) else x)

    try:
        cs.check_configuration(config)
    except ValueError as e:
        # there seems to be a bug with ConfigSpace that raises error even when a config is valid
        # Issue #196: https://github.com/automl/ConfigSpace/issues/196
        # print(config)
        config = CS.Configuration(cs, config.get_dictionary())

    # print(config)
    config_pert = deepcopy(config)
    selected_dim = str(np.random.choice(cat_dims + int_dims, 1)[0])
    index_in_array = cs.get_idx_by_hyperparameter_name(selected_dim)
    while config_pert[selected_dim] is None or (frozen_dims is not None and index_in_array in frozen_dims):
        selected_dim = str(np.random.choice(cat_dims + int_dims, 1)[0])
        index_in_array = cs.get_idx_by_hyperparameter_name(selected_dim)

    # if the selected dimension is categorical, change the value to another variable
    if selected_dim in cat_dims:
        config_pert[selected_dim] = np.random.choice(cs[selected_dim].choices, 1)[0]
        while config_pert[selected_dim] == config[selected_dim]:
            config_pert[selected_dim] = np.random.choice(cs[selected_dim].choices, 1)[0]
    elif selected_dim in int_dims:
        lb, ub = cs[selected_dim].lower, cs[selected_dim].upper
        if selected_dim in ["NAS_policy_num_layers", "NAS_q_num_layers"]:
            candidates = list({max(lb, config[selected_dim] - 1), min(ub, config[selected_dim] + 1)})
        else:
            candidates = list(
                range(
                    max(lb, min(config[selected_dim] - 1, round(config[selected_dim] * 0.8))),
                    min(ub, max(round(config[selected_dim] * 1.2), config[selected_dim] + 1)) + 1,
                )
            )
        config_pert[selected_dim] = np.random.choice(candidates, 1)[0]
        while config_pert[selected_dim] == config[selected_dim]:
            config_pert[selected_dim] = np.random.choice(candidates, 1)[0]
    config_pert = deactivate_inactive_hyperparameters(config_pert, cs)
    x_pert = config_pert.get_array()
    if fixed_dims is not None:
        x_pert = np.concatenate([x_pert, fixed_dims])
    return x_pert


def interleaved_search(
    cs: CS.ConfigurationSpace,
    n_dim,
    x_center,
    f: Callable,
    max_dist_cont: float,
    max_dist_cat: float = None,
    cont_int_lengthscales: float = None,
    n_restart: int = 1,
    step: int = 40,
    batch_size: int = 1,
    interval: int = 1,
    dtype=torch.float,
    frozen_dims: List[int] = None,
    frozen_vals: list = None,
    num_fixed_dims: int = None,
    verbose: bool = True,
):
    """
    x_center: the previous best x location that will be the centre of the bounding box
    f: the objective function of the interleaved_search. In this case, it is usually the acquisition function.
        This objective should be minimized.
    max_dist_cont: the bounding box length of the continuous trust region
    max_dist_cat: the bounding box length of the categorical trust region. This is in terms of normalized Hamming distance >0, <=1.
    cont_int_lengthscales: the lengthscales of the learnt GP model on the continuous and integer dimensions
    n_restart: number of restarts for the acquisition function optimization.
    """
    # when a x_center with a higher dimension than that specified by he configspace object, the additional dimensions
    #   are treated as "contextual" dimensions which are fixed during acquisition optimization.
    if max_dist_cat is None:
        max_dist_cat = 1.0  # the normalized hamming distance is upper bounded by 1.
    num_fixed_dims = n_dim - len(cs) if n_dim > len(cs) else 0
    if num_fixed_dims > 0:
        fixed_dims = list(range(len(cs), n_dim))
    else:
        fixed_dims = None

    cat_dims, cont_dims, int_dims = get_dim_info(cs, cs.sample_configuration().get_array(), return_indices=True)

    if x_center is not None:
        assert x_center.shape[0] == n_dim
        x_center_fixed = deepcopy(x_center[-num_fixed_dims:]) if num_fixed_dims > 0 else None

        # generate the initially random points by perturbing slightly from the best location
        x_center_local = deepcopy(x_center)
        if frozen_dims is not None:
            x_center_local[frozen_dims] = frozen_vals  # freeze these values
        x0s = []
        lb, ub = construct_bounding_box(cs, x_center_local, max_dist_cont, cont_int_lengthscales)
        for _ in range(n_restart):
            if num_fixed_dims:
                p = get_start_point(cs, x_center_local[:-num_fixed_dims], frozen_dims=frozen_dims)
                p = np.concatenate((p, x_center_fixed))
            else:
                p = get_start_point(cs, x_center_local, frozen_dims=frozen_dims)
            x0s.append(p)
    else:
        lb, ub = np.zeros(n_dim), np.ones(n_dim)
        x0s = [cs.sample_configuration().get_array() for _ in range(n_restart)]
        x_center_fixed = None

    x0 = np.array(x0s).astype(np.float)  # otherwise error on jade

    def _interleaved_search(x0):
        x = deepcopy(x0)
        acq_x = f(x).detach().numpy()
        n_step = 0
        while n_step <= step:
            # First optimise the continuous part, freezing the categorical part
            x_tensor = torch.tensor(x, dtype=dtype).requires_grad_(True)

            optimizer = torch.optim.Adam([{"params": x_tensor}], lr=0.1)
            for _ in range(interval):
                optimizer.zero_grad()
                acq = f(x_tensor)
                acq.backward()
                # freeze the grads of the non-continuous dimensions & the fixed dims
                for n, w in enumerate(x_tensor):
                    if (
                        n not in cont_dims
                        or (fixed_dims is not None and n in fixed_dims)
                        or (frozen_dims is not None and n in frozen_dims)
                    ):
                        x_tensor.grad[n] = 0.0
                if verbose and n_step % 20 == 0:
                    logging.info(f"Acquisition optimisation: Step={n_step}: Value={x_tensor}. Acq={acq_x}.")
                optimizer.step()
                with torch.no_grad():
                    x_nan_mask = torch.isnan(x_tensor)
                    # replace the data from the optimized tensor
                    x_tensor[cont_dims] = torch.maximum(
                        torch.minimum(x_tensor[cont_dims], torch.tensor(ub[cont_dims])), torch.tensor(lb[cont_dims])
                    ).to(x_tensor.dtype)
                    # enforces the nan entries remain nan
                    x_tensor[x_nan_mask] = torch.tensor(np.nan, dtype=x_tensor.dtype)

                    # fixed dimensions should not be updated during the optimization here. Enforce the constraint below
                    if x_center_fixed is not None:
                        # the fixed dimensions are not updated according to the gradient information.
                        x_tensor[-num_fixed_dims:] = torch.tensor(x_center_fixed, dtype=dtype)
                    # print(x_tensor)

            x = x_tensor.detach().numpy().astype(x0.dtype)
            del x_tensor

            # Then freeze the continuous part and optimise the categorical part
            if len(cat_dims) + len(int_dims) > 0:
                for j in range(interval):
                    neighbours = [sample_discrete_neighbour(cs, x, frozen_dims=frozen_dims) for _ in range(10)]
                    for i, neighbour in enumerate(neighbours):
                        neighbours[i][int_dims] = np.clip(neighbour[int_dims], lb[int_dims], ub[int_dims])
                    acq_x = f(x).detach().numpy()
                    acq_neighbour = np.array([f(n).detach().numpy() for n in neighbours]).astype(np.float)
                    acq_neighbour_argmin = np.argmin(acq_neighbour)
                    acq_neighbour_min = acq_neighbour[acq_neighbour_argmin]
                    if acq_neighbour_min < acq_x:
                        x = deepcopy(neighbours[acq_neighbour_argmin])
                        acq_x = acq_neighbour_min
            n_step += interval
        return x, acq_x

    def local_search(x):
        acq = np.inf
        x = deepcopy(x)
        logging.info(f"Bounds: {lb}, {ub}")

        if x_center_fixed is not None:
            x_config = CS.Configuration(cs, vector=x[:-num_fixed_dims])
        else:
            x_config = CS.Configuration(cs, vector=x)
        for _ in range(step):
            n_config = CS.util.get_random_neighbor(x_config, seed=int(np.random.randint(10000)))
            n = n_config.get_array()
            if x_center_fixed is not None:
                # the fixed dimensions are not updated according to the gradient information.
                n = np.concatenate((n, x_center_fixed))
            n = np.clip(n, lb, ub)
            acq_ = f(n).detach().numpy()
            if acq_ < acq:
                acq = acq_
                x = n
                x_config = n_config
        return x, acq

    X, fX = [], []
    for i in range(n_restart):
        res = _interleaved_search(x0[i, :])
        X.append(res[0])
        fX.append(res[1])

    top_idices = np.argpartition(np.array(fX).flatten(), batch_size)[:batch_size]
    return (
        np.array([x for i, x in enumerate(X) if i in top_idices]).astype(np.float),
        np.array(fX).astype(np.float).flatten()[top_idices],
    )


class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        kern,
        likelihood,
        outputscale_constraint,
    ):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kern, outputscale_constraint=outputscale_constraint)

    def forward(
        self,
        x,
        x_mask=None,
    ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(
            x,
            x1_mask=x_mask,
        )
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(
    configspace: CS.ConfigurationSpace,
    train_x,
    train_y,
    use_ard,
    num_steps,
    time_varying: bool = False,
    train_t=None,
    lengthscale_scaling: float = 2.0,
    hypers={},
    noise_variance=None,
    return_hypers=False,
    verbose: bool = False,
):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    (train_x, train_y): pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    """
    from math import sqrt

    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]
    if train_t is not None:
        if not isinstance(train_t, torch.Tensor):
            train_t = torch.tensor(train_t).to(dtype=train_x.dtype)

    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.001
        noise_constraint = gpytorch.constraints.constraints.Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.02
            noise_constraint = gpytorch.constraints.constraints.Interval(1e-6, 0.05)
        else:
            noise_constraint = gpytorch.constraints.constraints.Interval(0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = gpytorch.constraints.constraints.Interval(0.01, 0.5)
    else:
        lengthscale_constraint = gpytorch.constraints.constraints.Interval(
            0.01, sqrt(train_x.shape[1])
        )  # [0.005, sqrt(dim)]
    # outputscale_constraint = Interval(0.05, 20.0)
    outputscale_constraint = gpytorch.constraints.constraints.Interval(0.5, 5.0)

    # add in temporal dimension if t is not None
    if train_t is not None and time_varying:
        train_x = torch.hstack((train_t.reshape(-1, 1), train_x))

    # Create models
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint).to(
        device=train_x.device, dtype=train_y.dtype
    )

    kern = CasmoKernel(
        cs=configspace,
        lamda=0.5,
        ard=use_ard,
        time_varying=time_varying,
        continuous_lengthscale_constraint=lengthscale_constraint,
        categorical_lengthscale_constraint=lengthscale_constraint,
        lengthscale_scaling=lengthscale_scaling,
    )

    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        kern=kern,
        outputscale_constraint=outputscale_constraint,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Initialize model hypers
    loaded_hypers = False
    # if hyperparameters are already supplied, no need to optimize GP
    if hypers is not None and len(hypers):
        try:
            model.load_state_dict(hypers)
            loaded_hypers = True
        except Exception as e:
            logging.warning(
                f"Exception={e} occurred when loading the hyperparameters of the GP. Now training from scratch!"
            )

    if not loaded_hypers:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
        hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
        model.initialize(**hypers)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.2)

        for _ in range(num_steps):
            optimizer.zero_grad()
            output = model(
                train_x,
            )
            try:
                loss = -mll(output, train_y).float()
                loss.backward()
                optimizer.step()
                if verbose and _ % 50 == 0:
                    logging.info(f"Optimising GP log-likelihood: Iter={_}, Loss={loss.detach().numpy()}")

            except Exception as e:
                print(
                    f"RuntimeError={e} occurred due to non psd covariance matrix. returning the model at last successful iter"
                )
                model.eval()
                likelihood.eval()
                return model
    # Switch to eval mode
    model.eval()
    likelihood.eval()
    if return_hypers:
        return model, model.state_dict()
    else:
        return model


class CasmoKernel(gpytorch.kernels.Kernel):
    """Implementation of the kernel in Casmopolitan"""

    has_lengthscale = True

    def __init__(
        self,
        cs: CS.ConfigurationSpace,
        lamda=0.5,
        ard=True,
        lengthscale_scaling=3.0,
        time_varying=False,
        categorical_lengthscale_constraint=None,
        continuous_lengthscale_constraint=None,
        **kwargs,
    ):
        """
        Note that the integer dimensions are treated as continuous here (but as discrete during acquisition optimization).
        No explicit wrapping of the integer dimensions are required, as the samples are generated from local search
        (which always produces a valid configuration on the integer vertices).
        """
        super().__init__(has_lengthscale=True, **kwargs)
        self.cs = cs
        self.dim = len(self.cs.get_hyperparameters())
        self.lengthscale_scaling = lengthscale_scaling
        self.continuous_lengthscale_constraint = continuous_lengthscale_constraint
        self.lamda_ = lamda
        self.ard = ard
        # extract the dim indices of the continuous dimensions (incl. integers)
        self.cont_dims = [
            i
            for i, dim in enumerate(self.cs.get_hyperparameters())
            if type(dim) in [CSH.UniformIntegerHyperparameter, CSH.UniformFloatHyperparameter]
        ]
        self.cat_dims = [
            i for i, dim in enumerate(self.cs.get_hyperparameters()) if type(dim) == CSH.CategoricalHyperparameter
        ]

        # initialise the kernels
        self.continuous_kern = ConditionalMatern(
            cs=self.cs,
            nu=2.5,
            ard_num_dims=len(self.cont_dims) if ard else None,
            lengthscale_scaling=lengthscale_scaling,
            lengthscale_constraint=continuous_lengthscale_constraint,
        )
        self.categorical_kern = ExpCategoricalOverlap(
            ard_num_dims=len(self.cat_dims) if ard else None, lengthscale_constraint=categorical_lengthscale_constraint
        )
        self.time_varying = time_varying
        self.time_kernel = TemporalKernel() if time_varying else None

    def _set_lamda(self, value):
        self.lamda_ = max(0.0, min(value, 1.0))

    @property
    def lamda(self):
        return self.lamda_

    @lamda.setter
    def lamda(self, value):
        self._set_lamda(value)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_mask=None,
        x2_mask=None,
        diag=False,
        last_dim_is_batch=False,
        **params,
    ):
        """
        todo: for now the masking is only available for the integer/continuous dimensions. This works for now as
            none of the categorical variables is currently conditional. If and when we have conditional categoricals,
            the categorical kernels need to be amended correspondingly to avoid problems.
        """
        assert x1.shape[1] >= self.dim, f"Dimension mismatch! Expected = {self.dim} but got {x1.shape[1]}"
        # it is possible for x1.shape[1] to be larger than self.dim, due to auxiliary dimensions that are not part of
        #   the active configspace but provide extra information about the search space. These are appended to the end
        #   of the vector, and the cont_dims are changed accordingly (assuming these additional dimensions are all
        #   continuous)

        # WARNING: any additional contextual information MUST be added to the END of the vector. If it is present
        #   anywhere else, the kernel may give incorrect results, WITHOUT raising an exception.
        if self.time_varying:
            x1, t1 = x1[:, 1:], x1[:, :1]
            if x2 is not None:
                x2, t2 = x2[:, 1:], x2[:, :1]
            else:
                t2 = None
        else:
            t1 = t2 = None
        if x1.shape[1] > self.dim:
            self.continuous_kern = ConditionalMatern(
                cs=self.cs,
                nu=2.5,
                ard_num_dims=x1.shape[1] if self.ard else None,
                lengthscale_scaling=self.lengthscale_scaling,
                lengthscale_constraint=self.continuous_lengthscale_constraint,
            )
            self.cont_dims += list(range(self.dim, x1.shape[1]))
            self.dim = x1.shape[1]

        if x2 is not None:
            assert x2.shape[1] == x1.shape[1]
        if t1 is not None and self.time_kernel is not None:
            assert (
                t1.shape[0] == x1.shape[0]
            ), f"Dimension mismatch between x1 {x1.shape[0]} and its timestep vector t1 {t1.shape[0]}!"
        if t2 is not None and self.time_kernel is not None:
            assert t2.shape[0] == x2.shape[0], "Dimension mismatch between x2 and its timestep vector t2!"
        if len(self.cat_dims) == 0 and len(self.cont_dims) == 0:
            raise ValueError("Zero-dimensioned problem!")
        elif len(self.cat_dims) > 0 and len(self.cont_dims) == 0:  # entirely categorical
            spatial_ker_val = self.categorical_kern.forward(x1, x2, diag=diag, **params)
        elif len(self.cont_dims) > 0 and len(self.cat_dims) == 0:  # entirely continuous
            spatial_ker_val = self.continuous_kern.forward(
                x1, x2, diag=diag, x1_mask=x1_mask, x2_mask=x2_mask, **params
            )
        else:
            # mixed case
            x1_cont, x2_cont = x1[:, self.cont_dims], x2[:, self.cont_dims]
            x1_cat, x2_cat = x1[:, self.cat_dims], x2[:, self.cat_dims]
            spatial_ker_val = (1.0 - self.lamda) * (
                self.categorical_kern.forward(x1_cat, x2_cat, diag=diag, **params)
                + self.continuous_kern.forward(x1_cont, x2_cont, x1_mask=x1_mask, x2_mask=x2_mask, diag=diag, **params)
            ) + self.lamda * self.categorical_kern.forward(
                x1_cat, x2_cat, diag=diag, **params
            ) * self.continuous_kern.forward(
                x1_cont, x2_cont, x1_mask=x1_mask, x2_mask=x2_mask, diag=diag, **params
            )

        if self.time_kernel is None or t1 is None or t2 is None:
            ker_val = spatial_ker_val
        else:  # product kernel between the temporal and spatial kernel values.
            ker_val = self.time_kernel.forward(t1, t2) * spatial_ker_val
        return ker_val


class CategoricalOverlap(gpytorch.kernels.Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        with torch.no_grad():  # discrete kernels are not differentiable. Make is explicit as such
            # First, convert one-hot to ordinal representation
            diff = x1[:, None] - x2[None, :]
            # nonzero location = different cat
            diff[torch.abs(diff) > 1e-5] = 1
            # invert, to now count same cats
            diff1 = torch.logical_not(diff).float()
            if self.ard_num_dims is not None and self.ard_num_dims > 1:
                k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
            else:
                # dividing by number of cat variables to keep this term in range [0,1]
                k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
            if diag:
                return torch.diag(k_cat).float()
            return k_cat.float()


class ExpCategoricalOverlap(CategoricalOverlap):
    """
    Exponentiated categorical overlap kernel
    $$ k(x, x') = \\exp(\frac{\\lambda}{n}) \\sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \\exp(\frac{1}{n} \\sum_{i=1}^n \\lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp="rbf", **params):
        with torch.no_grad():  # discrete kernels are not differentiable. Make is explicit as such
            diff = x1[:, None] - x2[None, :]
            diff[torch.abs(diff) > 1e-5] = 1
            diff1 = torch.logical_not(diff).float()

            def rbf(d, ard):
                if ard:
                    return torch.exp(torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
                else:
                    return torch.exp(self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

            if exp == "rbf":
                k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
            else:
                raise ValueError("Exponentiation scheme %s is not recognised!" % exp)
            if diag:
                return torch.diag(k_cat).float()
        return k_cat.float()


class L1Distance(torch.nn.Module):
    """Compute L1 distance between two input vectors"""

    def __init__(self, postprocess_script=gpytorch.kernels.kernel.default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute l1 distance
        res = (x1.unsqueeze(1) - x2.unsqueeze(0)).abs().sum(-1)
        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res


class TemporalKernel(gpytorch.kernels.Kernel):
    """Kernel function to compute L1 distance between two vectors, without a lengthscale.
    This is useful for computing the distance between the time vectors in time-varying GP
    surrogate.
    epsilon (epsilon) is the "forgetting" parameter of the time-varying GP.
    """

    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TemporalKernel, self).__init__(**kwargs)
        self.distance_module = L1Distance()
        eps_constraint = gpytorch.constraints.constraints.Interval(0.0, 1.0)
        self.register_parameter(name="raw_epsilon", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_epsilon", eps_constraint)

    @property
    def epsilon(self):
        return self.raw_epsilon_constraint.transform(self.raw_epsilon)

    @epsilon.setter
    def epsilon(self, value):
        self._set_epsilon(value)

    def _set_epsilon(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_eps)
        self.initialize(raw_eps=self.raw_eps_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        dist = self.covar_dist(x1, x2, diag=diag, **params)
        time_ker = (1.0 - self.epsilon) ** (0.5 * dist)
        time_ker_diag = torch.diag(time_ker)
        if diag:
            return time_ker_diag
        return time_ker


class ConditionalMatern(gpytorch.kernels.MaternKernel):
    has_lengthscale = True

    def __init__(self, cs: CS.ConfigurationSpace, nu=2.5, **kwargs):
        self.cs = cs
        super().__init__(nu, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        else:
            raise RuntimeError(f"nu must be in {0.5, 1.5, 2.5} but got {self.nu}!")
        return constant_component * exp_component


class HyperparameterOptimizer(ABC):
    def __init__(self, env, max_iters: int = 100, batch_size: int = 1, n_repetitions: int = 4, anneal_lr: bool = False):
        self.env = env
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.n_repetitions = n_repetitions
        self.X, self.y = [], []

    @abstractmethod
    def run(self):
        pass


class Casmo4RL(HyperparameterOptimizer):
    def __init__(
        self,
        config_space,
        log_dir,
        max_iters: int,
        max_timesteps: int,
        batch_size: int = 1,
        n_init: int = None,
        verbose: bool = True,
        ard=False,
        use_reward: float = 0.0,
        log_interval: int = 1,
        time_varying=False,
        current_timestep: int = 0,
        acq: str = "lcb",
        obj_func: Callable = None,
        seed: int = None,
        use_standard_gp: bool = False,
    ):
        """
        Casmopolitan [Wan2021] with additional support for ordinal variables.
        Args:
            env: an instance of search_spaces.SearchSpace object
            log_dir: path str: the logging directory to save results.
            max_iters: int, maximum number of BO iterations.
            max_timesteps: int, maximum RL timestep.
            batch_size: int, batch size of BO
            n_init: int, number of initializing samples (i.e. random samples)
            ard: whether to use ARD kernel.
            use_reward: bool. When non-zero, we will take the average of the final ``use_reward`` fraction of a
                reward trajectory as the BO optimization target. Otherwise we only use the final reward.
            log_interval: int. Time interval to save & report the result.
            time_varying: bool whether to use time-varying GP modelling [Bogunovic2016].
            current_timestep: current timestep. Only applicable when time_varying is True
            acq: ['lcb', 'ei']. Choice of the acquisition function.
            obj_func: Callable: the objective function handle.
            seed: random seed.
            use_standard_gp: bool. Whether to use a standard GP. Otherwise we use trust region GP in [Eriksson2019]
                 and [Wan2021].
        References:
        [Bogunovic2016]: Bogunovic, I., Scarlett, J., & Cevher, V. (2016, May). Time-varying Gaussian process bandit optimization.
            In Artificial Intelligence and Statistics (pp. 314-323). PMLR.
        [Wan2021]: Wan, X., Nguyen, V., Ha, H., Ru, B., Lu, C.,; Osborne, M. A. (2021).
            Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces.
            International Conference on Machine Learning. http://arxiv.org/abs/2102.07188
        [Eriksson2019]: Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019). Scalable global optimization via
            local bayesian optimization. Advances in neural information processing systems, 32.
        """
        super().__init__(config_space, max_iters, batch_size, 1)
        self.max_timesteps = max_timesteps
        # check whether we need to do mixed optimization by inspecting whether there are any continuous dims.
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.verbose = verbose
        self.cur_iters = 0
        self.dim = len(config_space.get_hyperparameters())
        self.log_interval = log_interval
        self.n_init = n_init if n_init is not None and n_init > 0 else min(10, 2 * self.dim + 1)

        # settings related to the time-varying GP
        self.time_varying = time_varying
        self.current_timestep = current_timestep
        self.use_standard_gp = use_standard_gp

        self.seed = self.env.seed = seed
        self.ard = ard
        self.casmo = _Casmo(
            config_space,
            n_init=self.n_init,
            max_evals=self.max_iters,
            batch_size=None,  # this will be updated later. batch_size=None signifies initialisation
            verbose=verbose,
            ard=ard,
            acq=acq,
            use_standard_gp=self.use_standard_gp,
            time_varying=time_varying,
        )
        self.X_init = None
        self.use_reward = use_reward
        # save the RL learning trajectory for each run of the BO
        self.trajectories = []
        self.f = obj_func if obj_func is not None else self._obj_func_handle

    def restart(self):
        self.casmo._restart()
        self.casmo._X = np.zeros((0, self.casmo.dim))
        self.casmo._fX = np.zeros((0, 1))
        self.X_init = np.array([self.env.sample_configuration().get_array() for _ in range(self.n_init)])

    def suggest(
        self,
        n_suggestions=1,
    ):
        if self.casmo.batch_size is None:  # Remember the batch size on the first call to suggest
            self.casmo.batch_size = n_suggestions
            self.casmo.n_init = max([self.casmo.n_init, self.batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            # Remove these pending points
            self.X_init = self.X_init[n_init:, :]

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.casmo._X) > 0:  # Use random points if we can't fit a GP
                X = deepcopy(self.casmo._X)
                fX = copula_standardize(deepcopy(self.casmo._fX).ravel())  # Use Copula
                X_next[-n_adapt:, :] = self.casmo._create_and_select_candidates(
                    X,
                    fX,
                    length_cont=self.casmo.length,
                    length_cat=self.casmo.length_cat,
                    n_training_steps=100,
                    hypers={},
                )[
                    -n_adapt:,
                    :,
                ]
        suggestions = X_next
        return suggestions

    def suggest_conditional_on_fixed_dims(self, fixed_dims, fixed_vals, n_suggestions=1):
        """Suggest points based on BO surrogate, conditional upon some fixed dims and values"""
        assert len(fixed_vals) == len(fixed_dims)
        X = deepcopy(self.casmo._X)
        fX = copula_standardize(deepcopy(self.casmo._fX).ravel())  # Use Copula
        X_next = self.casmo._create_and_select_candidates(
            X,
            fX,
            length_cont=self.casmo.length,
            length_cat=self.casmo.length_cat,
            n_training_steps=100,
            frozen_dims=fixed_dims,
            frozen_vals=fixed_vals,
            batch_size=n_suggestions,
            hypers={},
        )
        return X_next

    def observe(self, X, y, t=None):
        """Send an observation of a suggestion back to the optimizer.
        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        t: array-like, shape (n, )
            Corresponding to the timestep vector of t
        """
        assert len(X) == len(y)
        if t is not None:
            assert len(t) == len(y)
        XX = X
        yy = np.array(y)[:, None]
        tt = np.array(t)[:, None] if t is not None else None

        if len(self.casmo._fX) >= self.casmo.n_init:
            self.casmo._adjust_length(yy)

        self.casmo.n_evals += self.batch_size
        self.casmo._X = np.vstack((self.casmo._X, deepcopy(XX)))
        self.casmo._fX = np.vstack((self.casmo._fX, deepcopy(yy.reshape(-1, 1))))
        self.casmo.X = np.vstack((self.casmo.X, deepcopy(XX)))
        self.casmo.fX = np.vstack((self.casmo.fX, deepcopy(yy.reshape(-1, 1))))
        if tt is not None:
            self.casmo._t = np.vstack((self.casmo._t, deepcopy(tt.reshape(-1, 1))))
            self.casmo.t = np.vstack((self.casmo.t, deepcopy(tt.reshape(-1, 1))))

        # Check for a restart
        if self.casmo.length <= self.casmo.length_min:
            self.restart()

    def run(self):
        self.cur_iters = 0
        self.res = pd.DataFrame(
            np.nan,
            index=np.arange(self.max_iters + self.batch_size),
            columns=["Index", "LastValue", "BestValue", "Time"],
        )
        self.X, self.y = [], []
        while self.cur_iters < self.max_iters:
            logging.info(f"Current iter = {self.cur_iters + 1} / {self.max_iters}")
            start = time.time()
            suggested_config_arrays = self.suggest(self.batch_size)
            # convert suggestions from np array to a valid configuration.
            suggested_configs = [
                CS.Configuration(self.env.config_space, vector=array) for array in suggested_config_arrays
            ]
            rewards = self.f(suggested_configs)
            self.X += suggested_configs
            self.y += rewards
            if isinstance(rewards, float):
                # to give a len to a singleton reward result
                rewards = np.array(rewards).reshape(1)
            self.observe(suggested_config_arrays, rewards)
            end = time.time()
            if len(self.y):
                if self.batch_size == 1:
                    self.res.iloc[self.cur_iters, :] = [
                        self.cur_iters,
                        float(self.y[-1]),
                        float(np.min(self.y[: self.cur_iters + 1])),
                        end - start,
                    ]
                else:
                    for j in range(self.cur_iters, self.cur_iters + self.batch_size):
                        self.res.iloc[j, :] = [j, float(self.y[j]), float(np.min(self.y[: j + 1])), end - start]
                argmin = np.argmin(self.y[: self.cur_iters + 1])

                logging.info(f"fX={rewards}." f"fX_best={self.y[argmin]}")
                if self.cur_iters % self.log_interval == 0:
                    if self.log_dir is not None:
                        logging.info(f'Saving intermediate results to {os.path.join(self.log_dir, "stats.pkl")}')
                        self.res.to_csv(os.path.join(self.log_dir, "stats-pandas.csv"))
                        pickle.dump([self.X, self.y], open(os.path.join(self.log_dir, "stats.pkl"), "wb"))
                        pickle.dump(self.trajectories, open(os.path.join(self.log_dir, "trajectories.pkl"), "wb"))
            self.cur_iters += self.batch_size

        return self.X, self.y

    def _obj_func_handle(
        self,
        config: list,
    ) -> list:
        """use_synthetic: use the sklearn data generation to generate synthetic functions."""
        trajectories = self.env.train_batch(
            config,
            exp_idx_start=self.cur_iters,
            nums_timesteps=[self.max_timesteps] * len(config),
            seeds=[self.seed] * len(config),
        )
        self.trajectories += trajectories
        reward = [
            -get_reward_from_trajectory(np.array(t["y"]), use_last_fraction=self.use_reward) for t in trajectories
        ]
        return reward

    def get_surrogate(self, current_tr_only=False):
        """Return the surrogate GP fitted on all the training data"""
        if not self.casmo.fX.shape[0]:
            raise ValueError("Casmo does not currently have any observation data!")
        if current_tr_only:
            # the _X and _fX only store the data collected since the last TR restart and got cleared every time after a restart.
            X = deepcopy(self.casmo._X)
            y = deepcopy(self.casmo._fX).flatten()
        else:
            X = deepcopy(self.casmo.X)
            y = deepcopy(self.casmo.fX).flatten()

        ard = self.ard
        if len(X) < self.casmo.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.casmo.device, self.casmo.dtype
        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(y).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5
            gp = train_gp(
                configspace=self.casmo.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=ard,
                num_steps=100,
                noise_variance=None,
            )
        return gp


class _Casmo:
    """A private class adapted from the TurBO code base"""

    def __init__(
        self,
        cs: CS.ConfigurationSpace,
        n_init,
        max_evals,
        batch_size: int = None,
        verbose: bool = True,
        ard="auto",
        acq: str = "ei",
        time_varying: bool = False,
        use_standard_gp: bool = False,
        **kwargs,
    ):
        # some env parameters
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        # assert batch_size > 0 and isinstance(batch_size, int)
        if DEVICE == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"
        self.cs = cs
        self.dim = len(cs.get_hyperparameters())
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = ard

        self.acq = acq
        self.kwargs = kwargs
        self.n_init = n_init

        self.time_varying = time_varying

        # hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))
        self.n_restart = 3  # number of restarts for each acquisition optimization

        # tolerances and counters
        self.n_cand = kwargs["n_cand"] if "n_cand" in kwargs.keys() else min(100 * self.dim, 5000)
        self.use_standard_gp = use_standard_gp
        self.n_evals = 0

        if use_standard_gp:  # this in effect disables any trust region
            logging.info("Initializing a standard GP without trust region or interleaved acquisition search.")
            self.tr_multiplier = 1.0
            self.failtol = 100000
            self.succtol = 100000
            self.length_min = self.length_min_cat = -1
            self.length_max = self.length_max_cat = 100000
            self.length_init_cat = self.length_init = 100000

        else:
            self.tr_multiplier = kwargs["multiplier"] if "multiplier" in kwargs.keys() else 1.5
            self.failtol = kwargs["failtol"] if "failtol" in kwargs.keys() else 10
            self.succtol = kwargs["succtol"] if "succtol" in kwargs.keys() else 3

            # Trust region sizes for continuous/int and categorical dimension
            self.length_min = kwargs["length_min"] if "length_min" in kwargs.keys() else 0.15
            self.length_max = kwargs["length_max"] if "length_max" in kwargs.keys() else 1.0
            self.length_init = kwargs["length_init"] if "length_init" in kwargs.keys() else 0.4

            self.length_min_cat = kwargs["length_min_cat"] if "length_min_cat" in kwargs.keys() else 0.1
            self.length_max_cat = kwargs["length_max_cat"] if "length_max_cat" in kwargs.keys() else 1.0
            self.length_init_cat = kwargs["length_init_cat"] if "length_init_cat" in kwargs.keys() else 1.0

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        # timestep: in case the GP surrogate is time-varying
        self.t = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = MIN_CUDA
        self.dtype = torch.float64
        self.device = torch.device("cuda") if DEVICE == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        self._restart()

    def _restart(self):
        self._X = np.zeros((0, self.dim))
        self._fX = np.zeros((0, 1))
        self._t = np.zeros((0, 1))
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init
        self.length_cat = self.length_init_cat

    def _adjust_length(self, fX_next):
        # print(fX_next, self._fX)
        if np.min(fX_next) <= np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += self.batch_size
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += self.batch_size

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([self.tr_multiplier * self.length, self.length_max])
            self.length_cat = min(self.length_cat * self.tr_multiplier, self.length_max_cat)
            self.succcount = 0
            logging.info(f"Expanding TR length to {self.length}")
        elif self.failcount == self.failtol:  # Shrink trust region
            self.failcount = 0
            self.length_cat = max(self.length_cat / self.tr_multiplier, self.length_min_cat)
            self.length = max(self.length / self.tr_multiplier, self.length_min)
            logging.info(f"Shrinking TR length to {self.length}")

    def _create_and_select_candidates(
        self,
        X,
        fX,
        length_cat,
        length_cont,
        x_center=None,
        n_training_steps=100,
        hypers={},
        return_acq=False,
        time_varying=None,
        t=None,
        batch_size=None,
        frozen_vals: list = None,
        frozen_dims: List[int] = None,
    ):
        d = X.shape[1]
        time_varying = time_varying if time_varying is not None else self.time_varying
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_ard in [True, False]:
            ard = self.use_ard
        else:
            # turn on ARD only when there are many data
            ard = True if fX.shape[0] > 150 else False
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype
        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5
            gp = train_gp(
                configspace=self.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=ard,
                num_steps=n_training_steps,
                hypers=hypers,
                noise_variance=self.kwargs["noise_variance"] if "noise_variance" in self.kwargs else None,
                time_varying=time_varying and t is not None,
                train_t=t,
                verbose=self.verbose,
            )
            # Save state dict
            hypers = gp.state_dict()

        # we are always optimizing the acquisition function at the latest timestep
        t_center = t.max() if time_varying else None

        def _ei(X, augmented=False):
            """Expected improvement (with option to enable augmented EI).
            This implementation assumes the objective function should be MINIMIZED, and the acquisition function should
                also be MINIMIZED (hence negative sign on both the GP prediction and the acquisition function value)
            """
            from torch.distributions import Normal

            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=dtype)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            gauss = Normal(torch.zeros(1), torch.ones(1))
            # flip for minimization problems
            gp.eval()
            if time_varying:
                X = torch.hstack([t_center * torch.ones((X.shape[0], 1)), X])
            preds = gp(X)
            with gpytorch.settings.fast_pred_var():
                mean, std = -preds.mean, preds.stddev
            mu_star = -fX.min()

            u = (mean - mu_star) / std
            ucdf = gauss.cdf(u)
            updf = torch.exp(gauss.log_prob(u))
            ei = std * updf + (mean - mu_star) * ucdf
            if augmented:
                sigma_n = gp.likelihood.noise
                ei *= 1.0 - torch.sqrt(torch.clone(sigma_n)) / torch.sqrt(sigma_n + std**2)
            return -ei

        def _lcb(X, beta=3.0):
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=dtype)
            if X.dim() == 1:
                X = X.reshape(1, -1)
            if time_varying:
                X = torch.hstack([t_center * torch.ones((X.shape[0], 1)), X])
            gp.eval()
            gp.likelihood.eval()
            preds = gp.likelihood(gp(X))
            with gpytorch.settings.fast_pred_var():
                mean, std = preds.mean, preds.stddev
                lcb = mean - beta * std
            return lcb

        if batch_size == 1:
            # Sequential setting
            if self.use_standard_gp:
                X_next, acq_next = grad_search(
                    self.cs,
                    x_center[0] if x_center is not None else None,
                    eval(f"_{self.acq}"),
                    n_restart=self.n_restart,
                    batch_size=batch_size,
                    verbose=self.verbose,
                    dtype=dtype,
                )
            else:
                X_next, acq_next = interleaved_search(
                    self.cs,
                    d,
                    x_center[0] if x_center is not None else None,
                    eval(f"_{self.acq}"),
                    max_dist_cat=length_cat,
                    max_dist_cont=length_cont,
                    cont_int_lengthscales=gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel(),
                    n_restart=self.n_restart,
                    batch_size=batch_size,
                    verbose=self.verbose,
                    frozen_dims=frozen_dims,
                    frozen_vals=frozen_vals,
                    dtype=dtype,
                )
        else:
            # batch setting: for these, we use the fantasised points {x, y}
            X_next = torch.tensor([], dtype=dtype, device=device)
            acq_next = np.array([])
            for p in range(batch_size):
                x_center_ = deepcopy(x_center[0]) if x_center is not None else None
                if self.use_standard_gp:
                    x_next, acq = grad_search(
                        self.cs, x_center_, eval(f"_{self.acq}"), n_restart=self.n_restart, batch_size=1, dtype=dtype
                    )
                else:
                    x_next, acq = interleaved_search(
                        self.cs,
                        d,
                        x_center_,
                        eval(f"_{self.acq}"),
                        max_dist_cat=length_cat,
                        max_dist_cont=length_cont,
                        cont_int_lengthscales=gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel(),
                        frozen_dims=frozen_dims,
                        frozen_vals=frozen_vals,
                        n_restart=self.n_restart,
                        batch_size=1,
                        dtype=dtype,
                    )

                x_next_torch = torch.tensor(x_next, dtype=dtype, device=device)
                if time_varying:
                    # strip the time dimension
                    x_next_torch = x_next_torch[:, 1:]

                y_next = gp(x_next_torch).mean.detach()
                with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                    X_torch = torch.cat((X_torch, x_next_torch), dim=0)
                    y_torch = torch.cat((y_torch, y_next), dim=0)
                    gp = train_gp(
                        configspace=self.cs,
                        train_x=X_torch,
                        train_y=y_torch,
                        use_ard=ard,
                        num_steps=n_training_steps,
                        hypers=hypers,
                        noise_variance=self.kwargs["noise_variance"] if "noise_variance" in self.kwargs else None,
                        time_varying=self.time_varying,
                        train_t=t,
                    )
                X_next = torch.cat((X_next, x_next_torch), dim=0)
                acq_next = np.hstack((acq_next, acq))
        del X_torch, y_torch, gp
        X_next = np.array(X_next)
        if return_acq:
            return X_next, acq_next
        return X_next
