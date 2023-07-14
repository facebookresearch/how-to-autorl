# A lot of this code is adapted from the original implementation of PB2 Mix/Mult:
# https://github.com/jparkerholder/procgen_autorl
import logging
from copy import deepcopy

import GPy
import numpy as np
from ConfigSpace.hyperparameters import (
    NormalIntegerHyperparameter,
    UniformIntegerHyperparameter,
)

from hydra_plugins.hydra_pbt_sweeper.hydra_pbt import HydraPBT
from hydra_plugins.hydra_pbt_sweeper.pb2_utils import (
    UCB,
    TV_MixtureViaSumAndProduct,
    TV_SquaredExp,
    exp3_get_cat,
    normalize,
    optimize_acq,
    standardize,
)

log = logging.getLogger(__name__)


class HydraPB2(HydraPBT):
    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        budget_arg_name,
        load_arg_name,
        save_arg_name,
        total_budget,
        cs,
        seeds=None,
        slurm=False,
        slurm_timeout=10,
        init_size=8,
        base_dir=False,
        population_size=64,
        config_interval=None,
        num_config_changes=None,
        quantiles=0.25,
        resample_probability=0.25,
        perturbation_factors=[1.2, 0.8],
        categorical_mutation="mix",
        warmstart=False,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=["pbt"],
        deepcave=False,
        maximize=False,
    ):
        """
        PB2: PBT but with a BO backend.
        This implemtation covers the BO supported selection of both continuous and categorical hyperparameters.

        Parameters
        ----------
        launcher: HydraLauncher
            A hydra launcher (usually either for local runs or slurm)
        budget_arg_name: str
            Name of the argument controlling the budget, e.g. num_steps.
        loading_arg_name: str
            Name of the argument controlling the loading of agent parameters.
        saving_arg_name: str
            Name of the argument controlling the checkpointing.
        total_budget: int
            Total budget for a single population member.
            This could be e.g. the total number of steps to train a single agent.
        cs: ConfigSpace
            Configspace object containing the hyperparameter search space.
        seeds: List[int] | False
            If not False, optimization will be run and averaged across the given seeds.
        model_based: bool
            Whether a model-based backend (such as BO) is used. Should always be false if using default PBT.
        base_dir: str | None
            Directory for logs.
        population_size: int
            Number of agents in the population.
        config_interval: int | None
            Number of steps before new configuration is chosen. Either this or num_config_changes must be given.
        num_config_changes: int | None
            Total number of times the configuration is changed. Either this or config_interval must be given.
        quantiles: float
            Upper/lower performance percentages beyond which agents are replaced.
            Lower numbers correspond to more exploration, higher ones to more exploitation.
        resample_probability: float
            Probability of a hyperparameter being resampled.
        perturbation_factors: List[int]
            Hyperparamters are multiplied with the first factor when their value is
            increased and with the second if their value is decreased.
        categorical_mutation: bool
            Decides how to handle data selection when finding new hyperparameters.'mult' divides the data
            for different categorical values, 'mix' instead gives the categoricals as additional information.
        Returns
        -------
        None
        """
        super().__init__(
            global_config=global_config,
            global_overrides=global_overrides,
            launcher=launcher,
            budget_arg_name=budget_arg_name,
            load_arg_name=load_arg_name,
            save_arg_name=save_arg_name,
            total_budget=total_budget,
            cs=cs,
            model_based=True,
            base_dir=base_dir,
            population_size=population_size,
            config_interval=config_interval,
            num_config_changes=num_config_changes,
            quantiles=quantiles,
            resample_probability=resample_probability,
            perturbation_factors=perturbation_factors,
            categorical_fixed=False,
            seeds=seeds,
            slurm=slurm,
            slurm_timeout=slurm_timeout,
            init_size=init_size,
            warmstart=warmstart,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
            deepcave=deepcave,
            maximize=maximize,
        )
        self.categorical_mutation = categorical_mutation
        self.hierarchical_config = len(self.configspace.get_all_conditional_hyperparameters()) > 0
        self.hp_bounds = np.array(
            [
                [
                    self.configspace[n].lower,
                    self.configspace[n].upper,
                ]
                for n in list(self.configspace.keys())
                if n not in self.categorical_hps
            ]
        )
        self.X = None

    def get_categoricals(self, config):
        """
        Get categorical hyperparameter values.

        Parameters
        ----------
        config: Configuration
            A configuration

        Returns
        -------
        Configuration
            The configuration with new categorical values.
        """
        cats = []
        for i, n in enumerate(self.categorical_hps):
            choices = self.configspace[n].choices
            if self.iteration <= 1 or not all([y > 0 for y in self.ys]):
                exp3_ys = self.ys[..., np.newaxis]
            else:
                exp3_ys = normalize(self.ys, self.ys)[..., np.newaxis]
            exp_xs = np.concatenate((self.fixed, self.cat_values, exp3_ys), axis=1)
            cat = exp3_get_cat(choices, exp_xs, self.num_config_changes, i + 2, self.population_size)
            config[n] = cat
            cats.append(cat)
        self.cat_current.append(cats)
        return config

    def get_continuous(self, config, performance, X, y):
        """
        Get continuous hyperparamters from GP.

        Parameters
        ----------
        performance: float
            A performance value
        config: Configuration
            A configuration
        X: List
            Current historical data
        y: List
            Historical performance values

        Returns
        -------
        Configuration]
            The configuration with new continuous values.
        """
        if len(self.current) == 0:
            m1 = deepcopy(self.m)
            cat_locs = [len(self.X[0]) - x - 1 for x in reversed(range(len(self.cat_values[0])))]
        else:
            # add the current trials to the dataset
            current_use = normalize(self.current, self.hp_bounds.T)
            if self.categorical_mutation == "mix" and len(self.categorical_hps) > 0:
                current_use = np.concatenate((current_use, self.cat_current[:-1]), axis=1)
            padding = np.array(
                [[len(self.history[0]["performances"]), performance] for _ in range(current_use.shape[0])]
            )
            best_current_id = np.argmin([self.history[i]["performances"][-1] for i in range(self.population_size)])
            max_perf = min(self.history[best_current_id]["performances"][-1], performance)
            padding = normalize(padding, [len(self.history[0]["performances"]), max_perf])
            current_use = np.hstack((padding, current_use))  # [:, np.newaxis]))
            current_use[current_use <= 0] = 0.01
            Xnew = np.hstack((self.X.T, current_use.T)).T

            # y value doesn't matter, only care about the variance.
            ypad = np.zeros(current_use.shape[0])
            ypad = ypad.reshape(-1, 1)
            if min(y) != max(y):
                y = normalize(y, [min(y), max(y)])
            ynew = np.vstack((y, ypad))
            ynew[ynew <= 0] = 0.01

            if self.categorical_mutation == "mix" and len(self.categorical_hps) > 0:
                cat_locs = [len(self.X[0]) - x - 1 for x in reversed(range(len(self.cat_values[0])))]
                kernel = TV_MixtureViaSumAndProduct(
                    self.X.shape[1],
                    variance_1=1.0,
                    variance_2=1.0,
                    variance_mix=1.0,
                    lengthscale=1.0,
                    epsilon_1=0.0,
                    epsilon_2=0.0,
                    mix=0.5,
                    cat_dims=cat_locs,
                )
            else:
                cat_locs = []
                kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)
            Xnew[Xnew >= 0.99] = 0.99
            Xnew[Xnew <= 0.01] = 0.01
            ynew[ynew >= 0.99] = 0.99
            ynew[ynew <= 0.01] = 0.01
            m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
            m1.optimize()

        xt = optimize_acq(UCB, self.m, m1, self.fixed, len(self.fixed[0]))
        # convert back...
        if self.categorical_mutation == "mix":
            try:
                cats = [xt[cat_locs]]
            except:
                cat_locs = np.array(cat_locs) - self.fixed.shape[1]
                cats = [xt[cat_locs]]
            xt = np.delete(xt, cat_locs)
            if len(xt) > len(self.continuous_hps):
                xt = xt[self.fixed.shape[1] :]
        else:
            cats = self.cat_current

        xt = xt * (np.max(self.hp_bounds.T, axis=0) - np.min(self.hp_bounds.T, axis=0)) + np.min(
            self.hp_bounds.T, axis=0
        )
        xt = xt.astype(np.float32)

        all_hps = [len(self.history[0]["performances"]), performance]
        xt_ind = 0
        cat_ind = 0
        for n in list(config.keys()):
            if n in self.continuous_hps:
                all_hps.append(xt[xt_ind])
                xt_ind += 1
            else:
                all_hps.append(cats[0][cat_ind])
                cat_ind += 1

        curr = []
        for v, n in zip(xt, self.continuous_hps):
            hp = self.configspace[n]
            if isinstance(hp, UniformIntegerHyperparameter) or isinstance(hp, NormalIntegerHyperparameter):
                v = int(v)
            else:
                v = float(v)
            config[n] = max(hp.lower, min(v, hp.upper))
            curr.append(max(hp.lower, min(v, hp.upper)))
        self.current.append(curr)
        return config

    def perturb_hps(self, config, performance, _, is_good):
        """
        Suggest next configuration.

        Parameters
        ----------
        performance: List[float]
            A list of the latest agent performances
        config: List[Configuration]
            A list of the recent configs
        _: not relevant here
        is_good: bool
            does this config belong to the best quantile

        Returns
        -------
        Configuration
            The next configuration.
        """
        if is_good:
            return config
        if self.categorical_mutation == "mult":
            for i, n in enumerate(self.categorical_hps):
                config[n] = self.cat_current[0][i]
        else:
            config = self.get_categoricals(config)
        if len(self.continuous_hps) > 0:
            config = self.get_continuous(config, performance, self.X, self.y)
        return config

    def get_model_data(self, performances=None, configs=None):
        """
        Parse history for relevant data.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        if self.categorical_mutation == "mult":
            # We get the categoricals based on all performance data, then filter everything after
            ys = []
            tps = []
            self.cat_values = []
            for i in reversed(range(1000 // self.population_size)):
                for j in range(self.population_size):
                    t = len(self.history[j]["performances"]) - i
                    if t - i <= i:
                        continue
                    p = self.history[j]["performances"][-i]
                    ys.append(self.history[j]["performances"][-i - 1] - self.history[j]["performances"][-i])
                    tps.append([t, p])
                    config = self.history[j]["configs"][-i]
                    cat = [v for v, n in zip(list(config.values()), list(config.keys())) if n in self.categorical_hps]
                    self.cat_values.append(cat)
            self.ys = np.array(ys)
            self.fixed = normalize(tps, [len(self.history[j]["performances"]), max(performances)])
            self.cat_current = []
            self.get_categoricals(configs[0])
            # Now filter data to user for the continuous variables
            data = deepcopy(self.history)
            to_keep = [[] for _ in range(self.population_size)]
            for i in range(self.population_size):
                for j in range(len(data[i]["configs"])):
                    cvs = [
                        v
                        for v, n in zip(data[i]["configs"][j].values(), data[i]["configs"][j].keys())
                        if n in self.categorical_hps
                    ]
                    if all([old == new for old, new in zip(cvs, self.cat_current[0])]):
                        to_keep[i].append(j)
                for k in data[i].keys():
                    if k == "overwritten":
                        continue
                    data[i][k] = [data[i][k][ind] for ind in to_keep[i]]
        else:
            data = self.history

        all_hps = []
        hp_values = []
        self.cat_values = []
        ts = []
        ps = []
        ys = []
        for i in reversed(range(1000 // self.population_size)):
            for j in range(self.population_size):
                t = len(data[j]["performances"]) - i
                if t <= 0:
                    continue
                ts.append(t)
                p = data[j]["performances"][-i]
                ys.append(data[j]["performances"][-i - 1] - p)
                config = data[j]["configs"][-i]
                hps = [v for v, n in zip(list(config.values()), list(config.keys())) if n in self.continuous_hps]
                cat = [v for v, n in zip(list(config.values()), list(config.keys())) if n in self.categorical_hps]
                all_hp = [v for v in list(config.values())]
                all_hps.append(all_hp)
                self.cat_values.append(cat)
                hp_values.append(hps)
                ps.append(p)

        # current_best_values = list(current_best[-1].values())
        self.ts = np.array(ts)
        self.hp_values = np.array(hp_values)
        self.all_hps = np.array(all_hps)
        self.ys = np.array(ys)

        if len(self.continuous_hps) > 0:
            self.X = normalize(self.hp_values, self.hp_bounds.T)
        else:
            self.X = self.hp_values
        self.y = standardize(self.ys).reshape(self.ys.size, 1)
        # If all values are the same, don't normalize to avoid nans, instead just cap.
        # This probably only happens if improvement is 0 for all anyway.
        if not min(self.y) == max(self.y):
            self.y = normalize(self.y, [min(self.y), max(self.y)])

        best_current_id = np.argmin([self.history[i]["performances"][-1] for i in range(self.population_size)])
        max_perf = min(self.history[best_current_id]["performances"][-1], min(performances))
        min_perf = max(
            np.max(np.stack([self.history[i]["performances"] for i in range(self.population_size)])), max(performances)
        )
        self.ts = normalize(self.ts, [0, len(self.history[0]["performances"])])
        ps = normalize(ps, [min_perf, max_perf])
        self.fixed = np.array([[t, p] for t, p in zip(self.ts, ps)])
        if self.X is not None:
            self.X = np.concatenate((self.fixed, self.X), axis=1)
        else:
            self.X = self.fixed

        self.X[self.X <= 0] = 0.01
        self.X[self.X >= 1] = 0.99

    def fit_model(self, performances, configs):
        """
        Fit the GP with current data.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        self.get_model_data(performances, configs)

        if self.categorical_mutation == "mix" and len(self.categorical_hps):
            self.X = np.concatenate((self.X, self.cat_values), axis=1)
            # self.fixed = np.concatenate((self.fixed, self.cat_values), axis=1)
            cat_locs = [len(self.X[0]) - x - 1 for x in reversed(range(len(self.cat_values[0])))]

            kernel = TV_MixtureViaSumAndProduct(
                self.X.shape[1],
                variance_1=1.0,
                variance_2=1.0,
                variance_mix=1.0,
                lengthscale=1.0,
                epsilon_1=0.0,
                epsilon_2=0.0,
                mix=0.5,
                cat_dims=cat_locs,
            )
        else:
            kernel = TV_SquaredExp(input_dim=self.X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)

        self.X = np.nan_to_num(self.X)
        self.y = np.nan_to_num(self.y)
        self.X[self.X <= 0.01] = 0.01
        self.X[self.X >= 0.99] = 0.99
        self.y[self.y <= 0.01] = 0.001
        self.y[self.y >= 0.99] = 0.99

        try:
            self.m = GPy.models.GPRegression(self.X, self.y, kernel)
            self.m.optimize()
        except np.linalg.LinAlgError:
            # add diagonal ** we would ideally make this something more robust...
            self.X += np.ones(self.X.shape) * 1e-3
            self.X[self.X <= 0.01] = 0.01
            self.X[self.X >= 0.99] = 0.99
            self.y[self.y <= 0.01] = 0.01
            self.y[self.y >= 0.99] = 0.99
            self.m = GPy.models.GPRegression(self.X, self.y, kernel)
            self.m.optimize()

        self.m.kern.lengthscale.fix(self.m.kern.lengthscale.clip(1e-5, 1))
        self.current = []
        if not self.categorical_mutation == "mult":
            self.cat_current = []
