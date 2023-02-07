# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging

logging.basicConfig(level=logging.INFO)

import warnings

import ConfigSpace as CS
import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

from run_idaac import run_idaac
import omegaconf


# Target Algorithm
def run_procgen(cfg, seed, budget):
    config = omegaconf.OmegaConf.load("./configs/base.yaml")
    config.num_timesteps = int(budget)
    config.env_name = "plunder"
    for k in cfg.keys():
        config[k] = cfg[k]
    perfs = []
    for s in [0, 1, 2, 3, 4]:
        config.seed = s
        perfs.append(run_idaac(config))
    return np.mean(perfs)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    eps = UniformFloatHyperparameter(
        "eps", 0.000001, 0.01, default_value=1e-5, log=True
    )
    learning_rate = UniformFloatHyperparameter(
        "learning_rate", 0.000001, 0.01, default_value=5e-4, log=True
    )
    ppo_epoch = UniformIntegerHyperparameter("ppo_epoch", 1, 5, default_value=3)
    alpha = UniformFloatHyperparameter("alpha", 0.8, 0.9999, default_value=0.99)
    gae_lambda = UniformFloatHyperparameter(
        "gae_lambda", 0.8, 0.9999, default_value=0.95
    )
    clip_param = UniformFloatHyperparameter("clip_param", 0.0, 0.5, default_value=0.2)
    entropy_coef = UniformFloatHyperparameter(
        "entropy_coef", 0.0, 0.5, default_value=1e-2
    )
    value_loss_coef = UniformFloatHyperparameter(
        "value_loss_coef", 0.0, 1.0, default_value=0.5
    )
    adv_loss_coef = UniformFloatHyperparameter(
        "adv_loss_coef", 0.0, 1.0, default_value=0.25
    )
    order_loss_coef = UniformFloatHyperparameter(
        "order_loss_coef", 0.0, 0.1, default_value=0.001
    )
    max_grad_norm = UniformFloatHyperparameter(
        "max_grad_norm", 0.0, 1.0, default_value=0.5
    )
    value_epoch = UniformIntegerHyperparameter("value_epoch", 1, 10, default_value=9)
    value_freq = UniformIntegerHyperparameter("value_freq", 1, 5, default_value=1)
    use_nonlinear_clf = CategoricalHyperparameter(
        "use_nonlinear_clf", [True, False], default_value=False
    )

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            learning_rate,
            gae_lambda,
            eps,
            ppo_epoch,
            alpha,
            clip_param,
            entropy_coef,
            value_loss_coef,
            adv_loss_coef,
            order_loss_coef,
            max_grad_norm,
            value_epoch,
            value_freq,
            use_nonlinear_clf,
        ]
    )

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "cs": cs,  # configuration space
            "deterministic": True,
            "runcount_limit": 728,
            "output_dir": "./smac_plunder_s1",
            "limit_resources": True,
        }
    )

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 25000000

    # Intensifier parameters
    intensifier_kwargs = {
        "initial_budget": 250000,
        "max_budget": max_epochs,
        "eta": 1.9,
    }

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(1),
        tae_runner=run_procgen,
        intensifier_kwargs=intensifier_kwargs,
    )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print(f"Incumbent: {incumbent}")
