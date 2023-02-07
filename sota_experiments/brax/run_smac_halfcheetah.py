# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import omegaconf
import numpy as np

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

from train_brax import train_brax


logging.basicConfig(level=logging.INFO)


# Target Algorithm
def run_brax(cfg, seed, budget):
    config = omegaconf.OmegaConf.load("./configs/base.yaml")
    config.num_timesteps = int(budget)
    config.env_name = "halfcheetah"
    for k in cfg.keys():
        config[k] = cfg[k]
    perfs = []
    for s in [0, 1, 2, 3, 4]:
        config.seed = s
        perfs.append(train_brax(config))
    return np.mean(perfs)


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    num_minibatches = UniformIntegerHyperparameter(
        "num_minibatches", 0, 7, default_value=6
    )
    batch_size = CategoricalHyperparameter(
        "batch_size", [128, 256, 512, 1024, 2048], default_value=1024
    )
    learning_rate = UniformFloatHyperparameter(
        "learning_rate", 0.000001, 0.01, default_value=3e-4, log=True
    )
    num_update_epochs = UniformIntegerHyperparameter(
        "num_update_epochs", 1, 15, default_value=4
    )
    gae_lambda = UniformFloatHyperparameter(
        "gae_lambda", 0.5, 0.9999, default_value=0.95
    )
    epsilon = UniformFloatHyperparameter("epsilon", 0.01, 0.9, default_value=0.3)
    reward_scaling = UniformFloatHyperparameter(
        "reward_scaling", 0.01, 1.0, default_value=0.1
    )
    entropy_cost = UniformFloatHyperparameter(
        "entropy_cost", 0.0001, 0.5, default_value=1e-2
    )
    vf_coef = UniformFloatHyperparameter("vf_coef", 0.01, 0.9, default_value=0.5)

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            batch_size,
            learning_rate,
            num_minibatches,
            num_update_epochs,
            gae_lambda,
            epsilon,
            reward_scaling,
            entropy_cost,
            vf_coef,
        ]
    )
    output_dir = "./smac_halfcheetah_s0"
    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "cs": cs,  # configuration space
            "deterministic": True,
            "runcount_limit": 250,
            "output_dir": output_dir,
            "limit_resources": True,
        }
    )

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 100000000

    # Intensifier parameters
    intensifier_kwargs = {
        "initial_budget": 1000000,
        "max_budget": max_epochs,
        "eta": 1.9,
    }

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(0),
        tae_runner=run_brax,
        intensifier_kwargs=intensifier_kwargs,
    )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print(f"Incumbent: {incumbent}")
