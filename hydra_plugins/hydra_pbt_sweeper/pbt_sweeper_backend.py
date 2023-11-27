# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations

from typing import List

import logging
import operator
import os
from functools import reduce

import numpy as np

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf, open_dict
from rich import print as printr

from hydra_plugins.hydra_pbt_sweeper.hydra_bgt import HydraBGT
from hydra_plugins.hydra_pbt_sweeper.hydra_pb2 import HydraPB2
from hydra_plugins.hydra_pbt_sweeper.hydra_pbt import HydraPBT
from hydra_plugins.utils.search_space_encoding import search_space_to_config_space

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class PBTSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig,
        resume: str | None = None,
        optimizer: str | None = "pbt",
        budget: int | None = None,
        budget_variable: str | None = None,
        loading_variable: str | None = None,
        saving_variable: str | None = None,
        pbt_kwargs: DictConfig | dict = {},
    ) -> None:
        """
        Backend for the PBT sweeper.
        Instantiate the sweeper with hydra and launch optimization.

        Parameters
        ----------
        search_space: DictConfig
            The search space, either a DictConfig from a hydra yaml config file,
            or a path to a json configuration space file in the format required of ConfigSpace,
            or already a ConfigurationSpace config space.
        optimizer: str
            Name of the acquisition function boil should use
        budget: int | None
            Total budget for a single population member.
            This could be e.g. the total number of steps to train a single agent.
        budget_variable: str | None
            Name of the argument controlling the budget, e.g. num_steps.
        loading_variable: str | None
            Name of the argument controlling the loading of agent parameters.
        saving_variable: str | None
            Name of the argument controlling the checkpointing.
        pbt_kwargs: DictConfig | None
            Additional PBT specific arguments. These differ between different versions of PBT.
        Returns
        -------
        None

        """
        self.search_space = search_space
        self.optimizer = optimizer
        self.budget_variable = budget_variable
        self.loading_variable = loading_variable
        self.saving_variable = saving_variable
        self.pbt_kwargs = pbt_kwargs
        self.budget = int(budget)
        self.resume = resume

        self.task_function: TaskFunction | None = None
        self.sweep_dir: str | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """
        Setup launcher.

        Parameters
        ----------
        hydra_context: HydraContext
        task_function: TaskFunction
        config: DictConfig

        Returns
        -------
        None

        """
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.task_function = task_function
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: List[str]) -> List | None:
        """
        Run PBT optimization and returns the incumbent configurations.

        Parameters
        ----------
        arguments: List[str]
            Hydra overrides for the sweep.

        Returns
        -------
        List[Configuration] | None
            Incumbent (best) configuration.

        """
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        printr("Config", self.config)
        printr("Hydra context", self.hydra_context)

        self.launcher.global_overrides = arguments
        if len(arguments) == 0:
            log.info("Sweep doesn't override default config.")
        else:
            log.info(f"Sweep overrides: {' '.join(arguments)}")

        configspace = search_space_to_config_space(
            search_space=self.search_space,
            seed=self.pbt_kwargs.get("pbt_seed", None),
        )
        np.random.seed(self.pbt_kwargs.get("pbt_seed", None))

        if self.optimizer == "pb2":
            opt_class = HydraPB2
        elif self.optimizer == "bgt":
            opt_class = HydraBGT
        elif self.optimizer == "pbt":
            opt_class = HydraPBT
        else:
            log.info("Optimizer unknown, defaulting to PBT")
            opt_class = HydraPBT

        pbt = opt_class(
            global_config=self.config,
            global_overrides=arguments,
            launcher=self.launcher,
            budget_arg_name=self.budget_variable,
            load_arg_name=self.loading_variable,
            save_arg_name=self.saving_variable,
            total_budget=self.budget,
            base_dir=self.sweep_dir,
            cs=configspace,
            **self.pbt_kwargs,
        )

        if self.resume is not None:
            pbt.load_pbt(self.resume)

        incumbent = pbt.run(verbose=True)

        final_config = self.config
        with open_dict(final_config):
            del final_config["hydra"]
        for a in arguments:
            n, v = a.split("=")
            key_parts = n.split(".")
            reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = v
        schedules = {}
        for i in range(len(incumbent)):
            for k, v in incumbent[i].items():
                if k not in schedules.keys():
                    schedules[k] = []
                schedules[k].append(v)
        for k in schedules.keys():
            key_parts = k.split(".")
            reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = schedules[k]
        with open(os.path.join(pbt.output_dir, "final_config.yaml"), "w+") as fp:
            OmegaConf.save(config=final_config, f=fp)

        return incumbent
