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

from hydra_plugins.hydra_dehb_sweeper.hydra_dehb import HydraDEHB
from hydra_plugins.utils.search_space_encoding import search_space_to_config_space

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class DEHBSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig,
        budget_variable: str | None = None,
        dehb_kwargs: DictConfig | dict = {},
        resume: str | None = None,
        n_jobs: int = 8,
        slurm: bool = False,
        slurm_timeout: int = 10,
        total_function_evaluations: int | None = None,
        total_brackets: int | None = None,
        total_cost: int | None = None,
        total_time_cost: float | None = None,
    ) -> None:
        """
        Backend for the DEHB sweeper. Instantiate and launch DEHB's optimization.

        Parameters
        ----------
        search_space: DictConfig
            The search space, either a DictConfig from a hydra yaml config file, or a path to a json configuration space
            file in the format required of ConfigSpace, or already a ConfigurationSpace config space.
        budget_variable: str | None
            Name of the variable controlling the budget, e.g. max_epochs. Only relevant for multi-fidelity methods.
        dehb_kwargs: DictConfig | None
            Keywords for DEHB
        total_function_evaluations: int | None
            Maximum number of function evaluations for the optimization. One of total_function_evaluations,
            total_brackets and total_cost must be given.
        total_brackets: int | None
            Maximum number of brackets for the optimization. One of total_function_evaluations, total_brackets and
            total_cost must be given.
        total_cost: int | None
            Total amount of seconds for the optimization (i.e. runtimes of all jobs will be summed up for this!).
            One of total_function_evaluations, total_brackets and total_cost must be given.

        Returns
        -------
        None

        """
        self.search_space = search_space
        self.dehb_kwargs = dehb_kwargs
        self.budget_variable = budget_variable

        self.fevals = total_function_evaluations
        self.brackets = total_brackets
        self.cost = total_cost
        self.time_cost = total_time_cost
        self.n_jobs = n_jobs
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
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
        Run optimization with DEHB.

        Parameters
        ----------
        arguments: List[str]
            Hydra overrides for the sweep.

        Returns
        -------
        Configuration | None
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
            seed=self.dehb_kwargs.get("dehb_seed", None),
        )
        np.random.seed(self.dehb_kwargs.get("dehb_seed", None))

        dehb = HydraDEHB(
            self.config,
            arguments,
            self.launcher,
            self.budget_variable,
            self.n_jobs,
            slurm=self.slurm,
            slurm_timeout=self.slurm_timeout,
            base_dir=self.sweep_dir,
            cs=configspace,
            f=self.task_function,
            **self.dehb_kwargs,
        )

        if self.resume is not None:
            dehb.load_dehb(self.resume)

        incumbent = dehb.run(
            fevals=self.fevals,
            brackets=self.brackets,
            total_cost=self.cost,
            total_time_cost=self.time_cost,
            verbose=True,
        )

        final_config = self.config
        with open_dict(final_config):
            del final_config["hydra"]
        log.info(final_config.keys())
        for a in arguments:
            n, v = a.split("=")
            key_parts = n.split(".")
            reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = v
        for k, v in incumbent.items():
            key_parts = k.split(".")
            reduce(operator.getitem, key_parts[:-1], final_config)[key_parts[-1]] = v
        with open(os.path.join(dehb.output_path, "final_config.yaml"), "w+") as fp:
            OmegaConf.save(config=final_config, f=fp)

        return incumbent
