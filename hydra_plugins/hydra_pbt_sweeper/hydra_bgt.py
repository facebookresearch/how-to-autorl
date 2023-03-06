# A lot of this code is adapted from the original BG-PBT: https://github.com/xingchenwan/bgpbt
import os
import time
import logging
import shutil
import wandb
import torch
import random
import numpy as np
from copy import deepcopy
import ConfigSpace as CS

from hydra_plugins.hydra_pbt_sweeper.hydra_pb2 import HydraPB2
from hydra_plugins.hydra_pbt_sweeper.bgt_utils import (
    normalize,
    copula_standardize,
    train_gp,
    Casmo4RL,
    MAX_CHOLESKY_SIZE,
    MIN_CUDA,
    _Casmo,
)

from hydra_plugins.hydra_pbt_sweeper.pb2_utils import standardize
from hydra_plugins.utils.lazy_imports import lazy_import

gpytorch = lazy_import("gpytorch")

log = logging.getLogger(__name__)


class HydraBGT(HydraPB2):
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
        init_policy="bo",
        seeds=None,
        slurm=False,
        slurm_timeout=10,
        init_size=16,
        patience=15,
        restart_every=False,
        base_dir=False,
        population_size=64,
        config_interval=None,
        num_config_changes=None,
        quantiles=0.25,
        resample_probability=0.25,
        perturbation_factors=[1.2, 0.8],
        distill_arg_name=None,
        replace_arg_name=None,
        distill_every=False,
        distill_timesteps=5e5,
        time_varying=False,
        acq="lcb",
        ard=False,
        use_standard_gp=False,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=["pbt"],
        deepcave=False,
        maximize=False,
    ):
        """
        Bayesian Generational Training - PBT with BO backend, GP restarts and
        architecture optimization through distillation.


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
            Total budget for a single population member. This could be e.g.
            the total number of steps to train a single agent.
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
            Hyperparamters are multiplied with the first factor when their value is increased
            and with the second if their value is decreased.
        distill_arg_name: str
            Name of the argument controlling when distillation happens.
        replace_arg_name: str
            Name of the argument controlling when teacher agents are replaced by the student.
        distill_every: int
            Number of iterations between which distillation should happen.
        time_varying: bool
            Whether to use time-varying GP.
        acq: str
            Name of the acquisition function to use.
        ard: bool
            Whether to use length scaling.
        use_standard_gp: bool
            Whether to use standard GP or a trust region.
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
            base_dir=base_dir,
            population_size=population_size,
            config_interval=config_interval,
            num_config_changes=num_config_changes,
            quantiles=quantiles,
            resample_probability=resample_probability,
            perturbation_factors=perturbation_factors,
            categorical_mutation="mix",
            seeds=seeds,
            slurm=slurm,
            slurm_timeout=slurm_timeout,
            init_size=init_size,
            warmstart=True,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
            deepcave=deepcave,
            maximize=maximize,
        )
        self.guided_restart = False
        self.n_distills = 0
        self.n_restarts = 0
        self.n_fail = 0
        self.patience = patience
        self.best_init = 0
        if not restart_every:
            self.restart_every = np.inf
        else:
            self.restart_every = restart_every
        self.backtrack = False
        self.best_cost = np.inf
        self.best_checkpoint_dir = os.path.join(self.output_dir, "best_agents")

        # We need to remove the architecture hps from the normal search space
        # because this would cause dimensionality issues when loading checkpoints
        self.nas_hps = [n for n in self.configspace.keys() if "NAS".casefold() in n.casefold()]
        self.nas_defaults = []
        for n in self.nas_hps:
            hp = self.configspace.get(n)
            if hasattr(hp, "default_value"):
                self.nas_defaults.append(hp.default_value)
            else:
                self.nas_defaults.append(hp.rvs())
        self.categorical_hps = [n for n in self.categorical_hps if n not in self.nas_hps]
        self.continuous_hps = [n for n in self.continuous_hps if n not in self.nas_hps]
        self.full_configspace = deepcopy(self.configspace)
        self.nas_configspace = CS.ConfigurationSpace(name="nas_space")
        for n in self.nas_hps:
            hp = deepcopy(self.full_configspace.get(n))
            self.nas_configspace.add_hyperparameter(hp)
        self.configspace = CS.ConfigurationSpace(name="hpo_space")
        for n in self.categorical_hps + self.continuous_hps:
            hp = deepcopy(self.full_configspace.get(n))
            self.configspace.add_hyperparameter(hp)

        self.hp_bounds = np.array(
            [
                [
                    self.configspace.get_hyperparameter(n).lower,
                    self.configspace.get_hyperparameter(n).upper,
                ]
                for n in list(self.configspace.keys())
                if n not in self.categorical_hps
            ]
        )

        self.archive = {}
        self.init_data = {}
        self.init_policy = init_policy

        self.distill_arg_name = distill_arg_name
        if not distill_every:
            self.distill_every = np.inf
        else:
            self.distill_every = distill_every
        self.n_distillation_timesteps = distill_timesteps
        self.last_distill_iter = 0
        self.last_restart = 0
        self.distill = False
        self.replace_arg_name = replace_arg_name
        self.best_archs = None

        self.casmo = _Casmo(
            self.configspace,
            n_init=self.init_size,
            max_evals=self.num_config_changes,
            batch_size=None,  # this will be updated later. batch_size=None signifies initialisation
            verbose=False,
            ard=ard,
            acq=acq,
            use_standard_gp=use_standard_gp,
            time_varying=time_varying,
        )

        for i in range(self.population_size):
            self.history[i]["config_source"] = []
            self.history[i]["nas_hps"] = []

    def _set_state(self, state):
        super()._set_state(state)
        self.archive = state["archive"]
        self.init_data = state["init_data"]
        self.n_distills = state["n_distills"]
        self.n_fail = state["n_fail"]
        self.best_init = state["best_init"]

    def _get_state(self):
        state = super()._get_state()
        state["archive"] = self.archive
        state["init_data"] = self.init_data
        state["n_distills"] = self.n_distills
        state["n_fail"] = self.n_fail
        state["best_init"] = self.best_init
        return state

    def get_bo_init_points(self):
        """
        Get initial points using BO with distillation.
        This is only called after restarting.

        Returns
        -------
        List[Tuple]
            Overrides for the hydra launcher with the config values as well as saving and loading
        List[Configuration]
            The initial configurations.
        """
        # whether to constrain the data to the current n_distills
        data = self.archive[self.n_distills - 1]  # get the data from the previous stage #
        # the configuration array
        # data[["x{}".format(i) for i in range(len(self.df.conf[0]))]] = pd.DataFrame(
        #     self.df.conf.tolist(), index=self.df.index
        # )

        nas_configs = [
            (data[i]["nas_hps"][j], data[i]["performances"][j])
            for i in range(len(data))
            for j in range(len(data[i]["performances"]))
        ]
        # performances = [
        #     [data[i]["performances"][j] for j in range(len(data[i]["performances"]))] for i in range(len(data))
        # ]
        # sort by architectures and get best performances
        # best_perf_each_arch = archs.groupby(["x{}".format(i) for i in self.nas_dims]).min().reset_index()
        best_perf_each_arch = {}
        best_arch = None
        best_perf = 0
        for n in nas_configs:
            if ",".join([str(v) for v in n[0]]) in best_perf_each_arch.keys():
                if best_perf_each_arch[",".join([str(v) for v in n[0]])] < n[1]:
                    best_perf_each_arch[",".join([str(v) for v in n[0]])] = n[1]
            else:
                best_perf_each_arch[",".join([str(v) for v in n[0]])] = n[1]
            if n[1] < best_perf:
                best_perf = n[1]
                best_arch = np.array(n[0])

        # the last column is the return
        # best_arch = best_perf_each_arch[best_perf_each_arch.R == best_perf_each_arch.R.min()].iloc[0, :-1].values
        init_bo = Casmo4RL(
            config_space=self.nas_configspace, log_dir=self.output_dir, max_iters=100, max_timesteps=self.total_budget
        )  # dummy value
        # TODO: do these need to be pandas dfs?
        init_bo._X = [n[0] for n in nas_configs]
        init_bo._fX = best_perf_each_arch.values()
        # Fill half of the population with BO...
        suggested_archs = init_bo.suggest(n_suggestions=max(1, self.init_size // 2))
        # Constrain samples to configspace
        for i in range(len(suggested_archs)):
            for j in range(len(suggested_archs[0])):
                hp = self.full_configspace.get(self.nas_hps[j])
                if type(hp).__name__ in [
                    "UniformIntegerHyperparameter",
                    "BetaIntegerHyperparameter",
                    "NormalIntegerHyperparameter",
                ]:
                    suggested_archs[i][j] = int(min(max(round(suggested_archs[i][j]), hp.lower), hp.upper))
                else:
                    suggested_archs[i][j] = min(max(suggested_archs[i][j], hp.lower), hp.upper)
        # and the rest with randomly sampled archs
        len_suggested_archs = len(suggested_archs)
        if self.best_archs is None:
            self.best_archs = best_arch.reshape(1, -1)
        else:
            self.best_archs = np.unique(np.concatenate((self.best_archs, best_arch.reshape(1, -1))), axis=0)

            # also add the best configs at each of the previous n_distill into the pool of suggested archs
        suggested_archs = np.concatenate((suggested_archs, self.best_archs))
        if len_suggested_archs < self.init_size:
            for _ in range(len_suggested_archs, self.init_size + 1):
                random_arch = self.nas_configspace.sample_configuration()
                random_arch = np.array([random_arch[n] for n in self.nas_hps]).reshape(1, -1)
                suggested_archs = np.concatenate((suggested_archs, random_arch))

        init_configs = []
        overrides = []
        for i, suggested_arch in enumerate(suggested_archs):
            if i < len_suggested_archs:
                # for distillation, we use the default hyperparams for the HPO dimensions.
                hist_cfg = deepcopy(data[random.choice(np.arange(self.population_size))]["configs"][-1])
                init_configs.append(hist_cfg)
                init_config = hist_cfg.get_array()
            else:
                sample_cfg = self.configspace.sample_configuration()
                init_configs.append(sample_cfg)
                init_config = sample_cfg.get_array()
            init_config = np.concatenate((init_config, suggested_arch))
            config = CS.Configuration(self.full_configspace, vector=init_config)
            names = list(config.keys()) + [self.budget_arg_name] + [self.save_arg_name]
            if self.seeds:
                for s in self.seeds:
                    values = (
                        list(config.values())
                        + [self.config_interval]
                        + [os.path.join(f"{self.checkpoint_dir}", f"InitConfig{i}_Stage{self.n_distills}_s{s}.pt")]
                    )
                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                    )
                    overrides.append(job_overrides)
            else:
                values = (
                    list(config.values())
                    + [self.config_interval]
                    + [os.path.join(f"{self.checkpoint_dir}", f"InitConfig{i}_Stage{self.n_distills}.pt")]
                )
                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}" for name, val in zip(names, values)
                )
                overrides.append(job_overrides)
        return init_configs, overrides, suggested_archs

    def distillation_step(self, init_configs, architectures, best_agents):
        """
        Distill architectures and replace teachers with best students.

        Parameters
        ----------
        init_configs: List[float]
            Configs for the students
        best_agent: List[Configuration]
            Agents to learn from

        Returns
        -------
        List[Configuration]
            The configurations for the next iteration.
        """
        teacher_configs, teacher_ckpts, teacher_archs = [], [], []
        for i in range(len(init_configs)):
            best_agent = np.random.choice(best_agents)
            teacher_ckpt = os.path.join(
                f"{self.checkpoint_dir}", f"model_iteration_{self.iteration}_id_{best_agent}.pt"
            )
            teacher_configs.append(self.archive[max(self.archive.keys())][best_agent]["configs"][-1])
            teacher_archs.append(self.archive[max(self.archive.keys())][best_agent]["nas_hps"][-1])
            teacher_ckpts.append(teacher_ckpt)
        log.info("Running distillation at restart.")
        best_configs_for_distill = deepcopy(init_configs)

        # here we run successive halving to determine the top-'self.pop_size' configs for the next stage.
        distill_ckpts = deepcopy(teacher_ckpts)
        s = int(np.ceil(np.log(len(init_configs)) / np.log(self.population_size)))
        eta = 2.0  # halving by default -- set anything above 2 for more aggressive elimination
        distill_timestep = 0

        elapsed_timestep = [0] * len(teacher_configs)
        for rung in range(s):
            if rung < s - 1:
                timesteps_this_rung = int(
                    self.n_distillation_timesteps * eta ** (rung - s)
                )  # see SuccessiveHalving paper
            else:
                timesteps_this_rung = int(
                    self.n_distillation_timesteps - distill_timestep
                )  # for the final rung, simply use up a
            log.info(
                f"Running SuccessiveHalving Rung {rung + 1}/{s}. Budgeted timesteps: {timesteps_this_rung}. "
                f"Number of configs surviving in this rung: {len(best_configs_for_distill)}"
            )

            overrides = []
            for i in range(len(best_configs_for_distill)):
                # We simply override the teacher checkpoint with the student checkpoint here
                student_keys = [f"+{n}_student" for n in list(best_configs_for_distill[i].keys())]
                student_nas_keys = [f"+{n}_student" for n in self.nas_hps]
                teacher_keys = [n for n in list(teacher_configs[i].keys())]
                teacher_nas_keys = self.nas_hps
                names = (
                    student_keys
                    + student_nas_keys
                    + teacher_keys
                    + teacher_nas_keys
                    + [self.load_arg_name]
                    + [self.save_arg_name]
                    + [self.replace_arg_name]
                    + [self.budget_arg_name]
                    + [self.distill_arg_name]
                )
                values = (
                    list(best_configs_for_distill[i].values())
                    + list(architectures[i % len(architectures)])
                    + list(teacher_configs[i].values())
                    + teacher_archs[i]
                    + [teacher_ckpts[i]]
                    + [teacher_ckpts[i]]
                    + [rung >= s - 1]
                    + [timesteps_this_rung]
                    + [True]
                )
                overrides.append(
                    tuple(self.global_overrides) + tuple(f"{name}={val}" for name, val in zip(names, values))
                )

            perf, _ = self.run_configs(overrides)
            costs = -np.array(perf)
            distill_timestep = timesteps_this_rung
            ranked_reward_indices = np.argsort(costs)
            survived_agent_indices = ranked_reward_indices[: max(self.population_size, int(round(len(costs) / eta)))]
            best_configs_for_distill = [best_configs_for_distill[j] for j in survived_agent_indices]
            architectures = [architectures[j] for j in survived_agent_indices]
            teacher_configs = [teacher_configs[j] for j in survived_agent_indices]
            distill_ckpts = [distill_ckpts[j] for j in survived_agent_indices]
            elapsed_timestep = [elapsed_timestep[j] for j in survived_agent_indices]
            log.info(f"Surviving agents: {survived_agent_indices}. ")

        # update self.pop with the new configs
        # the NAS dimensions of the new configs come from the previously identified best archs for this iteration,
        # and the HPO dimensions are a mix of randomly sampled and perturbed configs from the best config's HPO
        # dimensions before distillation.
        top_configs = []
        best_archs = []
        # current_t = self.df.t.max() + 1
        for idx in range(self.population_size):
            log.info(f"Assigning student config {idx % len(best_configs_for_distill)} to Agent {idx}")
            top_configs.append(best_configs_for_distill[idx % len(best_configs_for_distill)])
            best_archs.append(architectures[idx % len(best_configs_for_distill)])
        return top_configs, best_archs

    def record_iteration(self, performances, configs):
        """
        Add current iteration to history.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs
        """
        for i in range(self.population_size):
            self.history[i]["configs"].append(configs[i])
            self.history[i]["performances"].append(performances[i])
            # The architecture only ever changes in the init phase
            if len(self.history[i]["nas_hps"]) < len(self.history[i]["performances"]):
                self.history[i]["nas_hps"].append(self.history[i]["nas_hps"][-1])
        self.current_steps += self.config_interval
        self.iteration += 1
        with open(os.path.join(self.output_dir, "pbt.log"), "a+") as f:
            f.write(f"Generation {self.iteration}/{self.num_config_changes+1}\n")
            perf = [str(np.round(p, decimals=2)) for p in performances]
            perf = " | ".join(perf) + "\n"
            ids = "  |   ".join([str(n) for n in np.arange(self.population_size)]) + "\n"
            separator = "".join(["-" for _ in range(int(len(ids) * 1.2))]) + "\n"
            f.write(ids)
            f.write(separator)
            f.write(perf)
            f.write("\n")
            f.write("\n")

        if self.wandb_project:
            stats = {}
            stats["iteration"] = self.iteration
            stats["optimization_time"] = time.time() - self.start
            stats["incumbent_performance"] = -min(performances)
            for i in range(self.population_size):
                stats[f"performance_{i}"] = -performances[i]
                for n in configs[0].keys():
                    stats[f"config_{i}_{n}"] = configs[i].get(n)
            stats["num_steps"] = self.iteration * self.config_interval
            best_config = configs[np.argmin(performances)]
            for n in best_config.keys():
                stats[f"incumbent_{n}"] = best_config.get(n)
            wandb.log(stats)

    def get_initial_configs(self, best_agents=None):
        """
        Search for a good initialization by doing end-to-end (i.e. non-population based) BO for a short timeframe.
        Used at init and when restarting kernel.

        Parameters
        ----------
        best_agents: List[float] | None
            A list of the best current agents (for restarts with distillation).

        Returns
        -------
        List[Tuple]
            Overrides for the hydra launcher with the config values as well as saving and loading
        List[Configuration]
            The new initial configurations.
        """
        # In case checkpoint after init but before first iteration is loaded
        if self.current_steps == 0 and 0 in self.init_data.keys():
            log.info("Using loaded init points at 0 distills")
            init_configs = self.init_data[0]["configs"]
            init_results = self.init_data[0]["performances"]
        else:
            log.info(f"Sampling intial points at {self.n_distills} distills")
            self.init_idx = 0
            if self.init_policy == "random" or self.n_distills == 0:
                init_overrides = []
                init_configs = []
                for i in range(self.init_size):
                    config = self.configspace.sample_configuration()
                    names = list(config.keys()) + [self.budget_arg_name] + [self.save_arg_name]
                    init_configs.append(config)
                    if self.seeds:
                        for s in self.seeds:
                            values = (
                                list(config.values())
                                + [self.total_budget]
                                + [
                                    os.path.join(
                                        f"{self.checkpoint_dir}", f"InitConfig{i}_Stage{self.n_distills}_s{s}.pt"
                                    )
                                ]
                            )
                            job_overrides = tuple(self.global_overrides) + tuple(
                                f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                            )
                            init_overrides.append(job_overrides)
                    else:
                        values = (
                            list(config.values())
                            + [self.config_interval]
                            + [os.path.join(f"{self.checkpoint_dir}", f"InitConfig{i}_Stage{self.n_distills}.pt")]
                        )
                        job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names, values)
                        )
                        init_overrides.append(job_overrides)
                for i in range(self.population_size):
                    self.history[i]["config_source"].append("random")
            elif self.init_policy == "ucb":
                init_configs, init_overrides = self._generate_initializing_points_ucb(
                    init_size=max(self.init_size, self.pop_size)
                )
                for i in range(self.population_size):
                    self.history[i]["config_source"].append("ucb")
            else:
                init_configs, init_overrides, architectures = self.get_bo_init_points()
                for i in range(self.population_size):
                    self.history[i]["config_source"].append("bo")

            # Launch initial points
            log.info("Launching initial evaluation.")
            init_results, _ = self.run_configs(init_overrides)
            self.best_init = min(init_results)
            log.info("Initial evaluation finished!")
            self.init_data[self.n_distills] = {}
            self.init_data[self.n_distills]["configs"] = init_configs
            self.init_data[self.n_distills]["performances"] = init_results
            self.checkpoint_pbt()

        # Distillation
        if self.distill_arg_name is not None and self.n_distills > 0:
            log.info("Launching distillation step")
            init_configs, init_archs = self.distillation_step(init_configs, architectures, best_agents)

        for i in range(self.population_size):
            if self.n_distills == 0:
                self.history[i]["nas_hps"].append(self.nas_defaults)
            elif len(self.history[i]["nas_hps"]) == 0:
                self.history[i]["nas_hps"].append(self.archive[max(self.archive.keys())][i]["nas_hps"][-1])
            elif self.init_policy == "ucb" or self.init_policy == "random":
                self.history[i]["nas_hps"].append(self.history[i]["nas_hps"][-self.population_size])
            else:
                self.history[i]["nas_hps"].append(init_archs[i])

        # Reduce to population_size many configs
        if self.init_size <= self.population_size:
            top_config_ids = [
                np.arange(len(init_configs)).tolist() for _ in range(self.population_size // len(init_configs))
            ]
            top_config_ids = [i for li in top_config_ids for i in li]
            top_config_ids = top_config_ids[: self.population_size]
        else:
            # using the ``pop_size'' best as the initialising population
            top_config_ids = np.argsort(init_results).tolist()[: self.population_size]

        log.info("Selected population members.")
        # Get overrides for initial population
        configs = []
        overrides = []
        for i, config_id in enumerate(top_config_ids):
            names = list(init_configs[config_id].keys()) + self.nas_hps + [self.budget_arg_name] + [self.save_arg_name]
            configs.append(init_configs[config_id])
            if self.seeds:
                for s in self.seeds:
                    save_path = os.path.join(self.checkpoint_dir, f"model_iteration_{self.iteration}_id_{i}_s{s}.pt")
                    values = (
                        list(init_configs[config_id].values())
                        + self.history[i]["nas_hps"][-1]
                        + [self.config_interval]
                        + [save_path]
                    )
                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                    )
                    overrides.append(job_overrides)
            else:
                save_path = os.path.join(self.checkpoint_dir, f"model_iteration_{self.iteration}_id_{i}.pt")
                values = (
                    list(init_configs[config_id].values())
                    + self.history[i]["nas_hps"][-1]
                    + [self.config_interval]
                    + [save_path]
                )
                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}" for name, val in zip(names, values)
                )
                overrides.append(job_overrides)
        log.info("Got population configs.")
        return overrides, configs

    def fit_model(self, performances, old_configs):
        """
        Decide whether to restart and get data for model.

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        old_configs: List[Configuration]
            A list of the recent configs
        """
        self.adjust_tr_length(restart=True)

        if self.backtrack:
            best_cost = min(performances)
            if best_cost < self.best_cost:
                self.best_cost = best_cost
                overall_best_agent = np.argmin(performances)
                best_path = os.path.join(
                    self.checkpoint_dir,
                    f"model_iteration_{self.iteration-1}_id_{overall_best_agent}.pt",
                )
                shutil.copy(best_path, self.best_checkpoint_dir)

        latest = [self.history[i]["performances"][-1] for i in range(self.population_size)]
        best_so_far = min([min(self.history[i]["performances"]) for i in range(self.population_size)])
        if any([ll >= best_so_far for ll in latest]):
            self.n_fail = 0
        else:
            self.n_fail += 1
            log.info(f"Failed, now at {self.n_fail} failures.")
            # restart when the casmo trust region is below threshold
        if (
            self.n_fail >= self.patience
            or self.iteration - self.last_restart > self.restart_every
            or self.iteration - self.last_distill_iter > self.distill_every
        ):
            self.n_fail = 0
            if self.n_fail >= self.patience:
                log.info("n_fail reached patience. Restarting GP.")
            elif self.iteration - self.last_distill_iter > self.distill_every:
                log.info(f"Routine distillation after {self.iteration} iterations.")
            else:
                log.info(f"Routine restart after {self.iteration} iterations.")
            self.restart(
                restart_kernel=self.n_fail >= self.patience or self.iteration - self.last_restart > self.restart_every
            )
            log.info(f"At {self.n_fail} failures after restart.")
            performance_quantiles = np.quantile(performances, [self.quantiles])[0]
            best_agents = [i for i in range(len(performances)) if performances[i] < performance_quantiles[0]]
            self.get_initial_configs(best_agents)
            self.last_restart = self.iteration
            self.last_distill_iter = self.iteration
        else:
            for i in range(self.population_size):
                self.history[i]["config_source"].append("bo")
        self.get_model_data(performances, old_configs)
        self.current = []

    def restart(self, restart_kernel=True):
        """Restart kernel."""
        log.info("Restarting!")
        self.archive[self.n_distills] = deepcopy(self.history)
        self.n_distills += 1
        self.history = {}
        for i in range(self.population_size):
            self.history[i] = {"configs": [], "performances": [], "overwritten": [], "config_source": [], "nas_hps": []}
        if restart_kernel:
            self.n_restarts += 1
            self.casmo.length = self.casmo.length_init
            self.casmo.length_cat = self.casmo.length_init_cat
            self.casmo.failcount = self.casmo.succcount = 0

    def perturb_hps(self, config, performance, best_agent, is_good):
        """
        Get next configuration

        Parameters
        ----------
        config: Configuration
            The current config
        performance: float
            The latest performance
        best_agent: int
            Current best agent id
        is_good: bool
            Whether this config was in the upper quantile

        Returns
        -------
        Configuration
            The updated configuration.
        """
        if is_good:
            # As config isn't changed, we keep old config source
            if len(self.history[len(self.current)]["config_source"]) == 1:
                pass
            else:
                self.history[len(self.current)]["config_source"][-1] = self.history[len(self.current)]["config_source"][
                    -2
                ]
            return config

        ts = [t for t in self.ts if t > 0]
        X = np.concatenate((self.all_hps, self.fixed), axis=1)
        # contextual dimension
        # config array of the running parameters
        current = self.current  # [c for cur in self.current for c in cur]
        # it is important that the contextual information is appended to the end of the vector.
        if len(self.history[best_agent]["performances"]) == 0:
            curr_rew_diff = (
                max(self.init_data[max(self.init_data.keys())]["performances"])
                - self.archive[max(self.archive.keys())][best_agent]["performances"][-1]
            )
            best = np.argmax(self.init_data[max(self.init_data.keys())]["performances"])
            X_best = [
                list(self.init_data[max(self.init_data.keys())]["configs"][best].values())
                + [self.iteration]
                + [self.init_data[max(self.init_data.keys())]["performances"][best]]
            ]
        else:
            if len(self.history[best_agent]["performances"]) == 1:
                curr_rew_diff = self.history[best_agent]["performances"][-1] - self.best_init
            else:
                curr_rew_diff = (
                    self.history[best_agent]["performances"][-1] - self.history[best_agent]["performances"][-2]
                )
            X_best = [
                list(self.history[best_agent]["configs"][-1].values())
                + [self.iteration]
                + [self.history[best_agent]["performances"][-1]]
            ]
        if len(current) > 0:
            t_current = np.tile(self.iteration + 1, len(current))
            current = np.array([c + [self.iteration + 1] + [curr_rew_diff] for c in current]).astype(float)
            # get the hp of the best agent selected from -- this will be trust region centre
            new_config_array = self.get_config(X, self.y, ts, current, t_current, x_center=X_best)
            new_config = CS.Configuration(self.configspace, vector=new_config_array)
        else:
            try:
                new_config_array = self.get_config(X, self.y, ts, None, None, x_center=X_best)
                new_config = CS.Configuration(self.configspace, vector=new_config_array)
            except:
                new_config = super().perturb_hps(config, None, None, None)
        self.current.append(list(new_config.values()))
        return new_config

    def select_configs(self, performances, old_configs):
        """
        Given a set of configs and their performances, this method:
        - overwrites the lowest performing ones with the highest performing ones
        - iterates on the configurations
        - creates overrides for the next iteration

        Parameters
        ----------
        performances: List[float]
            A list of the latest agent performances
        configs: List[Configuration]
            A list of the recent configs

        Returns
        -------
        List[Tuple]
            Overrides for the hydra launcher with the config values as well as saving and loading
        List[Configuration]
            The initial configurations.
        """
        overrides, configs = super().select_configs(performances, old_configs)
        noverrides = []
        for i in range(len(overrides)):
            base_tuple = overrides[i]
            if len(overrides) > self.population_size:
                noverrides = overrides
                break
            nas_overrides = tuple(f"{name}={val}" for name, val in zip(self.nas_hps, self.history[i]["nas_hps"][-1]))
            noverrides.append(base_tuple + nas_overrides)
        return noverrides, configs

    def get_config(self, X, y, t, X_current=None, t_current=None, x_center=None, frozen_dims=None, frozen_vals=None):
        """
        Main BO Loop (corresponding to the self.suggest function in Casmo.py.

        Parameters
        ----------
        X: List
            Historical data for model
        y: List
            Historical performances
        X_current:
            Selected but not finished configurations
        t_current:
            Current optimization step
        x_center:
            Search space center (= best config)
        frozen_dims:
            frozen dimension indices
        forzen_vals:
            frozen values

        Returns
        -------
        Configuration
            The updated configuration.
        """
        # 1. normalize the fixed dimensions (note that the variable dimensions are already scaled to [0,1]^d using
        # config_space sometimes we get object array and cast them to float
        X = np.array(X).astype(float)
        y = np.array(y).astype(float)
        if t is not None:
            t = np.array(t).astype(float)
        if X_current is not None:
            X_current = np.array(X_current).astype(float)
        if t_current is not None:
            t_current = np.array(t_current).astype(float)
        # the dimensions attached to the end of the vector are fixed dims
        hypers = {}
        use_time_varying_gp = np.unique(t).shape[0] > 1
        num_fixed = X.shape[1] - len(self.configspace)
        if X_current is not None:
            if num_fixed > 0:
                oldpoints = X[:, -num_fixed:]  # fixed dimensions
                # appropriate rounding
                newpoint = X_current[:, -num_fixed:]
                fixed_points = np.concatenate((oldpoints, newpoint), axis=0)
                lims = np.concatenate((np.max(fixed_points, axis=0), np.min(fixed_points, axis=0))).reshape(
                    2, oldpoints.shape[1]
                )

                lims[0] -= 1e-8
                lims[1] += 1e-8

                X[:, -num_fixed:] = normalize(X[:, -num_fixed:], lims)

            # 2. Train a GP conditioned on the *real* data which would give us the
            # fantasised y output for the pending fixed_points
            if num_fixed > 0:
                X_current[:, -num_fixed:] = normalize(X_current[:, -num_fixed:], lims)
            y = copula_standardize(deepcopy(y).ravel())
            if len(X) < MIN_CUDA:
                device, dtype = torch.device("cpu"), torch.float32
            else:
                device, dtype = self.casmo.device, self.casmo.dtype

            with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                X_torch = torch.tensor(X).to(device=device, dtype=dtype)
                # here we replace the nan values with zero, but record the nan locations via the X_torch_nan_mask
                y_torch = torch.tensor(y).to(device=device, dtype=dtype)
                # add some noise to improve numerical stability
                y_torch += torch.randn(y_torch.size()) * 1e-5
                t_torch = torch.tensor(t).to(device=device, dtype=dtype)

                gp = train_gp(
                    configspace=self.casmo.cs,
                    train_x=X_torch,
                    train_y=y_torch,
                    use_ard=False,
                    num_steps=200,
                    time_varying=True if use_time_varying_gp else False,
                    train_t=t_torch,
                    # verbose=self.verbose,
                )
                hypers = gp.state_dict()
            # 3. Get the posterior prediction at the fantasised points
            gp.eval()
            if use_time_varying_gp:
                t_x_current = torch.hstack(
                    (
                        torch.tensor(t_current, dtype=dtype).reshape(-1, 1),
                        torch.tensor(X_current, dtype=dtype),
                    )
                )
            else:
                t_x_current = torch.tensor(X_current, dtype=dtype)
            pred_ = gp(t_x_current).mean
            y_fantasised = pred_.detach().numpy()
            y = np.concatenate((y, y_fantasised))
            X = np.concatenate((X, X_current), axis=0)
            t = np.concatenate((t, t_current))
            del X_torch, y_torch, t_torch, gp
        # scale the fixed dimensions to [0, 1]^d
        y = copula_standardize(deepcopy(y).ravel())
        # simply call the _create_and_select_candidates subroutine to return
        # But first: make sure normalization produced no value errors
        X[X <= 0.01] = 0.01
        X[X >= 0.99] = 0.99
        y[y <= 0.01] = 0.001
        y[y >= 0.99] = 0.99
        next_config = self.casmo._create_and_select_candidates(
            X,
            y,
            length_cat=self.casmo.length_cat,
            length_cont=self.casmo.length,
            hypers=hypers,
            batch_size=1,
            t=t if use_time_varying_gp else None,
            time_varying=use_time_varying_gp,
            x_center=np.array(x_center),
            frozen_dims=frozen_dims,
            frozen_vals=frozen_vals,
            n_training_steps=1,
        ).flatten()
        # truncate the array to only keep the hyperparameter dimenionss
        if num_fixed > 0:
            next_config = next_config[:-num_fixed]
        return next_config

    def adjust_tr_length(self, restart=False):
        """
        Adjust trust region size -- the criterion is that whether any config sampled by BO outperforms the other config
        sampled otherwise (e.g. randomly, or carried from previous timesteps). If true, then it will be a success or
        failure otherwise.

        Parameters
        ----------
        restart: bool
            Whether kernel can be restarted
        """
        # get the negative reward
        _, best_reward = self.get_incumbent()
        # get the agents selected by Bayesian optimization
        bo_agents = [i for i in range(self.population_size) if self.history[i]["config_source"][-1] == "bo"]
        bo_rewards = [
            self.history[i]["performances"][-1]
            for i in range(self.population_size)
            if self.history[i]["config_source"][-1] == "bo"
        ]
        if len(bo_agents) == 0:
            return
        # if the best reward is caused by a config suggested by BayesOpt
        if min(bo_rewards) == best_reward:
            self.casmo.succcount += 1
            self.casmo.failcount = 0
        else:
            self.casmo.failcount += 1
            self.casmo.succcount = 0
        if self.casmo.succcount == self.casmo.succtol:  # Expand trust region
            self.casmo.length = min([self.casmo.tr_multiplier * self.casmo.length, self.casmo.length_max])
            self.casmo.length_cat = min(self.casmo.length_cat * self.casmo.tr_multiplier, self.casmo.length_max_cat)
            self.casmo.succcount = 0
            log.info(f"Expanding TR length to {self.casmo.length}")
        elif self.casmo.failcount == self.casmo.failtol:  # Shrink trust region
            self.casmo.failcount = 0
            self.casmo.length_cat = max(self.casmo.length_cat / self.casmo.tr_multiplier, self.casmo.length_min_cat)
            self.casmo.length = max(self.casmo.length / self.casmo.tr_multiplier, self.casmo.length_min)
            log.info(f"Shrinking TR length to {self.casmo.length}")
        if restart and (
            self.casmo.length <= self.casmo.length_min or self.casmo.length_max_cat <= self.casmo.length_min_cat
        ):
            self._restart()

    def _generate_initializing_points_ucb(self, n_init):
        """
        Generate inital points after restart using ucb.

        Parameters
        -------
        n_init: int
            The number of configurations to return.

        Returns
        -------
        List[Configuration]
            A list of new configurations.
        """
        # for each of the previous restart (for restart > 0), based on the current GP, find the best points based on
        #    the auxiliary GP, ranked by their UCB score -- this is required for the Theoretical guarantee but is
        #    only applicable in the case without distillation and etc.
        if n_init is None:
            n_init = self.init_size
        # fit a GP based on the results from the previous restart
        # this will return the data in the latest n_distills
        self.get_model_data()
        y = np.array(self.ys).astype(float)
        t = np.array(self.ts).astype(float)
        X = self.hp_values
        y = copula_standardize(deepcopy(y).ravel())
        if len(X) < MIN_CUDA:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.casmo.device, self.casmo.dtype

        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(y).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5
            t_torch = torch.tensor(t).to(device=device, dtype=dtype)

            gp = train_gp(
                configspace=self.casmo.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=False,
                num_steps=200,
                time_varying=True,
                train_t=t_torch,
                # verbose=self.verbose,
            )
        gp.eval()
        # the training points to add for the auxiliary GP
        aux_train_input, aux_train_target = [], []
        for restart in range(self.n_distills):
            data = self.archive[restart]
            # dfnewpoint, data, _ = self.format_df(0, n_distills=restart)
            # X = data[["x{}".format(i) for i in range(len(self.df.conf[0]))]]
            X = []
            for i in range(len(data["configs"])):
                for j in self.population_size:
                    config = data[j]["configs"][i]
                    X.append([v for v in list(config.values())])
            X = np.array(X)
            t_current = (self.iteration * self.config_interval) * np.ones(X.shape[0])
            t_x_current = torch.hstack(
                (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(X, dtype=dtype))
            )
            pred_ = gp(t_x_current).mean
            # select the x with the best
            best_idx = np.argmin(pred_.detach().numpy())
            aux_train_input.append(X[best_idx, :])
            aux_train_target.append(pred_.detach().numpy()[best_idx, :])
        # now fit the auxiliary GP
        aux_gp = train_gp(
            configspace=self.casmo.cs,
            train_x=torch.tensor(aux_train_input).to(device=device, dtype=dtype),
            train_y=torch.tensor(aux_train_target).to(device=device, dtype=dtype),
            use_ard=False,
            num_steps=200,
            time_varying=True,
            train_t=t_torch,
            verbose=self.verbose,
        )
        aux_gp.eval()

        # now generate a bunch of random configs
        random_configs = [self.configspace.sample_configuration().get_array() for _ in range(10 * n_init)]
        random_config_arrays = [c.get_array() for c in random_configs]
        t_current = (self.iteration * self.config_interval) * np.ones(len(random_config_arrays))

        # selection by the UCB score using the predicted mean + var of the auxiliary GP.
        random_config_array_t = torch.hstack(
            (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(random_config_arrays, dtype=dtype))
        )
        pred = aux_gp(random_config_array_t)
        pred_mean, pred_std = pred.mean.detach().numpy(), pred.stddev.detach().numpy()
        ucb = pred_mean - 1.96 * pred_std
        top_config_ids = np.argpartition(np.array(ucb), n_init)[:n_init].tolist()
        return [random_configs[i] for i in top_config_ids]

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

        data = self.history
        init_data = False
        data_level = self.n_distills
        offset = 0
        all_hps = []
        hp_values = []
        self.cat_values = []
        ts = []
        tps = []
        ys = []
        for i in reversed(range(1000 // self.population_size)):
            t = len(data[0]["performances"]) - i
            if t <= 0:
                if not init_data and data_level > 0:
                    data = [self.init_data[data_level]]
                    init_data = True
                    data_level -= 1
                    offset += i
                elif data_level > 0:
                    data = self.archive[data_level]
                    init_data = False
                    data_level -= 1
                    offset += i
                    if len(data[0]["performances"]) == 0:
                        break
            for j in range(self.population_size):
                if j >= len(data):
                    continue
                t = len(data[j]["performances"]) - i
                if t + offset <= 0 or -i + offset >= len(data[j]["performances"]):
                    continue
                p = data[j]["performances"][-i + offset]
                ys.append(data[j]["performances"][-i - 1 + offset] - p)
                config = data[j]["configs"][-i + offset]
                hps = [v for v, n in zip(list(config.values()), list(config.keys())) if n in self.continuous_hps]
                cat = [v for v, n in zip(list(config.values()), list(config.keys())) if n in self.categorical_hps]
                all_hp = [v for v in list(config.values())]
                all_hps.append(all_hp)
                self.cat_values.append(cat)
                hp_values.append(hps)
                tps.append([t, p])

        # current_best_values = list(current_best[-1].values())
        self.ts = np.array(ts)
        self.hp_values = np.array(hp_values)
        self.all_hps = [np.array(a) for a in all_hps]
        self.all_hps = np.array(all_hps)
        self.ys = np.array(ys)

        self.X = normalize(self.hp_values, self.hp_bounds.T)
        self.y = standardize(self.ys).reshape(self.ys.size, 1)

        self.fixed = normalize(tps, [1000 // self.population_size, max(performances)])

        self.X = np.concatenate((self.fixed, self.X), axis=1)
