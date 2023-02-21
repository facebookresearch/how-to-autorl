# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import time
import json
import pickle
import wandb
import numpy as np
from ConfigSpace.hyperparameters import (
    BetaIntegerHyperparameter,
    CategoricalHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformIntegerHyperparameter,
)
from hydra.utils import to_absolute_path
from deepcave import Recorder, Objective
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


class HydraPBT:
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
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        init_size=8,
        model_based=False,
        base_dir=False,
        population_size=64,
        config_interval=None,
        num_config_changes=None,
        quantiles=0.25,
        resample_probability=0.25,
        perturbation_factors=[1.2, 0.8],
        categorical_fixed=True,
        categorical_prob=0.25,
        warmstart=False,
        wandb_project=False,
        wandb_entity=False,
        wandb_tags=["pbt"],
        deepcave=False,
        maximize=False,
    ):
        """
        Classic PBT Implementation with random search as backend.

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
            Hyperparamters are multiplied with the first factor when their value is increased
            and with the second if their value is decreased.
        categorical_fixed: bool
            Whether categorical hyperparameters are ignored or optimized jointly.
        categorical_prob: float
            Probability of categorical values being resampled.
        Returns
        -------
        None
        """
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_arg_name = budget_arg_name
        self.load_arg_name = load_arg_name
        self.save_arg_name = save_arg_name
        self.configspace = cs
        self.output_dir = to_absolute_path(base_dir) if base_dir else to_absolute_path("./")
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.job_idx = 0
        self.model_based = model_based
        self.seeds = seeds
        if seeds and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == "seed":
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    break
        self.warmstart = warmstart
        self.init_size = init_size
        self.maximize = maximize
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout

        self.population_size = population_size
        self.quantiles = [quantiles, 1 - quantiles]
        self.perturbation_factors = perturbation_factors
        self.resample_probability = resample_probability
        self.categorical_fixed = categorical_fixed
        self.categorical_prob = categorical_prob

        assert not (config_interval is None and num_config_changes is None)
        self.config_interval = config_interval
        self.num_config_changes = num_config_changes
        self.total_budget = total_budget
        if self.config_interval is None:
            self.config_interval = int(total_budget // self.num_config_changes)
        if self.num_config_changes is None:
            self.num_config_changes = int(total_budget // self.config_interval) - 1

        self.current_steps = 0
        self.iteration = 0
        self.opt_time = 0
        self.incumbent = []
        self.resume = False
        self.history = {}
        for i in range(self.population_size):
            self.history[i] = {"configs": [], "performances": [], "overwritten": []}

        self.categorical_hps = [
            n
            for n in list(self.configspace.keys())
            if isinstance(self.configspace.get_hyperparameter(n), CategoricalHyperparameter)
        ]
        self.categorical_hps += [
            n
            for n in list(self.configspace.keys())
            if isinstance(self.configspace.get_hyperparameter(n), OrdinalHyperparameter)
        ]
        self.continuous_hps = [n for n in list(self.configspace.keys()) if n not in self.categorical_hps]
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

        self.deepcave = deepcave
        if self.deepcave:
            reward_objective = Objective("reward", optimize="lower")
            deepcave_path = os.path.join(self.output_dir, "deepcave_logs")
            self.deepcave_recorder = Recorder(self.configspace, objectives=[reward_objective], save_path=deepcave_path)
        self.wandb_project = wandb_project
        if self.wandb_project:
            wandb_config = OmegaConf.to_container(global_config, resolve=False, throw_on_missing=False)
            assert wandb_entity, "Please provide an entity to log to W&B."
            wandb.init(
                project=self.wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                config=wandb_config,
            )

    def perturb_hps(self, config, _, __, ___):
        """
        Given a config, perturb the hyperparameters to find a better configuration.
        With a given probability, a hyperparameter is resampled.
        Else, if the hp is not categorical, it's randomly either increased or decreased.
        If it's a categorical hyperparameter and categorical hps are optimized, it's resampled with a given probability.

        Parameters
        ----------
        config: Configuration
            A config of an agent
        _: not relevant here
        __: not relevant here
        ___: not relevant here

        Returns
        -------
        Configuration
            The perturbed configuration.
        """
        for name in self.continuous_hps:
            hp = self.configspace.get_hyperparameter(name)
            if np.random.random() < self.resample_probability:
                # Resample hyperparamter
                config[name] = hp.rvs()
            else:
                # Perturb
                perturbation_factor = np.random.choice(self.perturbation_factors)
                perturbed_value = config[name] * perturbation_factor
                if (
                    isinstance(hp, UniformIntegerHyperparameter)
                    or isinstance(hp, NormalIntegerHyperparameter)
                    or isinstance(hp, BetaIntegerHyperparameter)
                ):
                    perturbed_value = int(perturbed_value)
                if (perturbation_factor > 1 and perturbed_value > 0) or (
                    perturbation_factor <= 1 and perturbed_value <= 0
                ):
                    config[name] = min(perturbed_value, hp.upper)
                else:
                    config[name] = max(perturbed_value, hp.lower)

        if not self.categorical_fixed:
            for n in self.categorical_hps:
                if np.random.random() < self.categorical_prob:
                    hp = self.configspace.get_hyperparameter(name)
                    config[name] = hp.rvs()

        return config

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
        log.info("Sampling intial points.")
        self.init_idx = 0
        init_overrides = []
        init_configs = []
        if self.warmstart:
            num_inits = self.init_size
            for i in range(num_inits):
                config = self.configspace.sample_configuration()
                names = list(config.keys()) + [self.budget_arg_name] + [self.save_arg_name]
                if self.slurm:
                    names += ["hydra.launcher.timeout_min"]
                    optimized_timeout = (
                        self.slurm_timeout * 1 / (self.total_budget // self.config_interval) + 0.1 * self.slurm_timeout
                    )
                init_configs.append(config)
                if self.seeds:
                    for s in self.seeds:
                        values = (
                            list(config.values())
                            + [self.total_budget]
                            + [os.path.join(f"{self.checkpoint_dir}", f"InitConfig{i}_s{s}.pt")]
                        )
                        if self.slurm:
                            values += [int(optimized_timeout)]
                        job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                        )
                        init_overrides.append(job_overrides)

                else:
                    values = (
                        list(config.values())
                        + [self.config_interval]
                        + [os.path.join(f"{self.checkpoint_dir}", f"InitConfig{i}.pt")]
                    )
                    if self.slurm:
                        values += [int(optimized_timeout)]
                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}" for name, val in zip(names, values)
                    )
                    init_overrides.append(job_overrides)

            if not self.warmstart:
                return init_overrides, init_configs

            # Launch initial points
            log.info("Launching initial evaluation.")
            init_results, _ = self.run_configs(init_overrides)
            self.best_init = min(init_results)
            log.info("Initial evaluation finished!")

            # Reduce to population_size many configs
            if self.init_size == self.population_size:
                top_config_ids = np.arange(self.population_size)
            elif self.init_size < self.population_size:
                top_config_ids = [
                    np.arange(len(init_configs)).tolist() for _ in range(self.population_size // len(init_configs))
                ]
                top_config_ids = [i for li in top_config_ids for i in li]
                top_config_ids = top_config_ids[: self.population_size]
            else:
                # using the `pop_size' best as the initialising population
                top_config_ids = np.argpartition(np.array(init_results), self.population_size)[
                    : self.population_size
                ].tolist()

            log.info("Selected population members.")
        else:
            init_configs = self.configspace.sample_configuration(size=self.population_size)
            top_config_ids = np.arange(self.population_size)

        # Get overrides for initial population
        configs = []
        overrides = []
        for i, config_id in enumerate(top_config_ids):
            names = list(init_configs[config_id].keys()) + [self.budget_arg_name] + [self.save_arg_name]
            if self.slurm:
                names += ["hydra.launcher.timeout_min"]
                optimized_timeout = (
                    self.slurm_timeout * 1 / (self.total_budget // self.config_interval) + 0.1 * self.slurm_timeout
                )
            configs.append(init_configs[config_id])
            if self.seeds:
                for s in self.seeds:
                    save_path = os.path.join(self.checkpoint_dir, f"model_iteration_{self.iteration}_id_{i}_s{s}.pt")
                    values = list(init_configs[config_id].values()) + [self.config_interval] + [save_path]
                    if self.slurm:
                        values += [int(optimized_timeout)]
                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                    )
                    overrides.append(job_overrides)
            else:
                save_path = os.path.join(self.checkpoint_dir, f"model_iteration_{self.iteration}_id_{i}.pt")
                values = list(init_configs[config_id].values()) + [self.config_interval] + [save_path]
                if self.slurm:
                    values += [int(optimized_timeout)]
                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{name}={val}" for name, val in zip(names, values)
                )
                overrides.append(job_overrides)
        log.info("Got population configs.")
        return overrides, configs

    def run_configs(self, overrides):
        """
        Run a set of overrides

        Parameters
        ----------
        overrides: List[Tuple]
            A list of overrides to launch

        Returns
        -------
        List[float]
            The resulting performances.
        List[float]
            The incurred costs.
        """
        res = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(overrides)
        costs = [self.config_interval for i in range(len(res))]
        done = False
        while not done:
            for j in range(len(overrides)):
                try:
                    res[j].return_value
                    done = True
                except:
                    done = False

        performances = []
        if self.seeds:
            for j in range(0, self.population_size):
                performances.append(np.mean([res[j * k + k].return_value for k in range(len(self.seeds))]))
        else:
            for j in range(len(overrides)):
                performances.append(res[j].return_value)
        if self.maximize:
            performances = [-p for p in performances]
        return performances, costs

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
        overrides = []
        configs = []
        if self.current_steps == 0:
            return self.get_initial_configs()
        else:
            if self.resume:
                performances = [self.history[i]["performances"][-1] for i in range(self.population_size)]
                old_configs = [self.history[i]["configs"][-1] for i in range(self.population_size)]
            # Check where to copy weights and where to discard
            performance_quantiles = np.quantile(performances, [self.quantiles])[0]
            worst_config_ids = [i for i in range(len(performances)) if performances[i] > performance_quantiles[1]]
            best_config_ids = [i for i in range(len(performances)) if performances[i] < performance_quantiles[0]]
            if len(best_config_ids) == 0:
                best_config_ids = [np.argmax(performances)]
            loading_values = np.arange(len(performances))
            for i in worst_config_ids:
                loading_values[i] = np.random.choice(best_config_ids)

            if self.model_based:
                self.fit_model(performances, old_configs)

            # Perturb hp values & generate overrides
            for i in range(self.population_size):
                new_config = self.perturb_hps(
                    old_configs[i], performances[i], np.random.choice(best_config_ids), i in best_config_ids
                )
                names = (
                    list(old_configs[i].keys()) + [self.budget_arg_name] + [self.save_arg_name] + [self.load_arg_name]
                )
                if self.slurm:
                    names += ["hydra.launcher.timeout_min"]
                    optimized_timeout = (
                        self.slurm_timeout * 1 / (self.total_budget // self.config_interval) + 0.1 * self.slurm_timeout
                    )
                configs.append(new_config)
                if self.seeds:
                    for s in self.seeds:
                        save_path = os.path.join(
                            self.checkpoint_dir, f"model_iteration_{self.iteration}_id_{i}_s{s}.pt"
                        )
                        load_path = os.path.join(
                            self.checkpoint_dir,
                            f"model_iteration_{self.iteration-1}_id_{loading_values[i]}_s{s}.pt",
                        )
                        values = list(new_config.values()) + [self.config_interval] + [save_path] + [load_path]
                        if self.slurm:
                            values += [int(optimized_timeout)]
                        job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                        )
                        overrides.append(job_overrides)
                else:
                    save_path = os.path.join(self.checkpoint_dir, f"model_iteration_{self.iteration}_id_{i}.pt")
                    load_path = os.path.join(
                        self.checkpoint_dir,
                        f"model_iteration_{self.iteration-1}_id_{loading_values[i]}.pt",
                    )
                    values = list(new_config.values()) + [self.config_interval] + [save_path] + [load_path]
                    if self.slurm:
                        values += [int(optimized_timeout)]
                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{name}={val}" for name, val in zip(names, values)
                    )
                    overrides.append(job_overrides)
                overwritten = False if loading_values[i] == i else loading_values[i]
                self.history[i]["overwritten"].append(overwritten)

        return overrides, configs

    def get_incumbent(self):
        """
        Get the best sequence of configurations so far.

        Returns
        -------
        List[Configuration]
            Sequence of best hyperparameter configs
        Float
            Best performance value
        """
        best_current_id = np.argmin([self.history[i]["performances"][-1] for i in range(self.population_size)])
        inc_performance = self.history[best_current_id]["performances"][-1]
        if not any(
            [
                isinstance(self.history[best_current_id]["overwritten"][i], int)
                for i in range(len(self.history[best_current_id]["overwritten"]))
            ]
        ):
            inc_config = self.history[best_current_id]["configs"]
        else:
            inc_config = []
            for i in range(len(self.history[best_current_id]["configs"])):
                index = len(self.history[best_current_id]["overwritten"]) - i - 1
                if isinstance(self.history[best_current_id]["overwritten"][index], int):
                    overwritten_by = self.history[best_current_id]["overwritten"][index]
                    inc_config += self.history[overwritten_by]["configs"][:index]
                    inc_config += self.history[best_current_id]["configs"][index:]
                    break
        return inc_config, inc_performance

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
        log.info(self.population_size)
        for i in range(self.population_size):
            self.history[i]["configs"].append(configs[i])
            self.history[i]["performances"].append(performances[i])
        log.info(self.history)
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
            stats["num_steps"] = self.iteration * self.config_interval
            for i in range(self.population_size):
                stats[f"performance_{i}"] = -performances[i]
                for n in configs[0].keys():
                    stats[f"config_{i}_{n}"] = configs[i].get(n)
            best_config = configs[np.argmin(performances)]
            for n in best_config.keys():
                stats[f"incumbent_{n}"] = best_config.get(n)
            wandb.log(stats)

    def _save_incumbent(self, name=None):
        """
        Log current incumbent to file (as well as some additional info).

        Parameters
        ----------
        name: str | None
            Optional filename
        """
        if name is None:
            name = "incumbent.json"
        res = dict()
        incumbent, inc_performance = self.get_incumbent()
        res["config"] = [config.get_dictionary() for config in incumbent]
        res["score"] = float(inc_performance)
        res["total_training_steps"] = self.iteration * self.config_interval * self.population_size
        res["total_wallclock_time"] = self.start - time.time()
        res["total_optimization_time"] = self.opt_time
        with open(os.path.join(self.output_dir, name), "a+") as f:
            json.dump(res, f)
            f.write("\n")

    def checkpoint_pbt(self):
        d = self._get_state()
        with open(os.path.join(self.output_dir, "pbt_state.pkl"), "wb") as f:
            pickle.dump(d, f)

    def load_pbt(self, path):
        with open(path, "rb") as f:
            past_state = pickle.load(f)
        self._set_state(past_state)
        self.resume = True

    def _get_state(self):
        state = {}
        state["history"] = self.history
        state["iteration"] = self.iteration
        state["current_steps"] = self.current_steps
        state["output_dir"] = self.output_dir
        return state

    def _set_state(self, state):
        self.history = state["history"]
        self.iteration = state["iteration"]
        self.current_steps = state["current_steps"]
        self.output_dir = state["output_dir"]

    def run(self, verbose=False):
        """
        Actual optimization loop.
        In each iteration:
        - get configs (either randomly upon init or through perturbation)
        - run current configs
        - record performances

        Parameters
        ----------
        verbose: bool
            More logging info

        Returns
        -------
        List[Configuration]
            The incumbent configurations.
        """
        if verbose:
            log.info("Starting PBT Sweep")
        performances = np.zeros(self.population_size)
        configs = None
        self.start = time.time()
        while self.iteration <= self.num_config_changes:
            opt_time_start = time.time()
            overrides, configs = self.select_configs(performances, configs)
            self.opt_time += time.time() - opt_time_start
            performances, _ = self.run_configs(overrides)
            opt_time_start = time.time()
            self.record_iteration(performances, configs)
            if self.deepcave:
                for c, p in zip(configs, performances):
                    self.deepcave_recorder.start(config=c, budget=self.config_interval)
                    self.deepcave_recorder.end(costs=p, config=c, budget=self.config_interval)
            if verbose:
                log.info(f"Finished Generation {self.iteration}!")
                log.info(f"Current best agent has reward of {-np.round(min(performances), decimals=2)}.")
            self._save_incumbent()
            self.checkpoint_pbt()
            self.opt_time += time.time() - opt_time_start
        total_time = time.time() - self.start
        inc_config, _ = self.get_incumbent()
        if verbose:
            log.info(
                f"Finished PBT Sweep! Total duration was {np.round(total_time, decimals=2)}s, \
                    best agent had a performance of {np.round(min(performances), decimals=2)}"
            )
            log.info(f"The incumbent schedule (changes every {self.config_interval} steps) was: {inc_config}")
        return self.incumbent
