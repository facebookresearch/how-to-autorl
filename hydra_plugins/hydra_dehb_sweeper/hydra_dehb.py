import logging
import os
import time
import json
import wandb
import numpy as np
import pickle
from copy import deepcopy
from hydra.utils import to_absolute_path
from deepcave import Recorder, Objective
from omegaconf import OmegaConf

from hydra_plugins.utils.lazy_imports import lazy_import

dehb = lazy_import("dehb.optimizers.dehb")

log = logging.getLogger(__name__)


# Most of this is very similar to the original DEHB
class HydraDEHB(dehb.DEHB):
    def __init__(
        self,
        global_config,
        global_overrides,
        launcher,
        budget_variable,
        n_jobs,
        base_dir,
        cs,
        f,
        seeds=False,
        slurm=False,
        slurm_timeout=10,
        dimensions=None,
        mutation_factor=0.5,
        crossover_prob=0.5,
        strategy="rand1_bin",
        min_budget=None,
        max_budget=None,
        eta=3,
        min_clip=None,
        max_clip=None,
        configspace=True,
        boundary_fix_type="random",
        max_age=np.inf,
        async_strategy="immediate",
        wandb_project=False,
        wandb_tags=["dehb"],
        deepcave=False,
        maximize=False,
        **kwargs,
    ):
        output_path = to_absolute_path(base_dir)
        kwargs["output_path"] = output_path
        assert min_budget is not None, "Please set a minimum budget per run"
        assert max_budget is not None, "Please set a maximum budget per run"
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            min_budget=min_budget,
            max_budget=max_budget,
            eta=eta,
            min_clip=min_clip,
            max_clip=max_clip,
            configspace=configspace,
            boundary_fix_type=boundary_fix_type,
            max_age=max_age,
            n_workers=1,
            client=None,
            async_strategy=async_strategy,
            **kwargs,
        )

        self.budget_variable = budget_variable
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.trial_id = 0
        self.n_jobs = n_jobs
        self.slurm = slurm
        self.slurm_timeout = slurm_timeout
        self.seeds = seeds
        if seeds and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == "seed":
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    self.global_overrides = (
                        self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    )
        self.current_total_steps = 0
        self.opt_time = 0
        self.maximize = maximize

        self.deepcave = deepcave
        if self.deepcave:
            reward_objective = Objective("reward", optimize="lower")
            deepcave_path = os.path.join(self.output_path, "deepcave_logs")
            self.deepcave_recorder = Recorder(self.cs, objectives=[reward_objective], save_path=deepcave_path)
            self.deepcave_recorder = Recorder(
                self.cs, objectives=[reward_objective], save_path=deepcave_path
            )
        if self.wandb_project:
            wandb_config = OmegaConf.to_container(global_config, resolve=False, throw_on_missing=False)
            wandb_config = OmegaConf.to_container(
                global_config, resolve=False, throw_on_missing=False
            )
                project=self.wandb_project,
                tags=wandb_tags,
                config=wandb_config,
            )

    def _save_incumbent(self, name=None):
        if name is None:
            name = "incumbent.json"

        res = dict()
        if self.configspace:
            config = self.vector_to_configspace(self.inc_config)
            res["config"] = config.get_dictionary()
        else:
            res["config"] = self.inc_config.tolist()
        res["score"] = self.inc_score
        res["info"] = self.inc_info
        res["total_training_steps"] = self.current_total_steps
        res["total_wallclock_time"] = self.start - time.time()
        res["total_optimization_time"] = self.opt_time
        with open(os.path.join(self.output_path, name), "a+") as f:
            json.dump(res, f)
            f.write("\n")

    def checkpoint_dehb(self):
        d = deepcopy(self.__getstate__())
        del d["logger"]
        del d["launcher"]
        del d["f"]
        del d["client"]
        for k in d["de"].keys():
            d["de"][k].f = None
        if "f" in d["de_params"].keys():
            del d["de_params"]["f"]
        try:
            with open(os.path.join(self.output_path, "dehb_state.pkl"), "wb") as f:
                pickle.dump(d, f)
        except Exception as e:
            log.warning("Checkpointing failed: {}".format(repr(e)))

        if self.wandb_project:
            stats = {}
            stats["optimization_time"] = time.time() - self.start
            stats["incumbent_performance"] = self.inc_score
            stats["num_steps"] = self.current_total_steps
            stats["inc_config"] = self.inc_config
            wandb.log(stats)

    def load_dehb(self, path):
        func = list(self.de.values())[0].f
        with open(path, "rb") as f:
            past_state = pickle.load(f)
        self.__dict__.update(**past_state)
        for k in self.de:
            self.de[k].f = func

    def _verbosity_runtime(self, fevals, brackets, total_cost, total_time_cost):
        if fevals is not None:
            remaining = (len(self.traj), fevals, "function evaluation(s) done.")
        elif brackets is not None:
            _suffix = "bracket(s) started; # active brackets: {}.".format(len(self.active_brackets))
            _suffix = "bracket(s) started; # active brackets: {}.".format(
                len(self.active_brackets)
            )
        elif total_time_cost is not None:
            elapsed = np.format_float_positional(time.time() - self.start, precision=2)
            remaining = (elapsed, total_time_cost, "seconds elapsed.")
        else:
            remaining = (int(self.current_total_steps) + 1, total_cost, "training steps run.")
            remaining = (
                int(self.current_total_steps) + 1,
                total_cost,
                "training steps run.",
            )

    def _is_run_budget_exhausted(self, fevals=None, brackets=None, total_cost=None, total_time_cost=None):
    def _is_run_budget_exhausted(
        self, fevals=None, brackets=None, total_cost=None, total_time_cost=None
    ):
        delimiters = [fevals, brackets, total_cost, total_time_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError("Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run.")
            raise ValueError(
                "Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run."
            )
            if len(self.traj) >= fevals:
                return True
        elif brackets is not None:
            if self.iteration_counter >= brackets:
                for bracket in self.active_brackets:
                    # waits for all brackets < iteration_counter to finish by collecting results
                    if bracket.bracket_id < self.iteration_counter and not bracket.is_bracket_done():
                    if (
                        bracket.bracket_id < self.iteration_counter
                        and not bracket.is_bracket_done()
                    ):
                return True
        elif total_time_cost is not None:
            if time.time() - self.start >= total_time_cost:
                return True
            if len(self.runtime) > 0 and self.runtime[-1] - self.start >= total_time_cost:
            if (
                len(self.runtime) > 0
                and self.runtime[-1] - self.start >= total_time_cost
            ):
        else:
            if self.current_total_steps >= total_cost:
                return True
        return False

    # This is very similar to the original DEHB, only that we replace the submit function
    # with the launcher and run one bracket at a time
    def run(
        self,
        fevals=None,
        brackets=None,
        total_cost=None,
        total_time_cost=None,
        single_node_with_gpus=False,
        verbose=True,
        debug=False,
        save_intermediate=True,
        save_history=True,
        name=None,
        **kwargs,
    ):
        """Main interface to run optimization by DEHB
        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.
        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)
        """
        num_brackets = brackets
        self.start = time.time()
        if verbose:
            log.info(
                "\nLogging at {} for optimization starting at {}\n".format(
                    os.path.join(self.output_path, self.log_filename),
                    time.strftime("%x %X %Z", time.localtime(self.start)),
                )
            )
        while True:
            if self._is_run_budget_exhausted(fevals, num_brackets, total_cost, total_time_cost):
            if self._is_run_budget_exhausted(
                fevals, num_brackets, total_cost, total_time_cost
            ):
            bracket = None
            opt_time_start = time.time()
            overrides = []
            bracket_jobs = []
            # Fill up job queue as far as possible
            jobs_left = self.n_jobs
            reset_job_count = False
            if fevals:
                jobs_left = min(jobs_left, fevals - len(self.traj))
            elif total_cost:
                reset_job_count
            while len(overrides) < jobs_left:
                num_queued = len(overrides)
                if len(self.active_brackets) == 0 or np.all(
                    [bracket.is_bracket_done() for bracket in self.active_brackets]
                ):
                    # start new bracket when no pending jobs from existing brackets or empty bracket list
                    bracket = self._start_new_bracket()
                else:
                    for _bracket in self.active_brackets:
                        # check if _bracket is not waiting for previous rung results of same bracket
                        # _bracket is not waiting on the last rung results
                        # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                        if not _bracket.previous_rung_waits() and _bracket.is_pending():
                            # bracket eligible for job scheduling
                            bracket = _bracket
                            break
                    if bracket is None:
                        # start new bracket when existing list has all waiting brackets
                        bracket = self._start_new_bracket()
                # budget that the SH bracket allots
                new_budget = bracket.get_next_job_budget()
                if new_budget is None:
                    break
                budget = new_budget
                if reset_job_count:
                    budget_left = (total_cost - self.current_total_steps) // budget
                    jobs_left = min(jobs_left, budget_left)
                    if jobs_left == 0:
                        break

                # Add more jobs if current bracket isn't waiting on results and also has jobs left
                space_in_bracket = (
                    bracket.sh_bracket[budget] > 0 and not bracket.previous_rung_waits() and bracket.is_pending()
                    bracket.sh_bracket[budget] > 0
                    and not bracket.previous_rung_waits()
                    and bracket.is_pending()
                while space_in_bracket and len(overrides) < jobs_left:
                    vconfig, parent_id = self._acquire_config(bracket, budget)
                    config = self.vector_to_configspace(vconfig)
                    bracket_jobs.append(
                        {
                            "cs_config": config,
                            "config": vconfig,
                            "budget": budget,
                            "parent_id": parent_id,
                            "bracket_id": bracket.bracket_id,
                            "info": {},
                        }
                    )
                    # log.info(f'Got config {config}, parent id {parent_id} and budget {int(budget)+1}')
                    names = list(config.keys()) + [self.budget_variable]
                    values = list(config.values()) + [int(budget) + 1]
                    if self.slurm:
                        names += ["hydra.launcher.timeout_min"]
                        optimized_timeout = (
                            self.slurm_timeout * 1 / (self.max_budget // budget) + 0.1 * self.slurm_timeout
                            self.slurm_timeout * 1 / (self.max_budget // budget)
                            + 0.1 * self.slurm_timeout
                        values += [int(optimized_timeout)]
                    if self.seeds:
                        for s in self.seeds:
                            job_overrides = tuple(self.global_overrides) + tuple(
                                f"{name}={val}" for name, val in zip(names + ["seed"], values + [s])
                                f"{name}={val}"
                                for name, val in zip(names + ["seed"], values + [s])
                            overrides.append(job_overrides)
                    else:
                        job_overrides = tuple(self.global_overrides) + tuple(
                            f"{name}={val}" for name, val in zip(names, values)
                        )
                        overrides.append(job_overrides)
                    self.current_total_steps += budget
                    bracket.register_job(budget)
                    space_in_bracket = bracket.sh_bracket[budget] > 0
                if len(overrides) == num_queued:
                    break

            # Run jobs
            while len(overrides) > 0:
                index = min(self.n_jobs, len(overrides))
                # Make sure that all seeds of a config are launched at the same time so we can aggregate
                if self.seeds:
                    index = (index // len(self.seeds)) * len(self.seeds) // len(self.seeds)
                    index = (
                        (index // len(self.seeds)) * len(self.seeds) // len(self.seeds)
                    )
                    bracket_jobs = bracket_jobs[index:]
                    to_launch = overrides[: index * len(self.seeds)]
                    overrides = overrides[index * len(self.seeds) :]
                else:
                    launching_jobs = bracket_jobs[:index]
                    bracket_jobs = bracket_jobs[index:]
                    to_launch = overrides[:index]
                    overrides = overrides[index:]

                if len(to_launch) == 0:
                    break
                if verbose:
                    log.info(
                        "Launching configurations with budget {} in bracket {}".format(
                            int(budget) + 1, bracket.bracket_id
                        )
                    )
                self.opt_time += time.time() - opt_time_start
                res = self.launcher.launch(to_launch, initial_job_idx=self.trial_id)
                self.trial_id += len(to_launch)

                for i in range(len(launching_jobs)):
                    launching_jobs[i]["cost"] = launching_jobs[i]["budget"]
                done = False
                while not done:
                    for i in range(len(to_launch)):
                        if res[i].status.name == "COMPLETED":
                            res[i].return_value
                            done = True
                        else:
                            done = False

                if self.seeds:
                    for i in range(0, len(launching_jobs)):
                        launching_jobs[i]["fitness"] = np.mean(
                            [res[i * len(self.seeds) + j].return_value for j in range(len(self.seeds))]
                            [
                                res[i * len(self.seeds) + j].return_value
                                for j in range(len(self.seeds))
                            ]
                        if self.maximize:
                            launching_jobs[i]["fitness"] = -launching_jobs[i]["fitness"]
                        self.futures.append(launching_jobs[i])
                        log.info(
                            f'Finished job across {len(self.seeds)} seeds with mean fitness \
                                {round(launching_jobs[i]["fitness"], 2)} and \
                                    mean cost {round(launching_jobs[i]["cost"], 2)}'
                        )
                else:
                    for i in range(len(to_launch)):
                        launching_jobs[i]["fitness"] = res[i].return_value
                        if self.maximize:
                            launching_jobs[i]["fitness"] = -launching_jobs[i]["fitness"]
                        self.futures.append(launching_jobs[i])
                        log.info(
                            f'Finished job with fitness {round(launching_jobs[i]["fitness"], 2)} and \
                                cost {round(launching_jobs[i]["cost"], 2)}'
                        )

                if self.deepcave:
                    for job in launching_jobs:
                        self.deepcave_recorder.start(config=job["cs_config"], budget=job["budget"])
                        self.deepcave_recorder.start(
                            config=job["cs_config"], budget=job["budget"]
                        )
                        self.deepcave_recorder.end(
                            costs=job["fitness"],
                            config=job["cs_config"],
                            budget=job["budget"],
                        )
                opt_time_start = time.time()
                self._fetch_results_from_workers()
                if verbose:
                    self._verbosity_runtime(fevals, brackets, total_cost, total_time_cost)
                    self._verbosity_runtime(
                        fevals, brackets, total_cost, total_time_cost
                    )
                    log.info(
                        "Best score seen/Incumbent score: {}".format(
                            np.round(self.inc_score, decimals=2)
                        )
                    )
                if self.inc_config is not None:
                    self._save_incumbent(name)
                if save_history and self.history is not None:
                    self._save_history(name)
                self.checkpoint_dehb()
                self.clean_inactive_brackets()
                self.opt_time += time.time() - opt_time_start
            log.info("Bracket finished")
            # end of while

        if verbose and len(self.futures) > 0:
            log.info("DEHB optimisation over! Waiting to collect results from workers running...")
            log.info(
                "DEHB optimisation over! Waiting to collect results from workers running..."
            )
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent(name)
            if save_history and self.history is not None:
                self._save_history(name)
            time.sleep(0.05)  # waiting 50ms

        if verbose:
            time_taken = time.time() - self.start
            log.info(
                "End of optimisation! Total duration: {}s; Optimization overhead: {}s; Total fevals: {}\n".format(
                    np.round(time_taken, decimals=2), np.round(self.opt_time, decimals=2), len(self.traj)
                    np.round(time_taken, decimals=2),
                    np.round(self.opt_time, decimals=2),
                    len(self.traj),
            )
            log.info("Incumbent score: {}".format(np.round(self.inc_score, decimals=2)))
            log.info("Incumbent config: ")
            if self.configspace:
                log.info(self.inc_config)
                config = self.vector_to_configspace(self.inc_config)
                for k, v in config.get_dictionary().items():
                    log.info("{}: {}".format(k, v))
            else:
                log.info("{}".format(self.inc_config))
        self._save_incumbent(name)
        self._save_history(name)
        return config  # np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)
