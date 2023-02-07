from hydra.test_utils.test_utils import run_python_script


def run_short_dehb(outdir, extras=None, termination=["+hydra.sweeper.total_function_evaluations=2"]):
    cmd = [
        "tests/dummy_dehb_function.py",
        "--multirun",
        f"hydra.sweep.dir={outdir}",
        "+hydra.sweeper.search_space={hyperparameters: {x: {type: uniform_float, upper:2, lower: 0}}}",
        "+hydra.sweeper.budget_variable=b",
        "+hydra.sweeper.dehb_kwargs.min_budget=1",
        "+hydra.sweeper.dehb_kwargs.max_budget=5",
    ]
    cmd += termination
    if extras:
        cmd += extras
    run_python_script(cmd, allow_warnings=True)


def run_short_pbt(outdir, extras=None):
    cmd = [
        "tests/dummy_pbt_function.py",
        "--multirun",
        f"hydra.sweep.dir={outdir}",
        "+hydra.sweeper.search_space={hyperparameters: {x: {type: uniform_float, upper:2, lower: 0}}}",
        "+hydra.sweeper.budget_variable=b",
        "+hydra.sweeper.pbt_kwargs.population_size=2",
        "+hydra.sweeper.budget=1",
        "+hydra.sweeper.pbt_kwargs.config_interval=1",
        "+hydra.sweeper.saving_variable=save",
        "+hydra.sweeper.loading_variable=load",
    ]
    if extras:
        cmd += extras
    run_python_script(cmd, allow_warnings=True)
