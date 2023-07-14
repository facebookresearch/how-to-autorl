# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import shutil
import time

import ConfigSpace as CS
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from pytest import mark

from dummy_dehb_function import run_dummy
from hydra_plugins.hydra_dehb_sweeper.dehb_sweeper import DEHBSweeper
from hydra_plugins.hydra_dehb_sweeper.hydra_dehb import HydraDEHB
from utils import run_short_dehb


def test_sweeper_found() -> None:
    assert DEHBSweeper.__name__ in [x.__name__ for x in Plugins.instance().discover(Sweeper)]


@mark.parametrize("cutoff", [(10), (30), (60)])
def test_termination_time(cutoff):
    outdir = "./tests/time_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraDEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        f=run_dummy,
        cs=cs,
        min_budget=2,
        max_budget=3,
        dimensions=0,
    )
    run_short_dehb(outdir, termination=[f"+hydra.sweeper.total_time_cost={cutoff}"])
    end_time = time.time()
    buffer = 10
    sweeper.load_dehb(os.path.join(outdir, "dehb_state.pkl"))
    assert end_time - sweeper.start < cutoff + buffer
    shutil.rmtree(outdir, ignore_errors=True)


def test_termination_steps():
    outdir = "./tests/step_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraDEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        f=run_dummy,
        cs=cs,
        min_budget=2,
        max_budget=3,
        dimensions=0,
    )
    run_short_dehb(outdir, termination=["+hydra.sweeper.total_cost=5"])
    sweeper.load_dehb(os.path.join(outdir, "dehb_state.pkl"))
    assert sweeper.current_total_steps < 6
    shutil.rmtree(outdir, ignore_errors=True)


def test_termination_brackets():
    outdir = "./tests/bracket_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraDEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        f=run_dummy,
        cs=cs,
        min_budget=2,
        max_budget=3,
        dimensions=0,
    )
    run_short_dehb(outdir, termination=["+hydra.sweeper.total_brackets=1"])
    sweeper.load_dehb(os.path.join(outdir, "dehb_state.pkl"))
    assert len(sweeper.active_brackets) == 1
    assert all([bracket.is_bracket_done() for bracket in sweeper.active_brackets])
    shutil.rmtree(outdir, ignore_errors=True)


def test_termination_fevals():
    outdir = "./tests/fevals_termination_test"
    cs = CS.ConfigurationSpace(
        space={
            "a": CS.Float("a", bounds=(0.1, 1.5), log=True),
        }
    )
    sweeper = HydraDEHB(
        global_overrides=[],
        global_config=None,
        launcher=None,
        budget_variable="b",
        n_jobs=3,
        base_dir=outdir,
        f=run_dummy,
        cs=cs,
        min_budget=2,
        max_budget=3,
        dimensions=0,
    )
    run_short_dehb(outdir, termination=["+hydra.sweeper.total_function_evaluations=2"])
    sweeper.load_dehb(os.path.join(outdir, "dehb_state.pkl"))
    assert len(sweeper.traj) == 2
    shutil.rmtree(outdir, ignore_errors=True)
