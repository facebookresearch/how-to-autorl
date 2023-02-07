from hydra_plugins.hydra_pbt_sweeper.pbt_sweeper import PBTSweeper
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from utils import run_short_pbt
import shutil


def test_sweepers_found() -> None:
    assert PBTSweeper.__name__ in [x.__name__ for x in Plugins.instance().discover(Sweeper)]


def test_warmstarting():
    outdir = "./tests/warmstarting_test"
    extras = [
        "+hydra.sweeper.pbt_kwargs.warmstart=true",
        "+hydra.sweeper.pbt_kwargs.init_size=4",
    ]
    run_short_pbt(outdir, extras)
    shutil.rmtree(outdir, ignore_errors=True)


def test_pb2_example():
    outdir = "./tests/pb2_test"
    extras = ["hydra.sweeper.optimizer=pb2"]
    run_short_pbt(outdir, extras)
    shutil.rmtree(outdir, ignore_errors=True)


def test_bgt_example():
    outdir = "./tests/bgt_test"
    extras = [
        "hydra.sweeper.optimizer=bgt",
        "+hydra.sweeper.pbt_kwargs.num_config_changes=1",
        "+hydra.sweeper.pbt_kwargs.init_size=4",
    ]
    run_short_pbt(outdir, extras)
    shutil.rmtree(outdir, ignore_errors=True)
