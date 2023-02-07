import os
import shutil
from utils import run_short_dehb, run_short_pbt
from omegaconf import OmegaConf
from hydra_plugins.hydra_dehb_sweeper.hydra_dehb import HydraDEHB
from hydra_plugins.hydra_pbt_sweeper.hydra_pbt import HydraPBT


def test_deepcave_logs():
    outdir = "./tests/dummy_logs"
    run_short_dehb(outdir, ["+hydra.sweeper.dehb_kwargs.deepcave=true"])
    assert os.path.exists(os.path.join(outdir, "deepcave_logs"))
    assert os.path.isfile(os.path.join(outdir, "deepcave_logs/run_1/configs.json"))
    assert os.path.isfile(os.path.join(outdir, "deepcave_logs/run_1/history.jsonl"))
    shutil.rmtree(outdir, ignore_errors=True)

    run_short_pbt(outdir, ["+hydra.sweeper.pbt_kwargs.deepcave=true"])
    assert os.path.exists(os.path.join(outdir, "deepcave_logs"))
    assert os.path.isfile(os.path.join(outdir, "deepcave_logs/run_1/configs.json"))
    assert os.path.isfile(os.path.join(outdir, "deepcave_logs/run_1/history.jsonl"))
    shutil.rmtree(outdir, ignore_errors=True)


def test_wandb_logs():
    try:
        HydraDEHB(
            OmegaConf.create(),
            None,
            None,
            None,
            None,
            base_dir=".",
            f=None,
            cs=None,
            min_budget=2,
            max_budget=3,
            dimensions=0,
            wandb_project="test",
            wandb_tags=["test_tag"],
        )
    except:
        assert False, "wandb init error in DEHB sweeper"

    try:
        HydraPBT(
            OmegaConf.create(),
            None,
            None,
            None,
            None,
            None,
            None,
            cs={},
            config_interval=1,
            num_config_changes=1,
            wandb_project="test",
            wandb_tags=["test_tag"],
        )
    except:
        assert False, "wandb init error in DEHB sweeper"


def test_checkpointing():
    outdir = "./tests/dummy_logs"
    run_short_dehb(outdir)
    assert os.path.isfile(os.path.join(outdir, "dehb_state.pkl"))
    assert os.path.isfile(os.path.join(outdir, "incumbent.json"))
    assert os.path.isfile(os.path.join(outdir, "final_config.yaml"))
    shutil.rmtree(outdir, ignore_errors=True)

    run_short_pbt(outdir)
    assert os.path.isfile(os.path.join(outdir, "pbt_state.pkl"))
    assert os.path.isfile(os.path.join(outdir, "incumbent.json"))
    assert os.path.isfile(os.path.join(outdir, "final_config.yaml"))
    shutil.rmtree(outdir, ignore_errors=True)


def test_restore():
    path = "./tests/dummy_logs"
    run_short_dehb(path)
    sweeper = HydraDEHB(
        None, None, None, None, None, base_dir=path, f=None, cs=None, min_budget=2, max_budget=3, dimensions=0
    )
    sweeper.load_dehb(os.path.join(path, "dehb_state.pkl"))
    assert sweeper.current_total_steps > 0
    assert sweeper.min_budget == 1
    assert sweeper.max_budget == 5
    assert sweeper.cs is not None
    shutil.rmtree(path, ignore_errors=True)

    run_short_pbt(path)
    sweeper = HydraPBT(None, None, None, None, None, None, None, cs={}, config_interval=1, num_config_changes=1)
    sweeper.load_pbt(os.path.join(path, "pbt_state.pkl"))
    assert sweeper.current_steps > 0
    assert sweeper.iteration > 0
    assert len(sweeper.history) > 0
    shutil.rmtree(path, ignore_errors=True)
