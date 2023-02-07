# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, Optional

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DEHBSweeperConfig:
    _target_: str = "hydra_plugins.hydra_dehb_sweeper.dehb_sweeper.DEHBSweeper"
    search_space: Dict[str, Any] = field(default_factory=dict)
    dehb_kwargs: Optional[Dict] = field(default_factory=dict)
    budget_variable: Optional[str] = None
    resume: Optional[str] = None
    n_jobs: Optional[int] = 8
    slurm: Optional[bool] = False
    slurm_timeout: Optional[int] = 10
    total_function_evaluations: Optional[int] = None
    total_brackets: Optional[int] = None
    total_cost: Optional[int] = None


ConfigStore.instance().store(group="hydra/sweeper", name="DEHB", node=DEHBSweeperConfig, provider="hydra_dehb_sweeper")
