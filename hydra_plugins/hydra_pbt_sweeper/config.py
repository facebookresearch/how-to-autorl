# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, Optional

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class PBTSweeperConfig:
    _target_: str = "hydra_plugins.hydra_pbt_sweeper.pbt_sweeper.PBTSweeper"
    search_space: Optional[Dict] = field(default_factory=dict)
    resume: Optional[str] = None
    optimizer: Optional[str] = "pbt"
    budget: Optional[Any] = None
    budget_variable: Optional[str] = None
    loading_variable: Optional[str] = None
    saving_variable: Optional[str] = None
    pbt_kwargs: Optional[Dict] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="PBT",
    node=PBTSweeperConfig,
    provider="hydra_pbt_sweeper",
)
