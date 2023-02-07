# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from examples.sb3_agent import train_sb3


@hydra.main(config_path="configs", config_name="sac_mountaincar_pbt")
def run_sac_pbt(cfg):
    return train_sb3(cfg)


if __name__ == "__main__":
    run_sac_pbt()
