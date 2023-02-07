# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
import numpy as np
import gym
from train_brax import train_brax


@hydra.main(version_base="1.1", config_path="configs", config_name="bgt")
def run_rs(cfg):
    try:
        return train_brax(cfg)
    except:
        return 5000


if __name__ == "__main__":
    run_rs()
