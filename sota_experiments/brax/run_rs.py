# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
import numpy as np
import gym
from train_brax import train_brax


@hydra.main(version_base="1.1", config_path="configs", config_name="rs")
def run_rs(cfg):
    perfs = []
    for s in cfg.tuning_seeds:
        cfg.seed = s
        perfs.append(train_brax(cfg))
    return np.mean(perfs)


if __name__ == "__main__":
    run_rs()
