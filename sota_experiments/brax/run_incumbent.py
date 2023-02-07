# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
import numpy as np
import gym
from train_brax import train_brax


@hydra.main(version_base="1.1", config_path="configs", config_name="dehb_halfcheetah_prelim")
def run_rs(cfg):
    return train_brax(cfg)


if __name__ == "__main__":
    run_rs()
