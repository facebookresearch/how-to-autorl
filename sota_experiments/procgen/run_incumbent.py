# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
import os
from idaac.train import train


@hydra.main(config_path="configs", config_name="dehb_climber_with_arch", version_base="1.1")
def run_idaac(cfg):
    if cfg.load_dir:
        cfg.load_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.load_dir)
    if cfg.save_dir:
        cfg.save_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.save_dir)
    print(cfg)
    train(cfg)


if __name__ == "__main__":
    run_idaac()
