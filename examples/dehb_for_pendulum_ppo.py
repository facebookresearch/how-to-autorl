import hydra
from examples.sb3_agent import train_sb3


@hydra.main(config_path="configs", config_name="ppo_pendulum_dehb")
def run_ppo_dehb(cfg):
    return train_sb3(cfg)


if __name__ == "__main__":
    run_ppo_dehb()
