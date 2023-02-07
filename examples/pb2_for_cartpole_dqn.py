import hydra
from examples.sb3_agent import train_sb3


@hydra.main(config_path="configs", config_name="dqn_cartpole_pb2")
def run_dqn_pb2(cfg):
    return train_sb3(cfg)


if __name__ == "__main__":
    run_dqn_pb2()
