# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import gymnasium as gym
import hydra
import torch
import stable_baselines3
from omegaconf import OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="sac_mountaincar_bgt")
def run_sac_pbt(cfg):
    log.info(OmegaConf.to_yaml(cfg))

    log.info(f"Training {cfg.algorithm.agent_class} Agent on {cfg.env_name} for {cfg.algorithm.total_timesteps} steps")
    policy_width = cfg.nas_hidden_size_p
    policy_layers = cfg.nas_num_layers_p
    value_width = cfg.nas_hidden_size_v
    value_layers = cfg.nas_num_layers_v
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[policy_width] * policy_layers, qf=[value_width] * value_layers),
        use_sde=False,
    )

    env = gym.make(cfg.env_name)
    if cfg.reward_curves:
        env = Monitor(env, ".")

    path = "./agent_logs"
    logger = configure(path, ["stdout", "csv"])

    agent_class = getattr(stable_baselines3, cfg.algorithm.agent_class)

    if cfg.load:
        model = agent_class.load(cfg.load, env=env, policy_kwargs=policy_kwargs, **cfg.algorithm.model_kwargs)
    else:
        model = agent_class(cfg.algorithm.policy_model, env, policy_kwargs=policy_kwargs, **cfg.algorithm.model_kwargs)

    model.set_logger(logger)
    model.learn(total_timesteps=cfg.algorithm.total_timesteps, reset_num_timesteps=False)

    if cfg.save:
        model.save(cfg.save)

    mean_reward, _ = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=cfg.algorithm.n_eval_episodes,
    )
    log.info(
        f"Mean evaluation reward at the end of training across {cfg.algorithm.n_eval_episodes} episodes was {mean_reward}"
    )
    return -mean_reward


if __name__ == "__main__":
    run_sac_pbt()
