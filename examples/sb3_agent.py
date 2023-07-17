# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import gymnasium as gym
import hydra
import stable_baselines3
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

__copyright__ = "Copyright 2022, Theresa Eimer"
__license__ = "3-clause BSD"

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="dqn_cartpole_pb2")
def train_sb3(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    log.info(f"Training {cfg.algorithm.agent_class} Agent on {cfg.env_name} for {cfg.algorithm.total_timesteps} steps")
    env = gym.make(cfg.env_name)
    if cfg.reward_curves:
        env = Monitor(env, ".")

    agent_class = getattr(stable_baselines3, cfg.algorithm.agent_class)

    if cfg.load:
        model = agent_class.load(cfg.load, env=env, **cfg.algorithm.model_kwargs)
    else:
        model = agent_class(cfg.algorithm.policy_model, env, **cfg.algorithm.model_kwargs)

    model.learn(total_timesteps=cfg.algorithm.total_timesteps, reset_num_timesteps=False)

    if cfg.save:
        model.save(cfg.save)

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=cfg.algorithm.n_eval_episodes,
    )
    log.info(
        f"Mean evaluation reward at the end of training across {cfg.algorithm.n_eval_episodes} episodes was {mean_reward}"
    )
    if cfg.reward_curves:
        episode_rewards = [-r for r in env.get_episode_rewards()]
        return episode_rewards
    else:
        return -mean_reward


if __name__ == "__main__":
    train_sb3()
