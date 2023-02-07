# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import gym
import hydra
import wandb
import numpy as np
import torch
import stable_baselines3
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from procgen import ProcgenEnv, ProcgenGym3Env
from procgen.env import ENV_NAMES
from stable_baselines3.common.vec_env import VecExtractDictObs
import brax
from brax import envs
from brax.envs import to_torch
from run_agent import (
    WandbCallback,
    BufferedVecMonitor,
    ToBaselinesVecEnv,
    ResNetCNN,
    BraxToSBWrapper,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="rs")
def train_sb3(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info(
        f"Training {cfg.algorithm.agent_class} Agent on {cfg.env_name} for {cfg.algorithm.total_timesteps} steps"
    )
    perfs = []
    for s in cfg.tuning_seeds:
        cfg.seed = s
        model_kwargs = OmegaConf.to_container(cfg.algorithm.model_kwargs)
        if "train_freq" in model_kwargs.keys():
            model_kwargs["train_freq"] = int(model_kwargs["train_freq"])
        if "buffer_size" in model_kwargs.keys():
            model_kwargs["buffer_size"] = int(model_kwargs["buffer_size"])
        if "learning_starts" in model_kwargs.keys():
            model_kwargs["learning_starts"] = int(model_kwargs["learning_starts"])
        if "gradient_steps" in model_kwargs.keys():
            model_kwargs["gradient_steps"] = int(model_kwargs["gradient_steps"])
        if cfg.env_name in ENV_NAMES:
            model_kwargs["policy_kwargs"]["features_extractor_class"] = ResNetCNN
            venv = ToBaselinesVecEnv(
                ProcgenGym3Env(
                    num=64,
                    env_name=cfg.env_name,
                    distribution_mode="easy",
                    num_levels=0,
                    start_level=0,
                    # rand_seed=cfg.seed,
                )
            )
            venv = VecExtractDictObs(venv, "rgb")
            venv.is_vector_env = True
            # venv = VecNormalize(venv=venv, norm_obs=False)
            venv = gym.wrappers.NormalizeReward(venv)
            venv = gym.wrappers.TransformReward(
                venv, lambda reward: np.clip(reward, -10, 10)
            )
            env = BufferedVecMonitor(
                env=venv, filename=f"{cfg.log_dir}/logs.txt", buffer_size=10000
            )
        elif cfg.env_name.startswith("MiniGrid"):
            env = gym.make(cfg.env_name)
            env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)
            env.reset()
            env = Monitor(env, cfg.log_dir)
        elif cfg.env_name in ["ant", "halfcheetah", "humanoid"]:
            model_kwargs["policy_kwargs"] = {
                "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
                "activation_fn": torch.nn.SiLU,
            }
            env = envs.create_gym_env(env_name=cfg.env_name, batch_size=1024)
            env = to_torch.JaxToTorchWrapper(env)
            env = BraxToSBWrapper(env, batch_size=1024)
            env = BufferedVecMonitor(
                env=env, filename=f"{cfg.log_dir}/logs.txt", buffer_size=10000
            )
            env.reset()
        else:
            env = gym.make(cfg.env_name)
            env = Monitor(env, cfg.log_dir)

        agent_class = getattr(stable_baselines3, cfg.algorithm.agent_class)

        if cfg.load:
            try:
                model = agent_class.load(cfg.load, env=env, **model_kwargs, verbose=2)
            except:
                return 5000
        else:
            model = agent_class(
                cfg.algorithm.policy_model, env, **model_kwargs, verbose=2
            )

        if cfg.wandb:
            wandb_config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                project="autorl-benchmarks",
                entity="theeimer",
                save_code=True,
                config=wandb_config,
                tags=cfg.wandb_tags,
            )
            try:
                model.learn(
                    total_timesteps=cfg.algorithm.total_timesteps,
                    reset_num_timesteps=False,
                    callback=WandbCallback(
                        custom_vec_monitor=cfg.env_name in ENV_NAMES
                        or cfg.env_name in ["ant", "halfcheetah", "humanoid"]
                    ),
                )
            except:
                if cfg.save:
                    model.save(cfg.save)
                return 5000
        else:
            try:
                model.learn(
                    total_timesteps=cfg.algorithm.total_timesteps,
                    reset_num_timesteps=False,
                )
            except:
                if cfg.save:
                    model.save(cfg.save)
                return 5000

        if cfg.save:
            model.save(cfg.save)

        try:
            mean_reward, std_reward = evaluate_policy(
                model,
                model.get_env(),
                n_eval_episodes=cfg.algorithm.n_eval_episodes,
            )
            log.info(
                f"Mean evaluation reward at the end of training across {cfg.algorithm.n_eval_episodes} episodes was {mean_reward}+-{std_reward}"
            )
            perfs.append(-mean_reward)
        except:
            perfs.append(5000)

    return np.mean(perfs)


if __name__ == "__main__":
    train_sb3()
