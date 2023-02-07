# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import gym
import hydra
import wandb
import numpy as np
import time
import torch
import stable_baselines3
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from procgen import ProcgenEnv, ProcgenGym3Env
from procgen.env import ENV_NAMES
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize,
    VecEnvWrapper,
    VecEnv,
)
from stable_baselines3.common.callbacks import BaseCallback
import brax
from brax import envs
from brax.envs import to_torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__copyright__ = "Copyright 2022, Theresa Eimer"
__license__ = "3-clause BSD"

log = logging.getLogger(__name__)

# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = torch.nn.functional.relu(x)
        x = self.conv0(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(torch.nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = torch.nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ResNetCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        h, w, c = observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            torch.nn.Flatten(),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=shape[0] * shape[1] * shape[2], out_features=features_dim
            ),
            torch.nn.ReLU(),
        ]
        self.network = torch.nn.Sequential(*conv_seqs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.permute((0, 3, 1, 2)))  # / 255.0)


class BraxToSBWrapper(VecEnv):
    def __init__(self, env, batch_size=512):
        self.env = env
        self.observation_space = gym.spaces.Box(
            self.env.observation_space.low[0], self.env.observation_space.high[0]
        )
        self.action_space = gym.spaces.Box(
            self.env.action_space.low[0], self.env.action_space.high[0]
        )
        self.num_envs = batch_size
        # VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        state, reward, done, info = self.env.step(self.actions)
        del info["first_qp"]
        del info["first_obs"]
        info = [
            {key: value[index] for key, value in info.items()}
            for index in range(max(map(len, info.values())))
        ]
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        state = self.env.reset()
        return np.asarray(state)

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return getattr(self.env, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(self.env, wrapper_class)]


class BufferedVecMonitor(VecMonitor):
    def __init__(self, env, filename, info_keywords=(), buffer_size=100) -> None:
        super().__init__(env, filename, info_keywords)
        self.buffer_size = buffer_size
        self.return_buffer = []
        self.length_buffer = []

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                self.return_buffer.append(episode_return)
                self.length_buffer.append(episode_length)
                if len(self.return_buffer) > self.buffer_size:
                    self.return_buffer = self.return_buffer[1:]
                    self.length_buffer = self.length_buffer[1:]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6),
                }
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos

    def get_episode_rewards(self):
        return self.return_buffer

    def get_episode_lengths(self):
        return self.length_buffer

    def reset_buffers(self):
        self.length_buffer = []
        self.return_buffer = []


class ToBaselinesVecEnv(VecEnv):
    """
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """

    def __init__(self, env):
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                name: gym.spaces.Box(
                    shape=subspace.shape,
                    low=0,
                    high=subspace.eltype.n - 1,
                    dtype=np.uint8,
                )
                for (name, subspace) in self.env.ob_space.items()
            }
        )
        self.action_space = gym.spaces.Discrete(self.env.ac_space.eltype.n)

    def reset(self):
        _rew, ob, first = self.env.observe()
        if not first.all():
            print("Warning: manual reset ignored")
        return ob

    def step_async(self, ac):
        self.env.act(ac)

    def step_wait(self):
        rew, ob, first = self.env.observe()
        return ob, rew, first, self.env.get_info()

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    @property
    def num_envs(self):
        return self.env.num

    def render(self, mode="human"):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()[0]
        if mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def close(self):
        pass

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return getattr(self.env, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(self.env, wrapper_class)]

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0, custom_vec_monitor=False):
        super().__init__(verbose)
        self.custom_vec_monitor = custom_vec_monitor

    def _on_step(self):
        if self.custom_vec_monitor:
            env = self.model.env
        else:
            env = self.model.env.envs[0]
        rewards = env.get_episode_rewards()
        lengths = env.get_episode_lengths()
        if len(rewards) > 0:
            if self.custom_vec_monitor:
                wandb.log({"reward": np.mean(rewards), "length": np.mean(lengths)})
                self.model.env.reset_buffers()
            else:
                wandb.log({"reward": rewards[-1], "length": lengths[-1]})
        return True


class CastToNumpyWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            self.env.observation_space.low[0], self.env.observation_space.high[0]
        )

    def reset(self):
        state = self.env.reset()
        return state.numpy()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state.numpy(), reward.numpy(), done.numpy(), info


@hydra.main(config_path="configs", config_name="base")
def train_sb3(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info(
        f"Training {cfg.algorithm.agent_class} Agent on {cfg.env_name} for {cfg.algorithm.total_timesteps} steps"
    )

    model_kwargs = OmegaConf.to_container(cfg.algorithm.model_kwargs)
    schedule = any([isinstance(v, list) for v in model_kwargs.values()])
    if schedule:
        schedule_hps = [
            k for k in model_kwargs.keys() if isinstance(model_kwargs[k], list)
        ]
        schedule_values = [v for v in model_kwargs.values() if isinstance(v, list)]
        schedule_length = len(model_kwargs[schedule_hps[0]])
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
        if cfg.algorithm.agent_class == "PPO":
            model_kwargs["policy_kwargs"] = {
                "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
                "activation_fn": torch.nn.SiLU,
            }
        else:
            model_kwargs["policy_kwargs"] = {
                "net_arch": [64, 64],
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

    if schedule:
        for t in range(schedule_length):
            for i, k in enumerate(schedule_hps):
                model_kwargs[k] = schedule_values[i][t]

            if t != 0:
                load_path = cfg.save if cfg.save else "./checkpoint.pt"
                model = agent_class.load(load_path, env=env, **model_kwargs, verbose=2)
            elif cfg.load:
                model = agent_class.load(cfg.load, env=env, **model_kwargs, verbose=2)
            else:
                model = agent_class(
                    cfg.algorithm.policy_model, env, **model_kwargs, verbose=2
                )
            model.learn(
                total_timesteps=cfg.algorithm.total_timesteps // schedule_length,
                reset_num_timesteps=False,
            )
            log.info(t)
            if cfg.save:
                model.save(cfg.save)
            else:
                model.save("./checkpoint.pt")
    else:
        if cfg.load:
            model = agent_class.load(cfg.load, env=env, **model_kwargs, verbose=2)
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
            model.learn(
                total_timesteps=cfg.algorithm.total_timesteps,
                reset_num_timesteps=False,
                callback=WandbCallback(
                    custom_vec_monitor=cfg.env_name in ENV_NAMES
                    or cfg.env_name in ["ant", "halfcheetah", "humanoid"]
                ),
            )
        else:
            model.learn(
                total_timesteps=cfg.algorithm.total_timesteps, reset_num_timesteps=False
            )

        if cfg.save:
            model.save(cfg.save)

    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=cfg.algorithm.n_eval_episodes,
    )
    log.info(
        f"Mean evaluation reward at the end of training across {cfg.algorithm.n_eval_episodes} episodes was {mean_reward}+-{std_reward}"
    )
    if cfg.reward_curves:
        episode_rewards = [-r for r in env.get_episode_rewards()]
        return episode_rewards
    else:
        return -mean_reward


if __name__ == "__main__":
    train_sb3()
