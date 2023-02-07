# This code is from the notebook in the Brax repository under: https://github.com/google/brax/blob/main/notebooks/training_torch.ipynb
# It has been adapted to run with hydra, checkpointing is now available and some more hyperparameters are now configurable

import gym
import collections
import functools
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence
import hydra
import os
import logging
import numpy as np

import brax
from brax import envs
from brax.envs import to_torch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    v = torch.ones(1, device="cuda")


class Agent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        epsilon: float,
        lambda_: float,
        vf_coef: float,
        device: str,
    ):
        super(Agent, self).__init__()

        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(nn.Linear(w1, w2))
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)

        value = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(nn.Linear(w1, w2))
            value.append(nn.SiLU())
        value.pop()  # drop the final activation
        self.value = nn.Sequential(*value)

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(policy_layers[0], device=device)
        self.running_variance = torch.zeros(policy_layers[0], device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.value_cost = vf_coef
        self.device = device

    # @torch.jit.export
    def dist_create(self, logits):
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    # @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    # @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    # @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    # @torch.jit.export
    def update_normalization(self, observation):
        observation = observation.to(self.device)
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean.to(self.device)
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean.to(self.device) + mean_diff
        input_to_new_mean = observation - self.running_mean.to(self.device)
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance.to(self.device) + var_diff

    # @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6).to(self.device)
        observation = observation.to(self.device)
        return ((observation - self.running_mean.to(self.device)) / variance.sqrt()).clip(-5, 5)

    # @torch.jit.export
    def get_logits_action(self, observation):
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    # @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        deltas = reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = deltas[ti] + self.discounting * (1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (reward + self.discounting * (1 - termination) * vs_t_plus_1 - values) * truncation_mask
        return vs, advantages

    # @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor]):
        observation = self.normalize(td["observation"])
        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td["reward"] * self.reward_scaling
        termination = td["done"] * (1 - td["truncation"])

        loc, scale = self.dist_create(td["logits"])
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, td["action"])
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td["action"])

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * self.value_cost

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy

        return policy_loss + v_loss + entropy_loss


StepData = collections.namedtuple("StepData", ("observation", "logits", "action", "reward", "done", "truncation"))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)


def eval_unroll(agent, env, length):
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    observation = torch.from_numpy(np.asarray(observation))
    episodes = torch.zeros((), device=agent.device)
    episode_reward = torch.zeros((), device=agent.device)
    for _ in range(length):
        _, action = agent.get_logits_action(observation)
        observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
        observation = torch.from_numpy(np.asarray(observation))
        reward = torch.from_numpy(np.asarray(reward))
        done = torch.from_numpy(np.asarray(done))
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
    return episodes, episode_reward / episodes


def train_unroll(agent, env, observation, num_unrolls, unroll_length):
    """Return step data over multple unrolls."""
    sd = StepData([], [], [], [], [], [])
    observation = torch.from_numpy(np.asarray(observation))
    for _ in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for _ in range(unroll_length):
            logits, action = agent.get_logits_action(observation)
            observation, reward, done, info = env.step(Agent.dist_postprocess(action))
            observation = torch.from_numpy(np.asarray(observation))
            reward = torch.from_numpy(np.asarray(reward))
            done = torch.from_numpy(np.asarray(done))
            one_unroll.observation.append(observation)
            one_unroll.logits.append(logits)
            one_unroll.action.append(action)
            one_unroll.reward.append(reward)
            one_unroll.done.append(done)
            one_unroll.truncation.append(torch.from_numpy(np.asarray(info["truncation"])))
        one_unroll = sd_map(torch.stack, one_unroll)
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
    td = sd_map(torch.stack, sd)
    return observation, td


def train(
    env_name: str = "ant",
    num_envs: int = 2048,
    episode_length: int = 1000,
    num_timesteps: int = 30_000_000,
    unroll_length: int = 5,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = 0.1,
    entropy_cost: float = 1e-2,
    discounting: float = 0.97,
    learning_rate: float = 3e-4,
    gae_lambda: float = 0.95,
    epsilon: float = 0.3,
    vf_coef: float = 0.5,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    outdir: str = ".",
    load_path: str = None,
    save_path: str = None,
    use_cuda: bool = True,
):
    """Trains a policy via PPO."""
    gym_name = f"brax-{env_name}-v0"
    if gym_name not in gym.envs.registry.env_specs:
        entry_point = functools.partial(envs.create_gym_env, env_name=env_name)
        gym.register(gym_name, entry_point=entry_point)
    env = gym.make(gym_name, batch_size=num_envs, episode_length=episode_length)
    # automatically convert between jax ndarrays and torch tensors:
    env = to_torch.JaxToTorchWrapper(env, device=device)

    # env warmup
    env.reset()
    action = torch.zeros(env.action_space.shape).to(device)
    env.step(action)

    # create the agent
    policy_layers = [env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2]
    value_layers = [env.observation_space.shape[-1], 64, 64, 1]
    agent = Agent(
        policy_layers, value_layers, entropy_cost, discounting, reward_scaling, epsilon, gae_lambda, vf_coef, device
    )
    # agent = torch.jit.script(agent.to(device))
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Loading
    if load_path is not None:
        checkpoint = torch.load(load_path)
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    sps = 0
    total_steps = 0
    observation = env.reset()
    num_steps = batch_size * num_minibatches * unroll_length
    num_epochs = max(2, num_timesteps // num_steps)
    num_unrolls = max(1, batch_size * num_minibatches // env.num_envs)
    t = time.time()
    for _ in range(num_epochs):
        observation, td = train_unroll(agent, env, observation, num_unrolls, unroll_length)
        # make unroll first
        def unroll_first(data):
            data = data.swapaxes(0, 1)
            return data.reshape([data.shape[0], -1] + list(data.shape[3:]))

        td = sd_map(unroll_first, td)

        # update normalization statistics
        agent.update_normalization(td.observation)

        for _ in range(num_update_epochs):
            # shuffle and batch the data
            with torch.no_grad():
                permutation = torch.randperm(td.observation.shape[1], device=device)

                def shuffle_batch(data):
                    data = data.to(device)[:, permutation]
                    data = data.reshape([data.shape[0], num_minibatches, -1] + list(data.shape[2:]))
                    return data.swapaxes(0, 1)

                epoch_td = sd_map(shuffle_batch, td)

            for minibatch_i in range(num_minibatches):
                td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
                loss = agent.loss(td_minibatch._asdict())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # total_loss += loss.detach()

    duration = time.time() - t
    total_steps += num_epochs * num_steps
    # total_loss = total_loss / max(num_epochs * num_update_epochs * num_minibatches, 1)
    sps = num_epochs * num_steps / duration

    logging.info("Finished training")
    if progress_fn:
        t = time.time()
        with torch.no_grad():
            episode_count, episode_reward = eval_unroll(agent, env, episode_length)
        duration = time.time() - t
        episode_avg_length = env.num_envs * episode_length / episode_count
        eval_sps = env.num_envs * episode_length / duration
        progress = {
            "eval/episode_reward": episode_reward,
            "eval/completed_episodes": episode_count,
            "eval/avg_episode_length": episode_avg_length,
            "speed/sps": sps,
            "speed/eval_sps": eval_sps,
            # "losses/total_loss": total_loss,
        }
        progress_fn(total_steps, progress)

    logging.info(f"Finished eval with {episode_reward.detach().cpu().numpy()}")
    # Checkpoint
    state = {
        "agent": agent.state_dict(),
        "optimizer": torch.optim.Adam(agent.parameters(), lr=learning_rate).state_dict(),
    }
    if save_path is not None:
        path = os.path.join(outdir, save_path)
        torch.save(state, path)

    return episode_reward.detach().cpu().numpy()


@hydra.main(version_base="1.1", config_path="configs", config_name="base")
def train_brax(cfg):
    logging.info(cfg)
    logging.info(f"Training for {cfg.num_timesteps} steps on {cfg.env_name}")
    with open(os.path.join(cfg.outdir, "log.csv"), "a+") as f:
        f.write("total_steps,n_eval_episodes,eval_reward,average_eval_episode_length,total_loss\ns")

    def progress(num_steps, metrics):
        reward = metrics["eval/episode_reward"].cpu()
        n_eps = metrics["eval/completed_episodes"].cpu()
        ep_length = metrics["eval/avg_episode_length"].cpu()
        loss = 0  # metrics["losses/total_loss"]
        with open(os.path.join(cfg.outdir, "log.csv"), "a+") as f:
            f.write(f"{num_steps},{n_eps},{reward},{ep_length},{loss}\n")

    final_eval_reward = train(
        env_name=cfg.env_name,
        num_envs=2048,
        episode_length=cfg.episode_length,
        num_timesteps=cfg.num_timesteps,  # 30_000_000,
        unroll_length=cfg.unroll_length,
        batch_size=cfg.batch_size,
        num_minibatches=2**cfg.num_minibatches,
        num_update_epochs=cfg.num_update_epochs,
        reward_scaling=cfg.reward_scaling,
        entropy_cost=cfg.entropy_cost,
        discounting=0.97,
        learning_rate=cfg.learning_rate,
        gae_lambda=cfg.gae_lambda,
        epsilon=cfg.epsilon,
        vf_coef=cfg.vf_coef,
        progress_fn=progress,
        outdir=cfg.outdir,
        load_path=cfg.load,
        save_path=cfg.save,
        use_cuda=cfg.use_cuda,
    )
    logging.info(f"Final eval reward on seed {cfg.seed} was {final_eval_reward}")
    return -final_eval_reward


if __name__ == "__main__":
    train_brax()
