# Small changes have been made to the original code to enable model loading and returning the most recent eval reward

import os
import gym
import torch
import numpy as np
import logging
from collections import deque

import idaac.hyperparams as hps
from idaac.test import evaluate
from procgen import ProcgenEnv

from baselines import logger
from baselines.common.vec_env import VecExtractDictObs, VecMonitor, VecNormalize

from idaac.ppo_daac_idaac import algo, utils
from idaac.ppo_daac_idaac.arguments import parser
from idaac.ppo_daac_idaac.model import (
    PPOnet,
    IDAACnet,
    LinearOrderClassifier,
    NonlinearOrderClassifier,
)
from idaac.ppo_daac_idaac.storage import (
    DAACRolloutStorage,
    IDAACRolloutStorage,
    RolloutStorage,
)
from idaac.ppo_daac_idaac.envs import VecPyTorchProcgen


def train(args):
    logging.info(args)
    logging.info(f"Training for {args.num_env_steps} steps on {args.env_name}")
    use_cuda = not args.no_cuda  # and torch.cuda.is_available()
    if args.use_best_hps:
        args.value_epoch = hps.value_epoch[args.env_name]
        args.value_freq = hps.value_freq[args.env_name]
        args.adv_loss_coef = hps.adv_loss_coef[args.env_name]
        args.clf_hidden_size = hps.clf_hidden_size[args.env_name]
        args.order_loss_coef = hps.order_loss_coef[args.env_name]
        if args.env_name in hps.nonlin_envs:
            args.use_nonlinear_clf = True
        else:
            args.use_nonlinear_clf = False
    print("\nArguments: ", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    log_file = "-{}-{}-s{}".format(args.env_name, args.algo, args.seed)
    logger.configure(dir=args.log_dir, format_strs=["csv", "stdout"], log_suffix=log_file)
    print("\nLog File: ", log_file)

    venv = ProcgenEnv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)

    obs_shape = envs.observation_space.shape
    if args.algo == "ppo":
        actor_critic = PPOnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={"hidden_size": args.hidden_size},
        )
    else:
        actor_critic = IDAACnet(
            obs_shape,
            envs.action_space.n,
            base_kwargs={"hidden_size": args.hidden_size},
        )
    logging.info(device)
    actor_critic.to(device)
    print("\n Actor-Critic Network: ", actor_critic)

    if args.algo == "idaac":
        if args.use_nonlinear_clf:
            order_classifier = NonlinearOrderClassifier(emb_size=args.hidden_size, hidden_size=args.clf_hidden_size).to(
                device
            )
        else:
            order_classifier = LinearOrderClassifier(emb_size=args.hidden_size)
        order_classifier.to(device)
        print("\n Order Classifier: ", order_classifier)

    if args.algo == "idaac":
        rollouts = IDAACRolloutStorage(
            args.num_steps,
            args.num_processes,
            envs.observation_space.shape,
            envs.action_space,
        )
    elif args.algo == "daac":
        rollouts = DAACRolloutStorage(
            args.num_steps,
            args.num_processes,
            envs.observation_space.shape,
            envs.action_space,
        )
    else:
        rollouts = RolloutStorage(
            args.num_steps,
            args.num_processes,
            envs.observation_space.shape,
            envs.action_space,
        )

    if args.load_dir:
        checkpoint = torch.load(args.load_dir)
        actor_critic = checkpoint["agent"]
        order_classifier = checkpoint["order_classifier"]
        envs.ob_rms = checkpoint["ob_rms"]

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)
    if args.algo == "idaac":
        agent = algo.IDAAC(
            actor_critic,
            order_classifier,
            args.clip_param,
            args.ppo_epoch,
            args.value_epoch,
            args.value_freq,
            args.num_mini_batch,
            args.value_loss_coef,
            args.adv_loss_coef,
            args.order_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.algo == "daac":
        agent = algo.DAAC(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.value_epoch,
            args.value_freq,
            args.num_mini_batch,
            args.value_loss_coef,
            args.adv_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = max(1, int(args.num_env_steps) // args.num_steps // args.num_processes)

    nsteps = torch.zeros(args.num_processes)
    for j in range(num_updates):
        actor_critic.train()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.algo == "ppo":
                    value, action, action_log_prob = actor_critic.act(rollouts.obs[step])
                else:
                    adv, value, action, action_log_prob = actor_critic.act(rollouts.obs[step])

            obs, reward, done, infos = envs.step(action)

            if args.algo == "idaac":
                levels = torch.LongTensor([info["level_seed"] for info in infos])
                if j == 0 and step == 0:
                    rollouts.levels[0].copy_(levels)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            nsteps += 1
            nsteps[done == True] = 0
            if args.algo == "idaac":
                rollouts.insert(
                    obs,
                    action,
                    action_log_prob,
                    value,
                    reward,
                    masks,
                    adv,
                    levels,
                    nsteps,
                )
            elif args.algo == "daac":
                rollouts.insert(obs, action, action_log_prob, value, reward, masks, adv)
            else:
                rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        if args.algo == "idaac":
            rollouts.to(device)
            rollouts.before_update()
            (
                order_acc,
                order_loss,
                clf_loss,
                adv_loss,
                value_loss,
                action_loss,
                dist_entropy,
            ) = agent.update(rollouts, device)
        elif args.algo == "daac":
            adv_loss, value_loss, action_loss, dist_entropy = agent.update(rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Save Model
        # if j == num_updates - 1 and args.save_dir != "":
        #     try:
        #         os.makedirs(os.path.dirname(args.save_dir))
        #     except OSError:
        #         pass

        #     torch.save(
        #         {"agent": actor_critic, "ob_rms": getattr(envs, "ob_rms", None)},
        #         args.save_dir,
        #     )

        # Save Logs
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("\nUpdate {}, step {}:".format(j, total_num_steps))
            print(
                "Last {} training episodes, mean/median reward {:.2f}/{:.2f}".format(
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                )
            )

            # Log training stats
            logger.logkv("train/total_num_steps", total_num_steps)
            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            # Log eval stats (on the full distribution of levels)
            eval_episode_rewards = evaluate(args, actor_critic, device)
            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))

            logger.dumpkvs()

    if args.save_dir != "":
        try:
            os.makedirs(os.path.dirname(args.save_dir))
        except OSError:
            pass

        torch.save(
            {
                "agent": actor_critic,
                "order_classifier": order_classifier,
                "ob_rms": getattr(envs, "ob_rms", None),
            },
            args.save_dir,
        )
    eval_episode_rewards = evaluate(args, actor_critic, device)
    logging.info(f"Final eval reward was {eval_episode_rewards}")
    return -np.mean(eval_episode_rewards)


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
