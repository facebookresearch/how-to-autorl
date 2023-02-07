import argparse
import torch

parser = argparse.ArgumentParser(description="RL")

# PPO arguments.
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon")
parser.add_argument("--alpha", type=float, default=0.99, help="RMSprop optimizer apha")
parser.add_argument("--gamma", type=float, default=0.999, help="discount factor for rewards")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy term coefficient")
parser.add_argument("--value_loss_coef", type=float, default=0.5, help="value loss coefficient (default: 0.5)")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max norm of gradients)")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--num_processes", type=int, default=64, help="how many training CPU processes to use")
parser.add_argument("--num_steps", type=int, default=256, help="number of forward steps in A2C")
parser.add_argument("--ppo_epoch", type=int, default=1, help="number of ppo epochs")
parser.add_argument("--num_mini_batch", type=int, default=8, help="number of batches for ppo")
parser.add_argument("--clip_param", type=float, default=0.2, help="ppo clip parameter")
parser.add_argument("--log_interval", type=int, default=10, help="log interval, one log per n updates")
parser.add_argument("--num_env_steps", type=int, default=25e6, help="number of environment steps to train")
parser.add_argument("--env_name", type=str, default="coinrun", help="environment to train on")
parser.add_argument("--algo", default="idaac", choices=["idaac", "daac", "ppo"], help="algorithm to use")
parser.add_argument("--log_dir", default="logs", help="directory to save agent logs")
parser.add_argument("--save_dir", type=str, default="models", help="augmentation type")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--hidden_size", type=int, default=256, help="state embedding dimension")

# DAAC arguments.
parser.add_argument(
    "--use_best_hps",
    action="store_true",
    default=False,
    help="use the best hyperparameters for each game. \
    if False, use the same hyperparameters for all games (i.e. the default ones)",
)
parser.add_argument("--value_epoch", type=int, default=9, help="number of ppo epochs")
parser.add_argument("--value_freq", type=int, default=1, help="number of value epochs")
parser.add_argument("--adv_loss_coef", type=float, default=0.25, help="coefficient for the GAE loss")

# IDAAC arguments.
parser.add_argument("--use_nonlinear_clf", action="store_true", default=False, help="use level invariance")
parser.add_argument("--order_loss_coef", type=float, default=0.001, help="coefficient for the GAE loss")
parser.add_argument("--clf_hidden_size", type=int, default=4, help="coefficient for the GAE loss")

# Procgen arguments.
parser.add_argument("--distribution_mode", default="easy", help="distribution of envs for procgen")
parser.add_argument("--num_levels", type=int, default=200, help="number of Procgen levels to use for training")
parser.add_argument("--start_level", type=int, default=0, help="start level id for sampling Procgen levels")
