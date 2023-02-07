## Stable Baseline Sweeps & Experiments

This directory contains the scripts and configs to reproduce our experiments on SB3. Before launching any slurm scripts, please add your own partition and review the settings.


The full commands for the hyperparameter sweeps are in 'sweep_commands.txt'.

PB2 and DEHB experiments can be launched via slurm using the corresponding scripts. For a small PPO search space for DEHB on Acrobot, for example, run:
```bash
sbatch run_dehb_search_space.sh ppo Acrobot-v1 ppo_small
```
The seed experiments can be launched the same way, e.g. 3 seeds of DQN on MiniGrid with PB2:
```bash
sbatch run_pb2_seeds.sh dqn MiniGrid-Empty-5x5-v0 3 [0,1,2]
```

The Optuna RS experiments require manually editing the yaml config by copying the hyperparameter search space (or a subset) from the optuna search space files in 'configs/search_space' to rs.yaml.
The number of tuning seeds can be set there as well. The bash script for launch is 'run_rsh.sh'.