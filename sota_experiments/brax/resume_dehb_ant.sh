#!/bin/bash

#SBATCH --array=2
#SBATCH --error=./dehb_ant_s%a.err
#SBATCH --job-name=dehb_ant
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./dehb_ant_s%a.out
#SBATCH --partition=learnlab
#SBATCH --time=2000

python run_dehb.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=ant num_timesteps=30000000 hydra.sweeper.dehb_kwargs.min_budget=300000 hydra.sweeper.total_cost=1920000000 hydra.launcher.partition=learnlab +hydra.sweeper.resume=tuning_output_dehb/ant_seed_$SLURM_ARRAY_TASK_ID/dehb_state.pkl
