#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./dehb_halfcheetah_s%a.err
#SBATCH --job-name=dehb_halfcheetah
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./dehb_halfcheetah_s%a.out
#SBATCH --partition=learnlab
#SBATCH --time=2000

python run_dehb.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=halfcheetah num_timesteps=100000000 hydra.sweeper.dehb_kwargs.min_budget=1000000 hydra.sweeper.total_cost=6400000000 hydra.launcher.partition=learnlab +hydra.sweeper.resume=tuning_output_dehb/halfcheetah_seed_$SLURM_ARRAY_TASK_ID/dehb_state.pkl
