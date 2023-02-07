#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./dehb.err
#SBATCH --job-name=dehb_brax
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./dehb.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python run_dehb.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$1 num_timesteps=$2 hydra.sweeper.dehb_kwargs.min_budget=$3 hydra.sweeper.total_cost=$4 hydra.sweep.dir=tuning_output_dehb_$4/$1_seed_$SLURM_ARRAY_TASK_ID hydra.run.dir=tuning_output_dehb_$4/$1_seed_$SLURM_ARRAY_TASK_ID