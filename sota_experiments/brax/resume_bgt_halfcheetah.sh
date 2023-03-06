#!/bin/bash

#SBATCH --array=2
#SBATCH --error=./bgt_halfcheetah_s%a.err
#SBATCH --job-name=bgt_halfcheetah
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./bgt_halfcheetah_s%a.out
#SBATCH --partition=learnfair
#SBATCH --time=2000

python run_bgt.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=halfcheetah num_timesteps=100000000 hydra.launcher.partition=learnfair +hydra.sweeper.resume=tuning_output_bgt_64/halfcheetah_seed_$SLURM_ARRAY_TASK_ID/pbt_state.pkl