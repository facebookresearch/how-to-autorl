#!/bin/bash

#SBATCH --array=0-4
#SBATCH --error=./dehb_seeds.err
#SBATCH --job-name=dehb_seeds
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./dehb_seeds.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python tune_dehb.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$2 algorithm=$1 wandb=false hydra.sweep.dir=tuning_output/dehb_seed_$SLURM_ARRAY_TASK_ID_$1_$2_seeds_$3 hydra.sweep.dir=tuning_output/dehb_seed_$SLURM_ARRAY_TASK_ID_$1_$2_seeds_$3 hydra.sweeper.dehb_kwargs=$4