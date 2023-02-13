#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./dehb.err
#SBATCH --job-name=dehb_procgen
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./dehb.out
#SBATCH --partition=learnlab
#SBATCH --time=2000

python run_dehb.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$1 clf_hidden_size=$2 hydra.sweeper.total_cost=$3 hydra.sweep.dir=tuning_output_dehb_$3/idaac_$1_seed_$SLURM_ARRAY_TASK_ID