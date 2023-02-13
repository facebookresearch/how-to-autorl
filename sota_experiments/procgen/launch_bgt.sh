#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./bgt.err
#SBATCH --job-name=bgt_procgen
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./bgt.out
#SBATCH --partition=learnlab
#SBATCH --time=2000

python run_bgt.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$1 clf_hidden_size=$2 hydra.launcher.partition=learnfair