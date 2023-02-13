#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./rs.err
#SBATCH --job-name=rs_procgen
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./rs.out
#SBATCH --partition=learnlab
#SBATCH --time=2000

python rs_run_idaac.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$1 clf_hidden_size=$2