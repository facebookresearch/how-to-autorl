#!/bin/bash

#SBATCH --array=0-4
#SBATCH --error=./rs_spaces.err
#SBATCH --job-name=rs_spaces
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./rs_spaces.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python tune_rs.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$2 algorithm=$1