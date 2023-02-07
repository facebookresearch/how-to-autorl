#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./pb2.err
#SBATCH --job-name=pb2_procgen
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./pb2.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python run_pb2.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$1 clf_hidden_size=$2