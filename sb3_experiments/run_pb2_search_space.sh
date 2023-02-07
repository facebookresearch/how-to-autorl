#!/bin/bash

#SBATCH --array=0-4
#SBATCH --error=./pb2_spaces.err
#SBATCH --job-name=pb2_spaces
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./pb2_spaces.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python tune_pb2.py --multirun seed=$SLURM_ARRAY_TASK_ID search_space=$3 env_name=$2 algorithm=$1 wandb=false +hydra.sweeper.pbt_kwargs.wandb_tags=['search_space_ablation_pb2']