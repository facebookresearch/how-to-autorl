#!/bin/bash

#SBATCH --array=0-4
#SBATCH --error=./pb2_seeds.err
#SBATCH --job-name=pb2_seeds
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./pb2_seeds.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python tune_pb2.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=$2 algorithm=$1 wandb=false hydra.sweep.dir=tuning_output/dehb_seed_$SLURM_ARRAY_TASK_ID_$1_$2_seeds_$3 hydra.sweep.dir=tuning_output/dehb_seed_$SLURM_ARRAY_TASK_ID_$1_$2_seeds_$3 hydra.sweeper.pbt_kwargs=$4