#!/bin/bash

#SBATCH --array=0-2
#SBATCH --error=./pb2_climber_s%a.err
#SBATCH --job-name=pb2_climber
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./pb2_climber_s%a.out
#SBATCH --partition=learnfair
#SBATCH --time=2000

python run_pb2.py --multirun seed=$SLURM_ARRAY_TASK_ID env_name=climber clf_hidden_size=64 hydra.launcher.partition=learnfair +hydra.sweeper.resume=tuning_output_pb2/idaac_climber_seed_$SLURM_ARRAY_TASK_ID/pbt_state.pkl