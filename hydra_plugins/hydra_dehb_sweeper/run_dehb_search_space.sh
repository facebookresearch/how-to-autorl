#!/bin/bash

#SBATCH --array=0-4
#SBATCH --error=/private/home/theeimer/autorl-hydra-sweepers/sb3/hydra_dehb.err
#SBATCH --job-name=run_hydra_dehb
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/private/home/theeimer/autorl-hydra-sweepers/sb3/hydra_dehb.out
#SBATCH --partition=learnfair
#SBATCH --time=2000

python run_hydra_dehb.py --multirun seed=$SLURM_ARRAY_TASK_ID search_space=$1