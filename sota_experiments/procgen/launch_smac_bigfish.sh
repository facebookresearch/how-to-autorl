#!/bin/bash

#SBATCH --error=./smac.err
#SBATCH --job-name=smac_procgen
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./smac.out
#SBATCH --partition=learnlab
#SBATCH --time=2000

python run_smac_bigfish.py