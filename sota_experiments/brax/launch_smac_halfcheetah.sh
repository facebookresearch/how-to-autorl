#!/bin/bash

#SBATCH --error=./smac.err
#SBATCH --job-name=smac_halfcheetah
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=./smac.out
#SBATCH --partition=<partition>
#SBATCH --time=2000

python run_smac_halfcheetah.py