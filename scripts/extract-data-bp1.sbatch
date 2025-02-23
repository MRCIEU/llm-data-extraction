#!/bin/bash

#SBATCH --job-name=MR-extract
#SBATCH --partition=mrcieu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --gpus=1
#SBATCH --mem=14G
#SBATCH --account=SSCM013902

micromamba run -n data-extraction python scripts/extract-data.py
