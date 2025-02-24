#!/bin/bash

#SBATCH --job-name=MR-extract
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem=14G

micromamba run -n data-extraction python scripts/extract-data.py
