#!/bin/bash

#SBATCH --job-name=MR-extract-array
#SBATCH --partition=mrcieu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=14G
#SBATCH --account=SSCM013902
#SBATCH --output=/user/work/eptrg/projects/MRlit-Llama3/MRlit-Llama3-%j.out
#SBATCH --error=/user/work/eptrg/projects/MRlit-Llama3/MRlit-Llama3-%j.err
#SBATCH --array=0-69%8

module load lang/python/anaconda/3.11-2024.02-1
module load lang/cuda/12.0.0-gcc-9.1.0
conda run -n llama --live-stream python extract_data.py
