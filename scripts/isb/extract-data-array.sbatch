#!/bin/bash

#SBATCH --job-name=extract-data-array
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem=14G
#SBATCH --array=0-69%8

echo $(date)

cd "${SLURM_SUBMIT_DIR}"
echo $(pwd)
echo $(hostname)

scontrol write batch_script "${SLURM_JOB_ID}" "${SLURM_SUBMIT_DIR}"/slurm-script-"${SLURM_JOB_ID}".out

module load cuda/12.2
module load cudatoolkit/23.9_12.2
micromamba run -n data-extraction python scripts/extract-data-array.py

echo $(date)
