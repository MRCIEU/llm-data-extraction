#!/bin/bash

#SBATCH --job-name=extract-data-pilot
#SBATCH --partition=mrcieu-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --gpus=1
#SBATCH --mem=14G
#SBATCH --account=SSCM013902

cd "${SLURM_SUBMIT_DIR}"
echo $(pwd)
echo $(hostname)

scontrol write batch_script "${SLURM_JOB_ID}" "${SLURM_SUBMIT_DIR}"/slurm-script-"${SLURM_JOB_ID}".out

micromamba run -n data-extraction python scripts/extract-data-pilot.py
