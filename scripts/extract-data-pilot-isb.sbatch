#!/bin/bash

#SBATCH --job-name=extract-data-pilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem=14G

cd "${SLURM_SUBMIT_DIR}"
echo $(pwd)
echo $(hostname)

out_log="${SLURM_SUBMIT_DIR}"/slurm-script-"${SLURM_JOB_ID}".out
scontrol write batch_script "${SLURM_JOB_ID}" ${out_log}

# TODO: update and remove "-test"
micromamba run -n data-extraction-test python scripts/extract-data-pilot.py
