#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --time=24:00:00

echo $(date "+%Y-%m-%dT%H-%M")

cd "${SLURM_SUBMIT_DIR}"
echo $(pwd)
echo $(hostname)

out_log="${SLURM_SUBMIT_DIR}"/slurm-script-"${SLURM_JOB_ID}".out
scontrol write batch_script "${SLURM_JOB_ID}" ${out_log}

eval "$(micromamba shell hook --shell bash)"
micromamba activate data-extraction

HOSTNAME=$(hostname)
LISTEN_IP=$(dig ${HOSTNAME} +short A)
LISTEN_PORT=8086

set -o xtrace
jupyter lab --no-browser --ip=${LISTEN_IP} --port=${LISTEN_PORT}

echo $(date "+%Y-%m-%dT%H-%M")
