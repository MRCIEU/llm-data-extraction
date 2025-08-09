#!/bin/bash

#SBATCH --job-name=extract-data-pilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem=14G
#SBATCH --output=output/slurm-logs/slurm-%A_%a.out
#SBATCH --array=0

echo $(date "+%Y-%m-%dT%H-%M")

cd "${SLURM_SUBMIT_DIR}"
echo $(pwd)
echo $(hostname)
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

prefix="isb-ai"
output_dir="${SLURM_SUBMIT_DIR}"/output/"${prefix}"-"${SLURM_ARRAY_JOB_ID}"
mkdir -p "${output_dir}"/{results,logs}
output_results_dir="${output_dir}"/results

output_log_file="${output_dir}"/logs/slurm-"${SLURM_ARRAY_JOB_ID}"_"${SLURM_ARRAY_TASK_ID}".out
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    submit_script_log_file="${output_dir}"/logs/script-"${SLURM_ARRAY_JOB_ID}".out
    scontrol write batch_script "${SLURM_ARRAY_JOB_ID}" ${submit_script_log_file}
fi

module load cuda/12.2
module load cudatoolkit/23.9_12.2
micromamba run -n data-extraction \
    python scripts/python/batch/extract-data.py \
    --output_dir "${output_results_dir}" \
    --array-id "${SLURM_ARRAY_TASK_ID}" \
    --model llama-3 \
    --pilot \
    > "${output_log_file}" 2>&1 

echo $(date "+%Y-%m-%dT%H-%M")
