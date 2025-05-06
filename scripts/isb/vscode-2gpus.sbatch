#!/bin/bash
#SBATCH --job-name=code_tunnel-2gpus
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm-logs/code_tunnel_%j.out

# Start named VS Code tunnel for remote connection to compute node
~/opt/vscode_cli/code tunnel --name "i-ai_compute-2gpus"

