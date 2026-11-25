#!/bin/bash
#SBATCH --job-name=ceval-multigpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Adapt these two lines to the cluster:
# module load cuda/12.1
# conda activate ceval

CONFIG="${CONFIG:-configs/experiments/baseline.yaml}"
RUN_DIR="${RUN_DIR:-outputs/exp00_baseline}"
NUM_SHARDS="${NUM_SHARDS:-4}"
bash scripts/run_parallel.sh "${CONFIG}" "${NUM_SHARDS}" "${RUN_DIR}"
