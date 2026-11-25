#!/bin/bash
#SBATCH --job-name=ceval-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
python scripts/run_eval.py --config "${CONFIG}" --run-dir "${RUN_DIR}"
python scripts/merge_shards.py --run-dir "${RUN_DIR}"
