#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# Usage:
#   sbatch --job-name=<name> run.slurm <python_module> [args...]
# Example:
#   sbatch --job-name=train-baseline run.slurm src.train_baseline --task B --output_dir outputs/baseline_test --epochs 1 --batch_size 64 --learning_rate 2e-5 --max_length 512

if [ $# -lt 1 ]; then
  echo "Usage: sbatch --job-name=<name> run.slurm <python_module> [args...]"
  exit 1
fi

mkdir -p logs

MODULE="$1"; shift

# Activate your environment
source /apps/local/anaconda3/conda_init.sh
conda activate llm

echo "Running: python -m ${MODULE} $*"
python -m "${MODULE}" "$@"
