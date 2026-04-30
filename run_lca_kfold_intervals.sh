#!/bin/bash
#SBATCH --job-name=LCA_kfold
#SBATCH --output=LCA_kfold_%j.log
#SBATCH --error=LCA_kfold_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load Python/3.11.3-GCCcore-12.3.0
module load OpenBLAS/0.3.23-GCC-12.3.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

# Defaults are folder-based and independence-preserving.
export LCA_DATA_ROOT=${LCA_DATA_ROOT:-Castings}
export LCA_OUT=${LCA_OUT:-testing/lca_kfold_intervals_report.csv}
export LCA_K=${LCA_K:-5}

python testing/lca_kfold_intervals.py \
  --data-root "${LCA_DATA_ROOT}" \
  --out "${LCA_OUT}" \
  --k "${LCA_K}"

echo "LCA K-fold interval run completed!"