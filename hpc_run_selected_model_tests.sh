#!/bin/bash
#SBATCH --job-name=selected_model_test
#SBATCH --output=selected_model_test_%j.log
#SBATCH --error=selected_model_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
module load OpenBLAS/0.3.23-GCC-12.3.0

cd /users/acb22re/Dissertation
source .venv/bin/activate

export K_FOLD_DEVICE=${K_FOLD_DEVICE:-0}

python testing/run_selected_model_tests.py \
  testing/First_60_baseline_model_run_1 \
  testing/First_60_baseline_model_run_2 \
  testing/First_60_baseline_model_run_3 \
  testing/First_60_baseline_model_run_4 \
  testing/First_60_guide_model_run_1 \
  testing/First_60_guide_model_run_2 \
  testing/First_60_guide_model_run_3 \
  testing/First_60_guide_model_run_4 \
  testing/First_60_rand_model_run_1 \
  testing/First_60_rand_model_run_2 \
  testing/First_60_rand_model_run_3 \
  testing/First_60_rand_model_run_4 \
  testing/First_60_guide2_model_run_1 \
  testing/First_60_guide2_model_run_2 \
  testing/First_60_guide2_model_run_3 \
  testing/First_60_guide2_model_run_4

echo "Test-only evaluation completed."