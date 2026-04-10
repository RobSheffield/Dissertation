#!/bin/bash
#SBATCH --job-name=K-flips_test
#SBATCH --output=training_bias2_%j.log
#SBATCH --error=training_bias2_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
module load OpenBLAS/0.3.23-GCC-12.3.0


cd /users/acb22re/Dissertation
#python -m venv .venv
source .venv/bin/activate

#pip install --upgrade pip
#pip install -r hpc_requirements.txt


#source /users/acb22re/CopiedDissertation/take2/X-Ray_Image_Analysis/.venv/bin/activate

# Ultralytics accepts device strings like "0" or "0,1".
# Override at submit time if needed, e.g.:
#   sbatch --export=ALL,K_FOLD_DEVICE=0,1 hpc_train.sh
export K_FOLD_DEVICE=${K_FOLD_DEVICE:-0,1,2,3}

python helpers/k_fold_leakage_test.py
echo "Training completed!"