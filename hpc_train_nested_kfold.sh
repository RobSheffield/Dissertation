#!/bin/bash
#SBATCH --job-name=nested_kfold_train
#SBATCH --output=nested_kfold_train_%j.log
#SBATCH --error=nested_kfold_train_%j.err
#SBATCH --time=24:00:00
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
#   sbatch --export=ALL,K_FOLD_DEVICE=0,1,2,3 hpc_train_nested_kfold.sh
export K_FOLD_DEVICE=${K_FOLD_DEVICE:-0,1,2,3}

# Nested K-fold parameters
# Override at submit time, e.g.:
#   sbatch --export=ALL,K_OUTER=5,K_INNER=4 hpc_train_nested_kfold.sh
export K_OUTER=${K_OUTER:-5}
export K_INNER=${K_INNER:-4}
export TEST_SIZE=${TEST_SIZE:-0.2}
export EPOCHS=${EPOCHS:-150}
export APPLY_AUGMENTATIONS=${APPLY_AUGMENTATIONS:-false}

python helpers/train_nested_kfold.py
echo "Nested K-Fold training and evaluation completed!"
