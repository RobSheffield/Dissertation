#!/bin/bash
#SBATCH --job-name=K-flips_test
#SBATCH --output=training_Kflips2_%j.log
#SBATCH --error=training_Kflips2_%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
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

python helpers/k_fold_leakage_test.py
echo "Training completed!"