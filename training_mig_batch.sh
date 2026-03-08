#!/bin/bash
#SBATCH --job-name=run-training-loop
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681

module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source venv/bin/activate

pip install --upgrade pip

# CRLF->LF conversion
sed -i 's/\r//' reqs.txt

# torch (needs special index)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# lerobot (not on PyPI)
pip install git+https://github.com/huggingface/lerobot.git

# remaining reqs (skip isaaclab/isaacsim)
grep -vE "isaaclab|isaacsim" reqs.txt > reqs_filtered.txt
pip install -r reqs_filtered.txt --extra-index-url https://download.pytorch.org/whl/cu121

python vbti/utils/train/train_smolvla_custom.py
deactivate