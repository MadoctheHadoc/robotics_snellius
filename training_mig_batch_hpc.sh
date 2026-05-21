#!/bin/bash
#SBATCH --job-name=run-training-loop
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=tue.gpu.q

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source venv/bin/activate

pip install --upgrade pip

pip install torch==2.5.1+cu121 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r reqs_training.txt

python vbti/utils/teleoperation/train_smolvla.py
deactivate