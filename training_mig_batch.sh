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

pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install transformers
pip install lerobot==0.4.4
pip install -e .
pip install dotenv
pip install huggingface_hub==0.35.3

python vbti/utils/teleoperation/train_smolvla.py
deactivate