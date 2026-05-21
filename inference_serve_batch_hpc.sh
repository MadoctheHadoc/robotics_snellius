#!/bin/bash
#SBATCH --job-name=smolvla-serve
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=tue.gpu.q

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# This might have to be done the first time we run it
# python -m venv venv

source venv/bin/activate

pip install pyzmq opencv-python-headless
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install lerobot
pip install "huggingface_hub==1.6.0"
pip install -e .

echo "============================================"
echo "Node hostname : $(hostname)"
echo "SSH tunnel    : ssh -L 5556:$(hostname):5556 somethingelse@IDKyet"
echo "============================================"

python vbti/utils/teleoperation/infer_smolvla.py \
    --serve \
    --zmq_port 5556 \
    --model_dir outputs/smolvla_so101

deactivate
