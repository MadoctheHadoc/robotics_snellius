#!/bin/bash
#SBATCH --job-name=smolvla-serve
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681

module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source venv/bin/activate

pip install pyzmq opencv-python-headless --quiet

echo "============================================"
echo "Node hostname : $(hostname)"
echo "SSH tunnel    : ssh -L 5556:$(hostname):5556 lpopdimitrova@snellius.surf.nl"
echo "============================================"

python vbti/utils/teleoperation/infer_smolvla.py \
    --serve \
    --zmq_port 5556 \
    --model_dir outputs/smolvla_so101

deactivate
