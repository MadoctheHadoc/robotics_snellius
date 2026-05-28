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

# Uses different HF version from the training for some reason
pip install -r reqs_inference.txt

echo "============================================"
echo "Node: $(hostname)  SLURM: $SLURM_NODELIST"
echo "============================================"

# Open a reverse tunnel: login-node:5556 -> this compute node:5556.
# This avoids firewall rules that block direct TCP to compute nodes.
# Requires passwordless SSH from compute node to login node; if this
# hangs, set up an SSH key: ssh-keygen && ssh-copy-id 20221051@hpc.tue.nl
ssh -N -f -o StrictHostKeyChecking=no \
    -R 5556:localhost:5556 \
    20221051@hpc.tue.nl
echo "Reverse tunnel established: hpc.tue.nl:5556 -> $(hostname):5556"
echo "Laptop tunnel: ssh -N -L 5556:localhost:5556 20221051@hpc.tue.nl"

python -u vbti/utils/teleoperation/infer_smolvla.py \
    --serve \
    --zmq_port 5556 \
    --model_dir outputs/smolvla_so101

deactivate
