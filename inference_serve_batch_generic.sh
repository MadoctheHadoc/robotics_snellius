#!/bin/bash
# Generic SLURM inference server script.
# Customise the SBATCH options and environment section below for your cluster.
#
# Submit:  sbatch inference_serve_batch_generic.sh
# Tunnel:  ssh -L 5556:<NODENAME>:5556 <user>@<login-node>
# Client:  python vbti/utils/teleoperation/infer_smolvla_client.py

# ── SBATCH options ────────────────────────────────────────────────────────────
# These are standard SLURM and portable across most clusters.
# Only --partition needs to change per site (common names: gpu, gpu_a100, gpu_v100, gpu_p100).
# Remove --reservation entirely if your cluster does not use reservations.
SBATCH --job-name=smolvla-serve
SBATCH --output=%x_%j.out
SBATCH --nodes=1
SBATCH --gpus=1
SBATCH --cpus-per-task=4
SBATCH --mem=16G
SBATCH --time=02:00:00
SBATCH --partition=gpu          # <-- change to your cluster's GPU partition name

# ── Environment ───────────────────────────────────────────────────────────────
# Option A: module-based clusters (common on EU HPC sites — edit module names).
# Comment out if your cluster uses conda or a pre-built container instead.
# module load Python/3.11
# module load CUDA/12.1

# Option B: conda — uncomment and adjust if you use a conda environment.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate smolvla

# Option C: plain virtualenv (used on Snellius with modules already loaded above).
source venv/bin/activate

# ── Tunnel info ───────────────────────────────────────────────────────────────
# Printed to the job log as soon as the node is allocated.
# Replace <login-node> with your cluster's login hostname.
LOGIN_NODE="<login-node>"           # e.g. login.hpc.example.ac.uk
ZMQ_PORT=5556
NODE=$(hostname)

echo "============================================"
echo "Node hostname : ${NODE}"
echo "SSH tunnel    : ssh -L ${ZMQ_PORT}:${NODE}:${ZMQ_PORT} <user>@${LOGIN_NODE}"
echo "============================================"

# ── Run server ────────────────────────────────────────────────────────────────
python vbti/utils/teleoperation/infer_smolvla.py \
    --serve \
    --zmq_port ${ZMQ_PORT} \
    --model_dir outputs/smolvla_so101

deactivate
