#!/bin/bash
#SBATCH -J sglang
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu[005-006]
#SBATCH --mem=100GB
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH -t 96:00:00
#SBATCH -o %x.%j.out

set -euo pipefail

##############################
# Paths & environment
##############################

# Your project root
export MPOPPO_ROOT="/lustre/fs0/scratch/zkim/Development/mpo-old"

# Load .env (expects a line like: HF_TOKEN=xxxx)
if [ -f "${MPOPPO_ROOT}/.env" ]; then
  # Export variables defined inside .env
  set -a
  # shellcheck source=/dev/null
  source "${MPOPPO_ROOT}/.env"
  set +a
else
  echo "[WARN] ${MPOPPO_ROOT}/.env not found; HF_TOKEN may be unset" >&2
fi

: "${HF_TOKEN:?HF_TOKEN must be set in ${MPOPPO_ROOT}/.env}"

# Container image (your SQSH)
export SGLANG_IMAGE="/lustre/fs0/scratch/zkim/sqsh-files/lmsysorg+sglang+latest.sqsh"
export IMAGE="${SGLANG_IMAGE}"

# HF cache on Lustre
export HOST_HF_CACHE="/lustre/fs0/scratch/zkim/hf_cache"

# In-container HF cache path
export HF_HOME="${HOST_HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

# Port for each sglang server
export SGLANG_PORT=30000

# Two logical model roles (currently same HF repo, but separated by name)
export RM_MODEL="qwen/Qwen2.5-32B-Instruct-AWQ"
export MRM_MODEL="qwen/Qwen2.5-32B-Instruct-AWQ"

##############################
# Node layout
##############################

# Get node hostnames as an array
mapfile -t HOSTS < <(scontrol show hostname "$SLURM_NODELIST")
RM_NODE="${HOSTS[0]}"
MRM_NODE="${HOSTS[1]}"

echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "Tasks:          $SLURM_NTASKS"
echo "Node list:      $SLURM_NODELIST"
echo "Image:          $SGLANG_IMAGE"
echo "Host HF cache:  $HOST_HF_CACHE"
echo "HF_HOME (ct):   $HF_HOME"
echo "RM node:        $RM_NODE"
echo "MRM node:       $MRM_NODE"
echo "RM model:       $RM_MODEL"
echo "MRM model:      $MRM_MODEL"
echo "Port:           $SGLANG_PORT"
echo

##############################
# Launch sglang on RM node
##############################

echo "[INFO] Launching RM server on ${RM_NODE}..."
srun --nodes=1 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --nodelist="${RM_NODE}" \
     --gpus-per-task=8 \
     --exclusive \
     --container-image="${IMAGE}" \
     --container-remap-root \
     --container-writable \
     --container-name=sglang-rm \
     --container-workdir=/root \
     --container-mounts="${HOST_HF_CACHE}:${HF_HOME}" \
     --container-env=HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN \
     python3 -m sglang.launch_server \
       --model-path "${RM_MODEL}" \
       --host 0.0.0.0 \
       --dp 8 \
       --port "${SGLANG_PORT}" \
       --grammar-backend xgrammar &
RM_PID=$!

##############################
# Launch sglang on MRM node
##############################

echo "[INFO] Launching MRM server on ${MRM_NODE}..."
srun --nodes=1 \
     --ntasks=1 \
     --ntasks-per-node=1 \
     --nodelist="${MRM_NODE}" \
     --gpus-per-task=8 \
     --exclusive \
     --container-image="${IMAGE}" \
     --container-remap-root \
     --container-writable \
     --container-name=sglang-mrm \
     --container-workdir=/root \
     --container-mounts="${HOST_HF_CACHE}:${HF_HOME}" \
     --container-env=HF_HOME,TRANSFORMERS_CACHE,HF_DATASETS_CACHE,HF_TOKEN \
     python3 -m sglang.launch_server \
       --model-path "${MRM_MODEL}" \
       --host 0.0.0.0 \
       --dp 8 \
       --port "${SGLANG_PORT}" \
       --grammar-backend xgrammar &
MRM_PID=$!

##############################
# Print endpoints & wait
##############################

echo
echo "[INFO] RM endpoint:  http://${RM_NODE}:${SGLANG_PORT}/v1/chat/completions"
echo "[INFO] MRM endpoint: http://${MRM_NODE}:${SGLANG_PORT}/v1/chat/completions"
echo
echo "[INFO] Use SSH port-forwarding if needed, e.g. from your laptop:"
echo "  ssh -L 30000:${RM_NODE}:${SGLANG_PORT} zkim@slogin001"
echo "  curl http://localhost:30000/v1/chat/completions ..."
echo

# Keep job alive as long as servers run
wait "${RM_PID}" "${MRM_PID}"
