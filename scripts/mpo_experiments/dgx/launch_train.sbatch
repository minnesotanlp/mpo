#!/bin/bash
#SBATCH -J mpoppo-train
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH -t 72:00:00
#SBATCH -p dgx
#SBATCH -o %x.%j.out

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <rm_address> <mrm_address>" >&2
  exit 1
fi

rm_address=$1
mrm_address=$2

###############################################################################
# Environment
###############################################################################

# Point to this repo (overrides the default in mpoppo_train.sh)
export MPO_ROOT="/lustre/fs0/scratch/zkim/Development/mpo-old"

# Load secrets (e.g., HF_TOKEN) if present
if [ -f "${MPO_ROOT}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${MPO_ROOT}/.env"
  set +a
fi

: "${HF_TOKEN:?HF_TOKEN must be set (place it in ${MPO_ROOT}/.env)}"

cd "${MPO_ROOT}"

# Force single-node, 8 GPU layout
export CUDA_DEVICES_OVERRIDE="${CUDA_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
export ACC_CONFIG_OVERRIDE="${ACC_CONFIG_OVERRIDE:-${MPO_ROOT}/examples/accelerate_configs/deepspeed_zero2.yaml}"

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node list:    ${SLURM_NODELIST}"
echo "CUDA devices: ${CUDA_DEVICES_OVERRIDE}"
echo "Accel config: ${ACC_CONFIG_OVERRIDE}"
echo

###############################################################################
# Launch training
###############################################################################

srun --ntasks=1 \
     bash scripts/mpo_experiments/dgx/mpoppo_train.sh "${rm_address}" "${mrm_address}"
