#!/bin/bash
#SBATCH --job-name=mpogrpo-4node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=<partition>
#SBATCH --exclusive

set -euo pipefail

RM_URL=${RM_URL:-}
MRM_URL=${MRM_URL:-}
RM_SIZE=${RM_SIZE:-${1:-}}
MRM_SIZE=${MRM_SIZE:-${2:-}}

ROOT="/home/kim01756/Development/mpo"
ACC_CONFIG="$ROOT/examples/accelerate_configs/multi_node_2x8.yaml"
LAUNCH_MPGRPO="$ROOT/scripts/mpogrpo_experiments/launch_mpogrpo_iclr.sh"
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}

HOSTS=($(scontrol show hostnames "$SLURM_NODELIST"))
SGLANG_HOSTS=("${HOSTS[@]:0:2}")
TRAIN_HOSTS=("${HOSTS[@]:2:2}")

MASTER_ADDR="${TRAIN_HOSTS[0]}"
MASTER_PORT="${MASTER_PORT:-29500}"
SGLANG_PORT_BASE="${SGLANG_PORT_BASE:-30000}"
SGLANG_MODEL_DEFAULT="${SGLANG_MODEL_DEFAULT:-<fill-model-name>}"

select_qwen_model() {
  local size=$1
  case "$size" in
    1.5B|3B|7B|14B|32B|72B) echo "Qwen/Qwen2.5-${size}-Instruct-AWQ" ;;
    *) echo "" ;;
  esac
}

SGLANG_MODEL_RM="${SGLANG_MODEL_RM:-$(select_qwen_model "$RM_SIZE")}"
[[ -z "$SGLANG_MODEL_RM" ]] && SGLANG_MODEL_RM="$SGLANG_MODEL_DEFAULT"
SGLANG_MODEL_MRM="${SGLANG_MODEL_MRM:-$(select_qwen_model "$MRM_SIZE")}"
[[ -z "$SGLANG_MODEL_MRM" ]] && SGLANG_MODEL_MRM="$SGLANG_MODEL_DEFAULT"

SAME_MODEL=false
if [[ -n "$RM_SIZE" && -n "$MRM_SIZE" && "$RM_SIZE" == "$MRM_SIZE" ]]; then
  SAME_MODEL=true
  SGLANG_MODEL_MRM="$SGLANG_MODEL_RM"
  echo "RM == MRM"
fi

# Derive default RM/MRM URLs from the first GPU on each sglang node
if [[ -z "$RM_URL" ]]; then
  RM_URL="http://${SGLANG_HOSTS[0]}:${SGLANG_PORT_BASE}"
fi
if [[ "$SAME_MODEL" == true ]]; then
  MRM_URL="$RM_URL"
elif [[ -z "$MRM_URL" ]]; then
  local_mrm_port=$((SGLANG_PORT_BASE + 100))
  local_mrm_host="${SGLANG_HOSTS[1]:-${SGLANG_HOSTS[0]}}"
  MRM_URL="http://${local_mrm_host}:${local_mrm_port}"
fi

pids=()

start_sglang_node() {
  local host=$1
  local base_port=$2
  local model=$3
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task="$CPUS_PER_TASK" -w "$host" \
    bash -lc "cd $ROOT; \
      for gpu in {0..7}; do \
        CUDA_VISIBLE_DEVICES=\$gpu sglang launch_server --model \"$model\" --host 0.0.0.0 --port \$((base_port + gpu)) --tp 1 --trust-remote-code & \
      done; wait" &
  pids+=($!)
}

test_endpoint() {
  local label=$1
  local url=$2
  local retries=${SGLANG_TEST_RETRIES:-10}
  local delay=${SGLANG_TEST_DELAY:-15}
  local payload='{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"How many letters are there in the word SGLang?"}]}'
  for ((i=1; i<=retries; i++)); do
    if curl -fsS -H "Content-Type: application/json" -d "$payload" "${url%/}/v1/chat/completions"; then
      echo "[$label] success (${url})"
      return 0
    fi
    echo "[$label] attempt $i failed, retrying in ${delay}s..."
    sleep "$delay"
  done
  echo "[$label] failed after ${retries} attempts (${url})"
  return 1
}

start_sglang_node "${SGLANG_HOSTS[0]}" "$SGLANG_PORT_BASE" "$SGLANG_MODEL_RM"
if [[ "$SAME_MODEL" == false && ${#SGLANG_HOSTS[@]} -gt 1 ]]; then
  start_sglang_node "${SGLANG_HOSTS[1]}" $((SGLANG_PORT_BASE + 100)) "$SGLANG_MODEL_MRM"
fi

# Optional connectivity test for the launched servers
test_endpoint "RM" "$RM_URL"
if [[ "$SAME_MODEL" == false ]]; then
  test_endpoint "MRM" "$MRM_URL"
fi

# srun --nodes=${#TRAIN_HOSTS[@]} \
#      --ntasks=${#TRAIN_HOSTS[@]} \
#      --ntasks-per-node=1 \
#      --gres=gpu:8 \
#      --cpus-per-task="$CPUS_PER_TASK" \
#      -w "$(IFS=','; echo "${TRAIN_HOSTS[*]}")" \
#      bash -lc "cd $ROOT; \
#        MASTER_ADDR=\"$MASTER_ADDR\" MASTER_PORT=\"$MASTER_PORT\" \
#        ACC_CONFIG_OVERRIDE=\"$ACC_CONFIG\" \
#        CUDA_DEVICES_OVERRIDE=\"0,1,2,3,4,5,6,7\" \
#        MACHINE_RANK_OVERRIDE=\"\$SLURM_PROCID\" \
#        NUM_MACHINES_OVERRIDE=\"${#TRAIN_HOSTS[@]}\" \
#        NUM_PROCESSES_OVERRIDE=16 \
#          \"$LAUNCH_MPGRPO\" \"$RM_URL\" \"$MRM_URL\""

# for pid in "${pids[@]}"; do
#   kill "$pid" 2>/dev/null || true
# done
