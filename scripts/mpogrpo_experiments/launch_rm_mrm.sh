#!/usr/bin/env bash
# ------------------------------------------------------------------
# launch_rm_mrm.sh PEM_KEY USER@HOST RM_SUFFIX MRM_SUFFIX [RM_PORT] [MRM_PORT]
# ------------------------------------------------------------------
set -euo pipefail

if (( $# < 4 )); then
  echo "Usage: $0 <pem_key> <user@host> <rm_suffix> <mrm_suffix> [rm_port] [mrm_port]"
  exit 1
fi

PEM_KEY=$1
REMOTE_HOST=$2
RM_SUFFIX=$3
MRM_SUFFIX=$4
RM_PORT=${5:-32000}
MRM_PORT=${6:-34000}

ssh -i "$PEM_KEY" -p 24411 "$REMOTE_HOST" bash -s -- \
     "$RM_SUFFIX" "$MRM_SUFFIX" "$RM_PORT" "$MRM_PORT" <<'REMOTE'
set -euo pipefail

RM_SUFFIX=$1
MRM_SUFFIX=$2
RM_PORT=$3
MRM_PORT=$4

SGLANG_VENV="$HOME/Development/sglang/sglang-venv"
LOG_DIR="$HOME/Development/sglang/logs"
mkdir -p "$LOG_DIR"

kill_gpu_procs () {
  sudo pkill -f sglang 2>/dev/null || true
}

echo "[INFO] Cleaning GPUs"; kill_gpu_procs; sleep 10

start_server () {
  local gpus=$1 dp=$2 suffix=$3 port=$4 log=$5

  echo "[INFO] Starting Qwen/Qwen2.5-${suffix^^}-Instruct-AWQ "\
       "on GPUs $gpus (port $port)"
  tmux new-session -d -s "sg_${suffix}_${port}" bash -c '
    set -euo pipefail
    source "'"$SGLANG_VENV"'/bin/activate"
    CUDA_VISIBLE_DEVICES='"$gpus"' \
    exec python -m sglang.launch_server \
      --model-path "Qwen/Qwen2.5-'"${suffix^^}"'-Instruct-AWQ" \
      --host 0.0.0.0 --port '"$port"' \
      --mem-fraction-static 0.9 --dp '"$dp"' \
      --schedule-conservativeness 0.35 \
      --attention-backend flashinfer \
      --enable-torch-compile \
      --grammar-backend outlines \
      &> "'"$log"'"
  '
}

# ------------------------------------------------------------------
#  Launch logic
# ------------------------------------------------------------------
if [[ "$RM_SUFFIX" == "$MRM_SUFFIX" ]]; then
  echo "[INFO] RM and MRM are the same size ➜ one server on all GPUs."
  start_server "0,1,2,3" 4 "$RM_SUFFIX" "$RM_PORT" "$LOG_DIR/rm.log"
  servers=("rm")
else
  start_server "0,1,2"   3 "$RM_SUFFIX"  "$RM_PORT"  "$LOG_DIR/rm.log"
  start_server "3"       1 "$MRM_SUFFIX" "$MRM_PORT" "$LOG_DIR/mrm.log"
  servers=("rm" "mrm")
fi

# ------------------------------------------------------------------
#  Wait for readiness (pure-bash /dev/tcp check)
# ------------------------------------------------------------------
wait_ready () {
  local port=$1 tag=$2
  echo -n "[INFO] Waiting for $tag on port $port "
  for _ in {1..100}; do
    if (exec 3<>"/dev/tcp/0.0.0.0/$port") 2>/dev/null; then
      exec 3<&- 3>&-
      echo "✓ up"; return 0
    fi
    printf '.'; sleep 5
  done
  echo "✗ timeout — check $LOG_DIR/${tag}.log"; exit 1
}

for s in "${servers[@]}"; do
  if [[ $s == "rm"  ]]; then wait_ready "$RM_PORT"  "rm";  fi
  if [[ $s == "mrm" ]]; then wait_ready "$MRM_PORT" "mrm"; fi
done

echo "[INFO] Server(s) running: ${servers[*]}"
REMOTE