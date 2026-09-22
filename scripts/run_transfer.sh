#!/bin/bash
# DWA cross-benchmark transfer, Qwen first (user priority).
# Shares /tmp/vlmhallu_gpu.lock with run_serial.sh so the two queues never collide on the GPU.
# Launch ONLY as: setsid nohup bash scripts/run_transfer.sh </dev/null >logs/q_transfer.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_transfer.lock; MINFREE=20000; STABLE=3
exec 8>"$SELF"; flock -n 8 || { echo "another run_transfer.sh is active; exiting"; exit 0; }
freemib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1; }
run(){
  local tag ok=0 good=0 f rc
  tag=$(echo "$*"|sed -E 's#.*/([a-z0-9_]+)\.py#\1#; s/[^A-Za-z0-9_]+/_/g')
  exec 9>"$LOCK"; flock 9
  for _ in $(seq 1 480); do
    f=$(freemib)
    if ! [[ "$f" =~ ^[0-9]+$ ]]; then echo "gate: nvidia-smi gave '$f'"; good=0; sleep 30; continue; fi
    if [ "$f" -ge "$MINFREE" ]; then good=$((good+1)); [ "$good" -ge "$STABLE" ] && { ok=1; break; }; sleep 10
    else good=0; sleep 30; fi
  done
  [ "$ok" -ne 1 ] && { echo "=== SKIPPED (no headroom): $* ==="; exec 9>&-; return; }
  echo "=== $(date +%H:%M:%S) $* (free $(freemib)MiB) ==="
  "$@" >"logs/transfer_${tag}.log" 2>&1; rc=$?
  grep -aE "loaded V\*|items from|standardised|no usable grid|Done ->|Traceback|Error|OutOfMemory" "logs/transfer_${tag}.log"|tail -8
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}
run python3 scripts/phase224_dwa_transfer.py qwen3_2b hr4k 400
run python3 scripts/phase224_dwa_transfer.py qwen2_7b hr4k 400
run python3 scripts/phase224_dwa_transfer.py qwen3_2b hr8k 400
run python3 scripts/phase224_dwa_transfer.py qwen2_7b hr8k 400
echo "TRANSFER QUEUE DONE $(date +%H:%M:%S)"
