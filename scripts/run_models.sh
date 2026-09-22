#!/bin/bash
# Expanding DWA beyond Qwen. Shares /tmp/vlmhallu_gpu.lock with the other queues.
# Launch: setsid nohup bash scripts/run_models.sh </dev/null >logs/q_models.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_models.lock; MINFREE=20000; STABLE=3
exec 8>"$SELF"; flock -n 8 || { echo "another run_models.sh is active"; exit 0; }
freemib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1; }
run(){
  local tag ok=0 good=0 f rc
  tag=$(echo "$*"|sed -E 's#.*/([a-z0-9_]+)\.py#\1#; s/[^A-Za-z0-9_]+/_/g')
  exec 9>"$LOCK"; flock 9
  for _ in $(seq 1 480); do
    f=$(freemib)
    if ! [[ "$f" =~ ^[0-9]+$ ]]; then good=0; sleep 30; continue; fi
    if [ "$f" -ge "$MINFREE" ]; then good=$((good+1)); [ "$good" -ge "$STABLE" ] && { ok=1; break; }; sleep 10
    else good=0; sleep 30; fi
  done
  [ "$ok" -ne 1 ] && { echo "=== SKIPPED (no headroom): $* ==="; exec 9>&-; return; }
  echo "=== $(date +%H:%M:%S) $* ==="
  "$@" >"logs/models_${tag}.log" 2>&1; rc=$?
  grep -aE "coverage|GUARD|items|Done ->|Traceback|Error|OutOfMemory|loaded V" "logs/models_${tag}.log"|tail -8
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}
# 1. Gemma: the guard was gating on the block-mean incumbent, not the grid. Ridge is 5.6x chance.
run python3 scripts/phase215_dwa_eval.py gemma3_4b
# 2. LLaVA-OV: weights fitted from its own V* maps; transfer to HR-Bench.
run python3 scripts/phase224_dwa_transfer.py llava_ov hr4k 400
run python3 scripts/phase224_dwa_transfer.py llava_ov hr8k 400
# 3. LLaVA-OV re-run at MATCHED budget (the old number was ~2.3x the bar).
run python3 scripts/phase215_dwa_eval.py llava_ov
echo "MODELS QUEUE DONE $(date +%H:%M:%S)"
