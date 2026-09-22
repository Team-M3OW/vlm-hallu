#!/bin/bash
# LLaVA-NeXT as the 4th DWA family: re-dump at the MEASURED 25x25 suffix grid, fit, evaluate.
# Launch: setsid nohup bash scripts/run_llavanext.sh </dev/null >logs/q_lnext.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_lnext.lock; MINFREE=20000; STABLE=3
exec 8>"$SELF"; flock -n 8 || { echo "another run_llavanext.sh is active"; exit 0; }
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
  "$@" >"logs/lnext_${tag}.log" 2>&1; rc=$?
  grep -aE "coverage|GUARD|NL=|items|Done ->|Traceback|Error|OutOfMemory|cells" "logs/lnext_${tag}.log"|tail -8
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}
run python3 scripts/phase214_dwa_multi.py llava_next 191
run python3 scripts/fit_ridge_weights.py llava_next
run python3 scripts/phase215_dwa_eval.py llava_next
run python3 scripts/phase224_dwa_transfer.py llava_next hr4k 400
echo "LLAVA-NEXT QUEUE DONE $(date +%H:%M:%S)"
