#!/bin/bash
# InternVL3-8B as the 4th DWA family. Layout MEASURED by phase220b: 16x16 at 448px, cov 0.792 = 10.1x
# chance, best band L18-27 (post-boundary, as the transport account predicts).
# Launch: setsid nohup bash scripts/run_ivl_full.sh </dev/null >logs/q_ivlfull.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_ivlfull.lock; MINFREE=22000; STABLE=3
exec 8>"$SELF"; flock -n 8 || { echo "another run_ivl_full.sh is active"; exit 0; }
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
  "$@" >"logs/ivlfull_${tag}.log" 2>&1; rc=$?
  grep -aE "coverage|GUARD|NL=|cells|items|Done ->|Traceback|Error|OutOfMemory|no usable" "logs/ivlfull_${tag}.log"|tail -10
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}
run python3 scripts/phase214_dwa_multi.py internvl3_8b 191
run python3 scripts/fit_ridge_weights.py internvl3_8b
run python3 scripts/phase215_dwa_eval.py internvl3_8b
run python3 scripts/phase224_dwa_transfer.py internvl3_8b hr4k 400
echo "INTERNVL FULL QUEUE DONE $(date +%H:%M:%S)"
