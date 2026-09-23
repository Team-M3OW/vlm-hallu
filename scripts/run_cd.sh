#!/bin/bash
# Phase 226: DWA-guided contrastive decoding on CV-Bench (no-headroom regime).
# Shares /tmp/vlmhallu_gpu.lock with the grid so the two never collide.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_cd.lock; MINFREE=20000; STABLE=3
exec 8>"$SELF"
flock -n 8 || { echo "another run_cd.sh holds $SELF; exiting"; exit 0; }
freemib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|head -1; }
run(){
  local tag ok=0 good=0 f rc
  tag=$(echo "$*"|sed -E 's#.*/([a-z0-9_]+)\.py#\1#; s/[^A-Za-z0-9_]+/_/g')
  exec 9>"$LOCK"; flock 9
  for _ in $(seq 1 960); do
    f=$(freemib)
    if ! [[ "$f" =~ ^[0-9]+$ ]]; then good=0; sleep 30; continue; fi
    if [ "$f" -ge "$MINFREE" ]; then good=$((good+1)); [ "$good" -ge "$STABLE" ] && { ok=1; break; }; sleep 10
    else good=0; sleep 30; fi
  done
  [ "$ok" -ne 1 ] && { echo "=== SKIPPED (no headroom): $* ==="; exec 9>&-; return; }
  echo "=== $(date +%H:%M:%S) $* ==="
  "$@" >"logs/cd_${tag}.log" 2>&1; rc=$?
  grep -aE "NL=|items |Done ->|Traceback|Error|OutOfMemory" "logs/cd_${tag}.log"|tail -6
  echo "--- EXIT $rc : $* ($(date +%H:%M:%S)) ---"
  exec 9>&-
}
run python3 scripts/phase226_cd_dwa.py qwen3_2b cvbench 0
echo "CD QUEUE DONE $(date +%H:%M:%S)"
