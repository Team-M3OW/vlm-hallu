#!/bin/bash
# THE 4x4 GRID: 4 models x 4 benchmarks, AVR and DWA arms in one pass, FULL public splits.
# No sampling, no per-type filtering, no cherry-picking: vstar 191, hr4k 800, cvbench 2638,
# realworldqa 765 = 4394 items/model. The scorer adapts (mcq2..mcq6 letter logits; short greedy
# generation + normalised exact match for RealworldQA's open items) so no item type is excluded.
# phase225 resumes by qid, so an interrupted cell continues rather than restarting.
# Launch: setsid nohup bash scripts/run_grid.sh </dev/null >logs/q_grid.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_grid.lock; MINFREE=22000; STABLE=3
exec 8>"$SELF"
# NEVER rm this lock file to "clear" a stale lock: deleting it while a holder is alive lets the
# next process create a fresh inode and acquire it, so two queues run concurrently. A real stale
# lock releases itself when its holder exits.
flock -n 8 || { echo "another run_grid.sh already holds $SELF; exiting"; exit 0; }
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
  "$@" >"logs/grid_${tag}.log" 2>&1; rc=$?
  grep -aE "weights=|NL=|items |Done ->|Traceback|Error|OutOfMemory|skipped" "logs/grid_${tag}.log"|tail -6
  echo "--- EXIT $rc : $* ($(date +%H:%M:%S)) ---"
  exec 9>&-
}
# cheap+small first so the grid fills evenly rather than finishing one model at a time
for B in vstar realworldqa hr4k cvbench; do
  for M in qwen3_2b qwen2_7b internvl3_8b llava_ov; do
    run python3 scripts/phase225_newbench.py $M $B 0
  done
done
echo "GRID QUEUE DONE $(date +%H:%M:%S)"
