#!/bin/bash
# Retries after three layout fixes:
#  - phase220b crashed on SmolVLM/InternVL because both pick tile count from ASPECT RATIO, so n_img
#    varied per item and the dump could not be stacked. Now a fixed square input + modal filter.
#  - phase224 had no llava_ov entry, fell through to a perfect-square test, and skipped every item.
# Launch: setsid nohup bash scripts/run_probe2.sh </dev/null >logs/q_probe2.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_probe2.lock; MINFREE=22000; STABLE=3
exec 8>"$SELF"; flock -n 8 || { echo "another run_probe2.sh is active"; exit 0; }
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
  "$@" >"logs/probe2_${tag}.log" 2>&1; rc=$?
  grep -aE "chance|cov 0\.|dumped|dropping|NL=|items from|Done ->|Traceback|Error|OutOfMemory|no usable" "logs/probe2_${tag}.log"|tail -10
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}
run python3 scripts/phase220b_layout_nonsquare.py internvl3_8b 24
run python3 scripts/phase220b_layout_nonsquare.py smolvlm 24
run python3 scripts/phase224_dwa_transfer.py llava_ov hr4k 400
echo "PROBE2 QUEUE DONE $(date +%H:%M:%S)"
