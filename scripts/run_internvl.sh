#!/bin/bash
# SmolVLM (Idefics3) as a 4th DWA family. Already cached (4.2G) -- disk is 100% full so a
# download was not an option. Layout is MEASURED by the probe, not assumed.
# Launch: setsid nohup bash scripts/run_internvl.sh </dev/null >logs/q_ivl.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_ivl.lock; MINFREE=22000; STABLE=3
exec 8>"$SELF"; flock -n 8 || { echo "another run_internvl.sh is active"; exit 0; }
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
  "$@" >"logs/ivl_${tag}.log" 2>&1; rc=$?
  grep -aE "chance|cov 0|NL=|dumped|coverage|GUARD|Done ->|Traceback|Error|OutOfMemory|cells" "logs/ivl_${tag}.log"|tail -10
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}
run python3 scripts/phase220b_layout_nonsquare.py internvl3_8b 24
echo "INTERNVL PROBE DONE $(date +%H:%M:%S) -- inspect before dumping"
