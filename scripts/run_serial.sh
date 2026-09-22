#!/bin/bash
# ONE queue, ONE lock. Parallel gates raced and caused two OOMs.
#
# Launch ONLY as:
#   setsid nohup bash scripts/run_serial.sh </dev/null >logs/q_serial.log 2>&1 & disown
# A bare `&` is not enough: three previous queues (q_serial, q_warp, q_post_queue)
# were reaped with their launching process group and produced 0-byte logs.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs

LOCK=/tmp/vlmhallu_gpu.lock       # cross-queue GPU serialisation
SELF=/tmp/vlmhallu_serial.lock    # singleton: one copy of THIS queue at a time
MINFREE=26000                     # MiB of headroom required before a job starts
GATE_TRIES=480                    # 480 * 30s = 4h max wait per job
STABLE=3                          # consecutive samples that must clear MINFREE

exec 8>"$SELF"
flock -n 8 || { echo "another run_serial.sh holds $SELF; exiting"; exit 0; }

freemib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

run () {
  local tag ok=0 good=0 f rc
  tag=$(echo "$*" | sed -E 's#.*/([a-z0-9_]+)\.py#\1#; s/[^A-Za-z0-9_]+/_/g')
  exec 9>"$LOCK"; flock 9
  for _ in $(seq 1 "$GATE_TRIES"); do
    f=$(freemib)
    if ! [[ "$f" =~ ^[0-9]+$ ]]; then echo "gate: nvidia-smi gave '$f'"; good=0; sleep 30; continue; fi
    if [ "$f" -ge "$MINFREE" ]; then
      good=$((good+1)); [ "$good" -ge "$STABLE" ] && { ok=1; break; }
      sleep 10
    else good=0; sleep 30; fi
  done
  if [ "$ok" -ne 1 ]; then
    echo "=== SKIPPED (no ${MINFREE}MiB headroom after gate timeout): $* ==="
    exec 9>&-; return
  fi
  echo "=== $(date +%H:%M:%S) $* (free $(freemib)MiB) -> logs/serial_${tag}.log ==="
  # Full output is preserved; only the console summary is filtered.
  # Redirect (not a pipe) so $? is the job's own exit code.
  "$@" >"logs/serial_${tag}.log" 2>&1
  rc=$?
  grep -aE "NL=|achievable|bar E_lo|GUARD|cov |Done ->|Traceback|Error|OutOfMemory|FEASIBLE|NOT budget|chance" \
       "logs/serial_${tag}.log" | tail -8
  echo "--- EXIT $rc : $* ---"
  exec 9>&-
}

# Rebuilt after the 21:18 reboot killed the previous run mid-job (setsid worked; the box went down).
# Order: (1) the layout probe that unblocks DWA on LLaVA-NeXT, (2) the AVR gap-fills that complete
# the 4-model x 4-benchmark AVR grid, (3) the jobs the reboot interrupted.
run python3 scripts/phase220b_layout_nonsquare.py llava_next 24
run python3 scripts/phase213_multi.py qwen2_7b pope 200
run python3 scripts/phase213_multi.py qwen2_7b hr8k 200
run python3 scripts/phase213_multi.py llava_ov hr8k 200
run python3 scripts/phase213_multi.py llava_next hr8k 200
run python3 scripts/phase213_multi.py qwen3_2b hr8k 200
run python3 scripts/phase219_exit_depth.py qwen2
run python3 scripts/phase217_dwa_warp.py
echo "SERIAL QUEUE DONE $(date +%H:%M:%S)"
