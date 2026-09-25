#!/usr/bin/env bash
# Route B chain. Launch with:  setsid nohup scripts/run_routeb.sh > logs/routeb.log 2>&1 </dev/null & disown
# Waits on EXPLICIT PIDs -- never `pgrep -f`, which matches this script's own command line
# (that self-match is what hung the earlier wait-shells and killed runs four times).
set -u
DL_PID="${1:-}"
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
echo "[routeb] $(date +%H:%M:%S) waiting on data download pid=$DL_PID"
if [ -n "$DL_PID" ]; then while kill -0 "$DL_PID" 2>/dev/null; do sleep 60; done; fi
if ! grep -q DONE logs/mmaupro_dl.log 2>/dev/null; then
  echo "[routeb] DATA DOWNLOAD DID NOT REPORT DONE -- aborting rather than running on a partial zip"
  tail -5 logs/mmaupro_dl.log; exit 1
fi
echo "[routeb] $(date +%H:%M:%S) data ready; waiting for model weights pid=${2:-none}"
if [ -n "${2:-}" ]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; fi
echo "[routeb] $(date +%H:%M:%S) PREFLIGHT: 1 item, both arms, token asserts, peak memory"
if ! python3 scripts/phase246_longaudio_g1.py 1; then
  echo "[routeb] PREFLIGHT FAILED -- not starting the full run"; exit 1
fi
echo "[routeb] $(date +%H:%M:%S) preflight ok; starting G1 (full long+ultra-long split)"
python3 scripts/phase246_longaudio_g1.py
echo "[routeb] $(date +%H:%M:%S) G1 exit=$?"
