#!/bin/bash
# Watchdog for the 4x4 grid. Relaunches run_grid.sh whenever it is not running and the grid is
# not yet complete. Safe to run repeatedly: run_grid.sh takes a flock singleton, and phase225
# resumes by qid, so a relaunch never duplicates or recomputes work.
# Exists because the box rebooted mid-run once tonight and killed every queue.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
# already complete?
if python3 scripts/grid_status.py 2>/dev/null | grep -q "cells complete: 16/16"; then
  echo "$(date +%F_%H:%M:%S) grid complete; supervisor idle" >> logs/supervisor.log
  exit 0
fi
# already running? (match the interpreter + exact script, never our own command line)
if pgrep -f "bash scripts/run_grid.sh" >/dev/null 2>&1; then exit 0; fi
echo "$(date +%F_%H:%M:%S) run_grid.sh not running -> relaunching" >> logs/supervisor.log
setsid nohup bash scripts/run_grid.sh </dev/null >>logs/q_grid.log 2>&1 &
disown
