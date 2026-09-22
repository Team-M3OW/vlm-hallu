#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_m1prof_queue" >/dev/null; do sleep 60; done
python3 scripts/m1_sweep.py
echo "sweep done"
