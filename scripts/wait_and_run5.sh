#!/bin/bash
# Waits on a FILE COUNT, not a process name -- a pgrep pattern would match this script's own
# command line, which deadlocked the queue three times today.
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
until [ "$(wc -l < data/phase72c_hrbench_fullprompt.jsonl 2>/dev/null || echo 0)" -ge 800 ]; do sleep 45; done
sleep 15
bash scripts/run_queue5.sh
