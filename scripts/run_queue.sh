#!/bin/bash
# Runs the queued GPU jobs in order. No pgrep self-matching: each job writes a marker on success.
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
set -x
python3 -u scripts/phase104_locator_bakeoff.py qwen3 && touch .q_104b_qwen3
python3 -u scripts/phase106_maxwin4_endtask.py qwen2 && touch .q_106_qwen2
python3 -u scripts/phase104_locator_bakeoff.py qwen2 && touch .q_104b_qwen2
echo QUEUE_DONE
