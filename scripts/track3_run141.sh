#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
flock /home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock python3 -u scripts/phase141_trainfree_endtask.py qwen3 > logs/phase141_qwen3.log 2>&1
flock /home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock python3 -u scripts/phase141_trainfree_endtask.py qwen2 > logs/phase141_qwen2.log 2>&1
touch .q_141_done
