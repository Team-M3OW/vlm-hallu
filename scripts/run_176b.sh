#!/bin/bash
export OMP_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
flock $L python3 -u scripts/phase176b_allrows_control.py qwen3 > logs/phase176b_qwen3.log 2>&1 && touch .q_176b_qwen3; echo CTRL_DONE
