#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
flock /home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock python3 -u scripts/phase121b_q_instr.py qwen3 && touch .q_121b_qwen3
flock /home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock python3 -u scripts/phase121b_q_instr.py qwen2 && touch .q_121b_qwen2
echo DONE_121B
