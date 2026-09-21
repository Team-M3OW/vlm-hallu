#!/bin/bash
export OMP_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
for m in qwen3 qwen2; do
  flock $L python3 -u scripts/phase183_vrh_reinforce.py $m > logs/phase183_$m.log 2>&1 && touch .q_183_$m
done
echo REINFORCE_DONE
