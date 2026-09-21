#!/bin/bash
export OMP_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
flock $L python3 -u scripts/phase180_dcr_fovea.py qwen3 --limit 12 --viz > logs/phase180_sanity.log 2>&1 && touch .q_180_sanity
for m in qwen3 qwen2; do
  flock $L python3 -u scripts/phase180_dcr_fovea.py $m > logs/phase180_$m.log 2>&1 && touch .q_180_$m
done
echo FOVEA_QUEUE_DONE
