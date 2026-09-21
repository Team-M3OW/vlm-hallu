#!/bin/bash
export OMP_NUM_THREADS=8
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
until [ -f data/phase188a_headscores_qwen2.json ]; do sleep 30; done
flock $L python3 -u scripts/phase188_layerwise_prune.py qwen3 --limit 6 > logs/phase188_sanity.log 2>&1 && touch .q_188_sanity
for m in qwen3 qwen2; do
  flock $L python3 -u scripts/phase188_layerwise_prune.py $m > logs/phase188_$m.log 2>&1 && touch .q_188_$m
done
echo PRUNE_QUEUE_DONE
