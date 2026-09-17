#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
# router: models already cached -- run first
flock $L python3 -u scripts/phase153_model_router.py > logs/phase153.log 2>&1 && touch .q_153
until grep -q "DOWNLOADS_DONE" logs/downloads_w6.log 2>/dev/null; do sleep 60; done
flock $L python3 -u scripts/phase155_hrbench8k.py > logs/phase155.log 2>&1 && touch .q_155
for spec in "q3_8b Qwen/Qwen3-VL-8B-Instruct" "q25_7b Qwen/Qwen2.5-VL-7B-Instruct"; do
  set -- $spec
  flock $L python3 -u scripts/phase154_newmodel_pipeline.py $1 $2 extract > logs/phase154_$1_extract.log 2>&1 && \
  OMP_NUM_THREADS=6 python3 -u scripts/phase154_newmodel_pipeline.py $1 $2 head > logs/phase154_$1_head.log 2>&1 && \
  flock $L python3 -u scripts/phase154_newmodel_pipeline.py $1 $2 endtask > logs/phase154_$1_endtask.log 2>&1 && touch .q_154_$1
done
echo REVIEWER_QUEUE_DONE
