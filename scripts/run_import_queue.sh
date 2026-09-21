#!/bin/bash
# Training-free imports: VRH + ProViP head selection (170 extract -> 171 screen), context graft (172).
# GPU steps take the project lock; the CPU screen does not.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
for m in qwen3 qwen2; do
  flock $L python3 -u scripts/phase170_perhead_extract.py $m > logs/phase170_$m.log 2>&1 && touch .q_170_$m
  python3 -u scripts/phase171_head_subset.py $m > logs/phase171_$m.log 2>&1 && touch .q_171_$m
done
for m in qwen3 qwen2; do
  flock $L python3 -u scripts/phase172_context_graft.py $m > logs/phase172_$m.log 2>&1 && touch .q_172_$m
done
echo IMPORT_QUEUE_DONE
