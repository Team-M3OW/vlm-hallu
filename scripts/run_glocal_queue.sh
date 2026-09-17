#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
for m in qwen3 qwen2; do flock $L python3 -u scripts/phase178_glocal.py $m > logs/phase178_$m.log 2>&1 && touch .q_178_$m; done
echo GLOCAL_QUEUE_DONE
