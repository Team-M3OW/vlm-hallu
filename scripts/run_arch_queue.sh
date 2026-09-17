#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
for m in qwen3 qwen2; do flock $L python3 -u scripts/phase161_span_window.py $m > logs/phase161_$m.log 2>&1 && touch .q_161_$m; done
for m in qwen3 qwen2; do flock $L python3 -u scripts/phase162_pseudolabel_head.py $m > logs/phase162_$m.log 2>&1 && touch .q_162_$m; done
echo ARCH_QUEUE_DONE
