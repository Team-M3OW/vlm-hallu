#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
until grep -q ARCH_QUEUE2_DONE logs/arch_queue2.log 2>/dev/null; do sleep 60; done
for m in qwen3 qwen2; do flock $L python3 -u scripts/phase162b_teacher_head.py $m > logs/phase162b_$m.log 2>&1 && touch .q_162b_$m; done
echo ARCH_QUEUE3_DONE
