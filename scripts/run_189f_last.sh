#!/bin/bash
# the 7B objectness leg runs LAST, after every user-requested queue
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until grep -q R192B_DONE logs/run_192b.log 2>/dev/null; do sleep 60; done
$G 34000 14400 && flock $L bash -c "$G 34000 600 && python3 -u scripts/phase189f_objprune.py qwen2" > logs/q_189f_qwen2.log 2>&1 && touch .q_189f_qwen2 && echo "OK q_189f_qwen2" || echo "FAIL q_189f_qwen2"
echo R189F_LAST_DONE
