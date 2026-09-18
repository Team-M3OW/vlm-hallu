#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until [ -f .q_188b_qwen2 ] || [ -f .q_188b_skipped ]; do sleep 60; done
$G 32000 7200 && flock $L bash -c "$G 32000 600 && python3 -u scripts/phase187_pyramid.py qwen2" > logs/q_187_qwen2.log 2>&1 && touch .q_187_qwen2 && echo "OK q_187_qwen2" || echo "FAIL q_187_qwen2"
echo R187D_DONE
