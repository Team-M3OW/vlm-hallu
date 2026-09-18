#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
# wait for MARKER FILES, not log strings: a failed run writes "FAIL" to the log but never touches the marker
until [ -f .q_195_4k_qwen3 ] && [ -f .q_195_8k_qwen3 ] && [ -f .q_195_4k_qwen2 ] && [ -f .q_192_q25_7b ] && [ -f .q_192_q3_8b ]; do sleep 60; done
$G 34000 14400 && flock $L bash -c "$G 34000 600 && python3 -u scripts/phase189f_objprune.py qwen2" > logs/q_189f_qwen2.log 2>&1 && touch .q_189f_qwen2 && echo "OK q_189f_qwen2" || echo "FAIL q_189f_qwen2"
echo R189F_FINAL_DONE
