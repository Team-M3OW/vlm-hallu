#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until [ -f .q_192_q25_7b ]; do sleep 30; done
$G 34000 14400 && flock $L bash -c "$G 34000 600 && python3 -u scripts/phase192_tsr_newmodel.py q3_8b Qwen/Qwen3-VL-8B-Instruct" > logs/q_192_q3_8b.log 2>&1 && touch .q_192_q3_8b && echo "OK q_192_q3_8b" || echo "FAIL q_192_q3_8b"
echo R192D_DONE
