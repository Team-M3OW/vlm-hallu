#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
run(){ local need=$1 mark=$2; shift 2; $G "$need" 14400 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 16000 q_195_4k_qwen3 scripts/phase195_tsr_hrbench.py 4k qwen3
run 16000 q_195_8k_qwen3 scripts/phase195_tsr_hrbench.py 8k qwen3
run 34000 q_195_4k_qwen2 scripts/phase195_tsr_hrbench.py 4k qwen2
echo R195_DONE
