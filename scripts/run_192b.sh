#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until grep -q R195_DONE logs/run_195.log 2>/dev/null; do sleep 60; done
run(){ local need=$1 mark=$2; shift 2; $G "$need" 14400 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 32000 q_192_q25_7b scripts/phase192_tsr_newmodel.py q25_7b Qwen/Qwen2.5-VL-7B-Instruct
run 34000 q_192_q3_8b  scripts/phase192_tsr_newmodel.py q3_8b  Qwen/Qwen3-VL-8B-Instruct
echo R192B_DONE
