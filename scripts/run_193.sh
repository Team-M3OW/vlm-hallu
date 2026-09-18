#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until grep -q R190_191_DONE logs/run_190_191.log 2>/dev/null; do sleep 60; done
run(){ local need=$1 mark=$2; shift 2; $G "$need" 14400 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 12000 q_193_qwen3 scripts/phase193_multicrop.py qwen3
run 28000 q_193_qwen2 scripts/phase193_multicrop.py qwen2
echo R193_DONE
