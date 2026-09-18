#!/bin/bash
# 188d: 400-token localisation maps (unpruned only) -- the budget-neutral design is loc@400 exit L20 + crop@300 = 16,800 TL
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until [ -f .q_188a_qwen2 ]; do sleep 30; done
run(){ local need=$1 mark=$2; shift 2; $G "$need" 7200 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 10000 q_188d_qwen3 scripts/phase188a_hires_loc_maps.py qwen3 400 nopruned
run 26000 q_188d_qwen2 scripts/phase188a_hires_loc_maps.py qwen2 400 nopruned
echo R188D_DONE
