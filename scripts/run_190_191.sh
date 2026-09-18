#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
run(){ local need=$1 mark=$2; shift 2; $G "$need" 14400 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
# relational first (fast, decisive), then the breadth replications
run 10000 q_191_qwen3 scripts/phase191_ctxcrop.py qwen3
run 26000 q_191_qwen2 scripts/phase191_ctxcrop.py qwen2
run 26000 q_190_q25_7b scripts/phase190_newmodel_ridge.py q25_7b Qwen/Qwen2.5-VL-7B-Instruct
run 28000 q_190_q3_8b  scripts/phase190_newmodel_ridge.py q3_8b Qwen/Qwen3-VL-8B-Instruct
run 12000 q_190_hr4k_qwen3 scripts/phase190_hr4k_qwen3_ridge.py
run 12000 q_190_hr8k_qwen3 scripts/phase190_hr8k_qwen3_ridge.py
run 30000 q_190_hr4k_qwen2 scripts/phase190_hr4k_qwen2_ridge.py
echo R190_191_DONE
