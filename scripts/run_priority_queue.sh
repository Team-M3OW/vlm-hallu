#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
run(){ # run <need MiB> <marker> <script> <args...>
  local need=$1 mark=$2; shift 2
  $G "$need" 7200 || { echo "SKIP $mark (no memory)"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"
}
run  9000 q_176c_qwen3 scripts/phase176c_transport.py qwen3        # 2B, fits now
run 22000 q_179_qwen2  scripts/phase179_baselines.py qwen2         # 7B, decides P1 on two models
run 22000 q_176c_qwen2 scripts/phase176c_transport.py qwen2        # 7B
run 22000 q_178_qwen2  scripts/phase178_glocal.py qwen2            # 7B, confirms an already-failed design
echo PRIORITY_QUEUE_DONE
