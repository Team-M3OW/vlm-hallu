#!/bin/bash
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until [ -f .q_188d_qwen2 ]; do sleep 30; done
python3 scripts/phase188_gate.py > logs/phase188_gate.log 2>&1; rc=$?; cat logs/phase188_gate.log
if [ $rc -ne 0 ]; then echo "188b NOT launched (gate failed)"; touch .q_188b_skipped; exit 0; fi
export OMP_NUM_THREADS=4
run(){ local need=$1 mark=$2; shift 2; $G "$need" 7200 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 10000 q_188b_qwen3 scripts/phase188b_hiresloc.py qwen3
run 26000 q_188b_qwen2 scripts/phase188b_hiresloc.py qwen2
echo R188B_DONE
