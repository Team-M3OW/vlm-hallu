#!/bin/bash
# 189f runs ONLY after the user's 190/191 queue is finished.
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until grep -q R190_191_DONE logs/run_190_191.log 2>/dev/null; do sleep 60; done
python3 scripts/phase189e_gate.py > logs/phase189e_gate.log 2>&1; rc=$?; cat logs/phase189e_gate.log
if [ $rc -ne 0 ]; then echo "189f NOT launched (gate failed)"; touch .q_189f_skipped; exit 0; fi
export OMP_NUM_THREADS=4
run(){ local need=$1 mark=$2; shift 2; $G "$need" 14400 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 16000 q_189f_qwen3 scripts/phase189f_objprune.py qwen3
run 34000 q_189f_qwen2 scripts/phase189f_objprune.py qwen2
echo R189F_DONE
