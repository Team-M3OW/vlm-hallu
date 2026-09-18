#!/bin/bash
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock; G=scripts/gpuwait.sh
until [ -f .q_189d_qwen2_900 ]; do sleep 30; done
for m in qwen3 qwen2; do python3 scripts/phase189e_transfer.py $m > logs/phase189e_$m.log 2>&1; done
python3 scripts/phase189e_gate.py > logs/phase189e_gate.log 2>&1; rc=$?; cat logs/phase189e_gate.log
if [ $rc -ne 0 ]; then echo "189f NOT launched (gate failed)"; touch .q_189f_skipped; exit 0; fi
export OMP_NUM_THREADS=4
run(){ local need=$1 mark=$2; shift 2; $G "$need" 7200 || { echo "SKIP $mark"; return; }
  flock $L bash -c "$G $need 600 && python3 -u $*" > logs/${mark}.log 2>&1 && touch .$mark && echo "OK $mark" || echo "FAIL $mark"; }
run 16000 q_189f_qwen3 scripts/phase189f_objprune.py qwen3
run 34000 q_189f_qwen2 scripts/phase189f_objprune.py qwen2
echo R189F_DONE
