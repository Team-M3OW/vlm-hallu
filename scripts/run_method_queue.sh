#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
for m in qwen3 qwen2; do flock $L python3 -u scripts/phase179_baselines.py $m > logs/phase179_$m.log 2>&1 && touch .q_179_$m; done
for m in qwen3 qwen2; do flock $L python3 -u scripts/phase176c_transport.py $m > logs/phase176c_$m.log 2>&1 && touch .q_176c_$m; done
echo METHOD_QUEUE_DONE
