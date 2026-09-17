#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
until [ -f .q_141_done ]; do sleep 60; done
flock /home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock python3 -u scripts/phase142_llava_divergence.py > logs/phase142.log 2>&1
OMP_NUM_THREADS=4 python3 -u scripts/phase142b_llava_gate.py > logs/phase142b.log 2>&1
touch .q_142_done
