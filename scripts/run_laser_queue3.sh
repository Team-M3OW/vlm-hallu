#!/bin/bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu; L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock
S=scripts/phase173_laser_maps.py
flock $L python3 -u $S qwen3 bf16      > logs/phase173_qwen3.log 2>&1      && touch .q_173_qwen3
flock $L python3 -u $S qwen3 fp16      > logs/phase173_qwen3_fp16.log 2>&1 && touch .q_173_qwen3_fp16
flock $L python3 -u $S qwen3 bf16 _rep > logs/phase173_qwen3_rep.log 2>&1  && touch .q_173_qwen3_rep
flock $L python3 -u $S qwen2 bf16      > logs/phase173_qwen2.log 2>&1      && touch .q_173_qwen2
echo LASER_QUEUE3_DONE
