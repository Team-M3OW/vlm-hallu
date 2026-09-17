#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
until grep -q "^DONE" logs/textvqa_download.log 2>/dev/null; do sleep 60; done
flock /home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock python3 -u scripts/phase133_textvqa_attn.py > logs/phase133.log 2>&1 && touch .q_133
OMP_NUM_THREADS=6 python3 -u scripts/phase134_textvqa_head_transfer.py > logs/phase134.log 2>&1 && touch .q_134
echo PIPELINE_DONE
