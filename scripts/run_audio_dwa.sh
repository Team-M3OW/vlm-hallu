#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
exec 8>/tmp/vlmhallu_audiodwa.lock; flock -n 8 || { echo "already running"; exit 0; }
exec 9>/tmp/vlmhallu_gpu.lock; flock 9
echo "=== $(date +%H:%M:%S) phase244 DWA-audio ceiling n=180 K=8 ==="
PYTHONWARNINGS=ignore python3 scripts/phase244_audio_dwa_ceiling.py 180 8 >logs/audio_phase244.log 2>&1
rc=$?
grep -aE "NL=|stratified|\[[0-9]+/|Done ->|Traceback|Error" logs/audio_phase244.log | tail -10
echo "--- EXIT $rc ($(date +%H:%M:%S)) ---"
exec 9>&-
