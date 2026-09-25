#!/bin/bash
# Track A3: AVR for audio via time-stretch, pruned at the MEASURED boundary L24.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
exec 8>/tmp/vlmhallu_audioavr.lock; flock -n 8 || { echo "already running"; exit 0; }
exec 9>/tmp/vlmhallu_gpu.lock; flock 9
echo "=== $(date +%H:%M:%S) phase243 audio AVR n=240 ==="
PYTHONWARNINGS=ignore python3 scripts/phase243_audio_avr.py 240 >logs/audio_phase243.log 2>&1
rc=$?
grep -aE "NL=|stratified|\[[0-9]+/|Done ->|Traceback|Error" logs/audio_phase243.log | tail -12
echo "--- EXIT $rc ($(date +%H:%M:%S)) ---"
exec 9>&-
