#!/bin/bash
# Track A1: audio transport boundary (Qwen2-Audio-7B-Instruct on MMAU).
# Launch: setsid nohup bash scripts/run_audio.sh </dev/null >logs/q_audio.log 2>&1 & disown
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu || exit 1
mkdir -p logs
LOCK=/tmp/vlmhallu_gpu.lock; SELF=/tmp/vlmhallu_audio.lock
exec 8>"$SELF"; flock -n 8 || { echo "another run_audio.sh holds $SELF"; exit 0; }
exec 9>"$LOCK"; flock 9
echo "=== $(date +%H:%M:%S) phase241 audio transport n=40 ==="
python3 scripts/phase241_audio_transport.py 60 >logs/audio_phase241.log 2>&1
rc=$?
grep -aE "NL=|PRE-REG|SANITY|n=|mean per-layer|^   L|prefix masks|suffix masks|PREDICTED|Done ->|Traceback|Error" logs/audio_phase241.log | tail -60
echo "--- EXIT $rc ($(date +%H:%M:%S)) ---"
exec 9>&-
