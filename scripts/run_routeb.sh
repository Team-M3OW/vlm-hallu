#!/usr/bin/env bash
# Route B chain. Launch with:  setsid nohup scripts/run_routeb.sh > logs/routeb.log 2>&1 </dev/null & disown
# Waits on EXPLICIT PIDs -- never `pgrep -f`, which matches this script's own command line
# (that self-match is what hung the earlier wait-shells and killed runs four times).
set -u
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
ZIP=/media/kavinder/hdd2/mmau_pro/data.zip
# Poll by OPENING the zip, never by a magic byte count: "47.51 GB" was a rounded display value, so
# a hardcoded threshold can sit above the true size and wait forever. A partial zip has no readable
# central directory, so a successful open IS the completion test.
echo "[routeb] $(date +%H:%M:%S) waiting for $ZIP to become a readable zip"
while :; do
  if python3 -c "import zipfile;z=zipfile.ZipFile('$ZIP');print('[routeb] zip ok, members:',len(z.namelist()))" 2>/dev/null; then break; fi
  sleep 120
done
# Model weights: a hung HF download leaves .incomplete blobs and zero growth (it happened twice
# tonight), so refuse to start rather than fail deep into the run.
INC=$(find /media/kavinder/hdd2/hf_cache/models--Qwen--Qwen2.5-Omni-7B -name "*.incomplete" 2>/dev/null | wc -l)
if [ "$INC" -ne 0 ]; then
  echo "[routeb] $INC incomplete Omni weight shards -- waiting"
  while [ "$(find /media/kavinder/hdd2/hf_cache/models--Qwen--Qwen2.5-Omni-7B -name '*.incomplete' 2>/dev/null | wc -l)" -ne 0 ]; do sleep 120; done
fi
echo "[routeb] $(date +%H:%M:%S) PREFLIGHT: 1 item, both arms, token asserts, peak memory"
if ! python3 scripts/phase246_longaudio_g1.py 1; then
  echo "[routeb] PREFLIGHT FAILED -- not starting the full run"; exit 1
fi
echo "[routeb] $(date +%H:%M:%S) preflight ok; starting G1 (full long+ultra-long split)"
python3 scripts/phase246_longaudio_g1.py
echo "[routeb] $(date +%H:%M:%S) G1 exit=$?"
