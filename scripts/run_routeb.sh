#!/usr/bin/env bash
# Route B chain. Launch with:  setsid nohup scripts/run_routeb.sh > logs/routeb.log 2>&1 </dev/null & disown
# Waits on EXPLICIT PIDs -- never `pgrep -f`, which matches this script's own command line
# (that self-match is what hung the earlier wait-shells and killed runs four times).
set -u
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
ZIP=/media/kavinder/hdd2/mmau_pro/data.zip
echo "[routeb] $(date +%H:%M:%S) waiting for $ZIP to reach full size"
while :; do
  SZ=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  [ "$SZ" -ge 47510000000 ] && break
  sleep 120
done
# A partial zip has no readable central directory, so verify before running on it.
if ! python3 -c "import zipfile,sys; z=zipfile.ZipFile('$ZIP'); print('[routeb] zip ok, members:',len(z.namelist()))"; then
  echo "[routeb] ZIP INCOMPLETE OR CORRUPT -- refusing to run"; exit 1
fi
echo "[routeb] $(date +%H:%M:%S) PREFLIGHT: 1 item, both arms, token asserts, peak memory"
if ! python3 scripts/phase246_longaudio_g1.py 1; then
  echo "[routeb] PREFLIGHT FAILED -- not starting the full run"; exit 1
fi
echo "[routeb] $(date +%H:%M:%S) preflight ok; starting G1 (full long+ultra-long split)"
python3 scripts/phase246_longaudio_g1.py
echo "[routeb] $(date +%H:%M:%S) G1 exit=$?"
