#!/usr/bin/env bash
# Resumable fetch of MMAU-Pro data.zip. hf_hub_download hung repeatedly (26 open sockets, zero
# file growth) and hf_transfer stalled outright; HF throttles this repo in AGGREGATE (4 parallel
# range requests split ~2.3MB/s rather than multiplying it), so a single resumable stream with
# retries is the right tool. curl -C - appends to whatever is already on disk.
DST=/media/kavinder/hdd2/mmau_pro/data.zip
URL=https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/data.zip
for i in $(seq 1 200); do
  SZ=$(stat -c%s "$DST" 2>/dev/null || echo 0)
  if [ "$SZ" -ge 47510000000 ]; then echo "[dl] COMPLETE $SZ"; break; fi
  echo "[dl] attempt $i from $((SZ/1000000)) MB $(date +%H:%M:%S)"
  curl -sL -C - --retry 5 --retry-delay 10 --connect-timeout 30 --speed-limit 20000 --speed-time 120 -o "$DST" "$URL"
  sleep 5
done
echo "[dl] final $(stat -c%s "$DST") bytes"
