#!/usr/bin/env bash
# Fetch the two missing Qwen2.5-Omni shards WITHOUT the broken local resolver.
# DIAGNOSIS: systemd-resolved's stub (127.0.0.53) is dead -- getent fails while 1.1.1.1 answers
# fine; and the v6 answers are unroutable here (ping6 100% loss). That, not HF throttling, is what
# kept stalling these downloads at 0 KB/s. We resolve A records via 1.1.1.1 and pin them with
# --resolve, and force IPv4 with -4.
set -u
SNAP=$(ls -d /media/kavinder/hdd2/hf_cache/models--Qwen--Qwen2.5-Omni-7B/snapshots/*/ | head -1)
ips() { nslookup -type=A "$1" 1.1.1.1 2>/dev/null | awk '/^Address: /{print $2}' | grep -E '^[0-9.]+$'; }
RES=""
for h in huggingface.co us.aws.cdn.hf.co eu.aws.cdn.hf.co cdn-lfs.hf.co cdn-lfs-us-1.hf.co cdn-lfs-eu-1.hf.co transfer.xethub.hf.co cas-server.xethub.hf.co; do
  for ip in $(ips "$h"); do RES="$RES --resolve $h:443:$ip"; done
done
echo "[omni] pinned: $(echo $RES | wc -w) resolve entries"
for f in model-00001-of-00005.safetensors model-00004-of-00005.safetensors; do
  DST="$SNAP$f"
  [ -L "$DST" ] && rm -f "$DST"          # replace the dangling symlink with a real file
  URL="https://huggingface.co/Qwen/Qwen2.5-Omni-7B/resolve/main/$f"
  for i in $(seq 1 60); do
    SZ=$(stat -c%s "$DST" 2>/dev/null || echo 0)
    echo "[omni] $f attempt $i from $((SZ/1000000))MB $(date +%H:%M:%S)"
    curl -4 -sL -C - $RES --retry 3 --retry-delay 5 --connect-timeout 30 \
         --speed-limit 20000 --speed-time 120 -o "$DST" "$URL"
    RC=$?
    NEW=$(stat -c%s "$DST" 2>/dev/null || echo 0)
    if [ $RC -eq 0 ] && [ "$NEW" -gt 0 ] && [ "$NEW" -eq "$SZ" ] && [ $i -gt 1 ]; then break; fi
    if [ $RC -eq 0 ] && [ "$NEW" -gt 1000000000 ]; then
      # curl exited clean; assume complete
      echo "[omni] $f done at $((NEW/1000000))MB"; break
    fi
    sleep 5
  done
done
echo "[omni] ALL DONE"; ls -lh "$SNAP"*.safetensors | awk '{print $5,$9}'
