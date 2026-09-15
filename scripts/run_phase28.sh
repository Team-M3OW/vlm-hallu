#!/bin/bash
# Wait for RQ-C model downloads, then sweep each architecture in turn.
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
export HF_HUB_CACHE=/media/kavinder/hdd2/hf_cache
until grep -q "ALL DONE" logs/rqc_download.log 2>/dev/null; do
  if grep -qi "error\|Traceback" logs/rqc_download.log 2>/dev/null; then
    echo "DOWNLOAD FAILED"; tail -5 logs/rqc_download.log; exit 1; fi
  sleep 60
done
echo "downloads complete: $(du -sh $HF_HUB_CACHE | cut -f1)"
for m in qwen2vl onevision llavanext; do
  echo "=== starting $m at $(date +%H:%M) ==="
  python -u scripts/phase28_arch_invariance.py $m 2>&1 | grep --line-buffered -v "Loading weights\|Fetching"
  echo "=== finished $m at $(date +%H:%M) ==="
done
echo "PHASE28 ALL MODELS DONE"
