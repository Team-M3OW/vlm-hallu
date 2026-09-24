#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for spec in "vstar 191" "textvqa 500" "docvqa 400" "hr4k 300"; do
  set -- $spec
  echo "=== 228 qwen3_2b $1 $2 $(date +%H:%M) ==="
  python3 scripts/phase228_equal300.py qwen3_2b $1 $2
done
echo "228 DONE $(date +%H:%M)"
