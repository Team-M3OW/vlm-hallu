#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for m in qwen3_2b qwen2_7b llava_ov internvl3_8b; do
  echo "=== $m textvqa $(date +%H:%M) ==="
  python3 scripts/phase225_newbench.py $m textvqa
done
echo "TEXTVQA GRID DONE $(date +%H:%M)"
