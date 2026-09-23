#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_textvqa" >/dev/null; do sleep 60; done
for m in qwen3_2b qwen2_7b llava_ov internvl3_8b; do
  echo "=== $m gqa $(date +%H:%M) ==="
  python3 scripts/phase225_newbench.py $m gqa 2000
done
echo "GQA DONE $(date +%H:%M)"
