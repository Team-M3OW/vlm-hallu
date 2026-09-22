#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for cell in "llava_ov vstar" "llava_next vstar" "llava_ov hr4k" "llava_next hr4k" \
            "llava_ov pope" "llava_next pope" "qwen3_2b pope" "qwen2_7b pope" \
            "qwen3_2b vstar" "qwen2_7b vstar"; do
  set -- $cell
  for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge 22000 ] && break
    sleep 30
  done
  echo "=== CELL $1 x $2 (free ${free}MiB) ==="
  python3 scripts/phase213_multi.py "$1" "$2" 200 2>&1 | grep -vE "it/s|%\|" | grep -E "NL=|achievable|bar E_lo|\[|Done ->|Traceback|Error" | tail -6
done
echo "QUEUE DONE"
