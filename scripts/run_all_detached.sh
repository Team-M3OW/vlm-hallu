#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
# resume downloads (snapshot_download is resumable)
python3 /tmp/dl_gemma.py >> logs/dl_gemma.log 2>&1
python3 /tmp/dl_ib.py    >> logs/dl_instructblip.log 2>&1
echo "=== downloads done ==="
for cell in "llava_next vstar" "llava_ov hr4k" "llava_next hr4k" "llava_ov pope" \
            "llava_next pope" "qwen3_2b pope" "qwen2_7b pope" "gemma3_4b vstar" "gemma3_4b pope"; do
  set -- $cell
  for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
    [ "$free" -ge 22000 ] && break; sleep 30
  done
  echo "=== CELL $1 x $2 ==="
  python3 scripts/phase213_multi.py "$1" "$2" 200 2>&1 | grep -vE "it/s|%\|" | grep -E "NL=|achievable|bar E_lo|Done ->|Traceback|Error|decode failed" | tail -5
done
echo "=== MATRIX DONE; starting DWA ==="
for m in llava_ov llava_next gemma3_4b; do
  echo "=== DWA MAPS $m ==="
  python3 scripts/phase214_dwa_multi.py $m 200 2>&1 | grep -vE "it/s|%\|" | tail -3
  echo "=== DWA EVAL $m ==="
  python3 scripts/phase215_dwa_eval.py $m 2>&1 | grep -vE "it/s|%\|" | tail -5
done
echo "ALL DONE"
