#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# ONE queue. Two competing queues caused GPU contention and an OOM last time.
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
wait_gpu () { for i in $(seq 1 480); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
  [ "$f" -ge 24000 ] && return 0; sleep 30; done; return 1; }
for cell in "llava_ov pope" "llava_next vstar" "llava_next hr4k" "llava_next pope" \
            "qwen3_2b pope" "qwen2_7b pope" "gemma3_4b vstar" "gemma3_4b pope"; do
  set -- $cell; wait_gpu
  echo "=== CELL $1 x $2 ==="
  python3 scripts/phase213_multi.py "$1" "$2" 200 2>&1 | grep -vE "it/s|%\|" \
    | grep -E "NL=|achievable|bar E_lo|Done ->|Traceback|Error|OutOfMemory|decode failed" | tail -5
done
for m in stock adaptive; do
  wait_gpu; echo "=== GEMMA $m ==="
  python3 scripts/phase216_gemma_pool.py $m vstar 200 2>&1 | grep -vE "it/s|%\|" | tail -3
done
for m in llava_ov llava_next gemma3_4b; do
  wait_gpu; echo "=== DWA MAPS $m ==="
  python3 scripts/phase214_dwa_multi.py $m 200 2>&1 | grep -vE "it/s|%\|" | tail -3
  wait_gpu; echo "=== DWA EVAL $m ==="
  python3 scripts/phase215_dwa_eval.py $m 2>&1 | grep -vE "it/s|%\|" | tail -5
done
echo "EVERYTHING DONE"
