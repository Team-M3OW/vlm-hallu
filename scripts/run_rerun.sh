#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
wait_gpu () { for i in $(seq 1 480); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
  [ "$f" -ge 24000 ] && return 0; sleep 30; done; }
wait_gpu; echo "=== RERUN gemma3_4b x vstar ==="
python3 scripts/phase213_multi.py gemma3_4b vstar 200 2>&1 | grep -vE "it/s|%\|" | grep -E "NL=|achievable|bar E_lo|Done ->|Error" | tail -4
wait_gpu; echo "=== RERUN gemma3_4b x pope ==="
python3 scripts/phase213_multi.py gemma3_4b pope 200 2>&1 | grep -vE "it/s|%\|" | grep -E "NL=|achievable|bar E_lo|Done ->|Error" | tail -4
wait_gpu; echo "=== RERUN gemma adaptive ==="
python3 scripts/phase216_gemma_pool.py adaptive vstar 200 2>&1 | grep -vE "it/s|%\|" | tail -3
wait_gpu; echo "=== RERUN qwen2_7b x pope (OOM before) ==="
python3 scripts/phase213_multi.py qwen2_7b pope 200 2>&1 | grep -vE "it/s|%\|" | grep -E "Done ->|Error|OutOfMemory" | tail -3
echo "RERUN DONE"
