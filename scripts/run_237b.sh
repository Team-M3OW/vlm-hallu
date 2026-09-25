#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for spec in "qwen2_7b vstar" "qwen2_7b hr4k" "qwen3_2b realworldqa" "qwen2_7b cvbench" "qwen2_7b realworldqa"; do
  set -- $spec
  while :; do
    ram=$(free -g | awk '/^Mem:/{print $7}')
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$ram" -ge 20 ] && [ "$gpu" -ge 8000 ] && break
    sleep 60
  done
  echo "=== START 237b $1 $2 $(date +%H:%M) ===" >> logs/seq.log
  python3 scripts/phase237_signal_dump.py $1 $2 > logs/seq_237_${1}_${2}.log 2>&1
  echo "=== DONE 237b $1 $2 rc=$? $(date +%H:%M) ===" >> logs/seq.log
done
echo "237b DONE $(date +%H:%M)" > logs/237b_done.flag
