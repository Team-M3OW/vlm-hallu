#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for b in docvqa textvqa; do
  while :; do g=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); [ "$g" -ge 10000 ] && break; sleep 60; done
  echo "=== 230 $b $(date +%H:%M) ===" >> logs/seq.log
  python3 scripts/phase230_lowbudget.py $b 300 > logs/seq_230_$b.log 2>&1
done
echo "230 DONE $(date +%H:%M)" > logs/230_done.flag
