#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
for mk in qwen3_2b qwen2_7b llava_ov internvl3_8b; do
  for bk in hr4k cvbench realworldqa; do
    while :; do
      ram=$(free -g | awk '/^Mem:/{print $7}')
      gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$ram" -ge 20 ] && [ "$gpu" -ge 8000 ] && break
      echo "236 waiting: ram=${ram}G gpu=${gpu}M $(date +%H:%M)" >> logs/seq.log; sleep 60
    done
    echo "=== START 236 $mk $bk $(date +%H:%M) ===" >> logs/seq.log
    python3 scripts/phase236_disp_dump.py $mk $bk > logs/seq_236_${mk}_${bk}.log 2>&1
    echo "=== DONE 236 $mk $bk rc=$? $(date +%H:%M) ===" >> logs/seq.log
  done
done
echo "236 DONE $(date +%H:%M)" > logs/236_done.flag
