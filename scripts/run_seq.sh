#!/bin/bash
# STRICTLY SEQUENTIAL, resource-guarded. One model at a time (the GPU-OOM and silent
# OOM-kill incidents both came from concurrency: 2-3 x 7-8B models and two full-dataset
# image materialisations). Waits for >=20 GB RAM free and >=8 GB GPU free before each job.
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
JOBS="
scale qwen3_2b docvqa 600
grid qwen3_2b docvqa 1000
grid qwen2_7b docvqa 1000
grid llava_ov docvqa 1000
grid internvl3_8b docvqa 1000
scale qwen2_7b docvqa 600
grid qwen3_2b textvqa 5000
grid qwen2_7b textvqa 5000
grid llava_ov textvqa 5000
grid internvl3_8b textvqa 5000
grid qwen3_2b gqa 2000
grid qwen2_7b gqa 2000
grid llava_ov gqa 2000
grid internvl3_8b gqa 2000
"
echo "$JOBS" | while read kind model bench n; do
  [ -z "$kind" ] && continue
  while :; do
    ram=$(free -g | awk '/^Mem:/{print $7}')
    gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$ram" -ge 20 ] && [ "$gpu" -ge 8000 ]; then break; fi
    echo "waiting: ram=${ram}G gpu=${gpu}M $(date +%H:%M)" >> logs/seq.log
    sleep 60
  done
  echo "=== START $kind $model $bench $n $(date +%H:%M) ram=$(free -g|awk '/^Mem:/{print $7}')G gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)M ===" >> logs/seq.log
  if [ "$kind" = "scale" ]; then
    python3 scripts/phase227_dwa_scale.py "$model" "$bench" "$n" > "logs/seq_${kind}_${model}_${bench}.log" 2>&1
  else
    python3 scripts/phase225_newbench.py "$model" "$bench" "$n" > "logs/seq_${kind}_${model}_${bench}.log" 2>&1
  fi
  echo "=== DONE $kind $model $bench rc=$? $(date +%H:%M) ===" >> logs/seq.log
done
echo "SEQ DONE $(date +%H:%M)" > logs/seq_done.flag
