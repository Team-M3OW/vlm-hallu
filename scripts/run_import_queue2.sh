#!/bin/bash
# Resumed after the ragged-grid fix. phase170 qwen3 is already on disk.
# Track A = CPU (coverage screen), Track B = GPU (extraction + end-task). They overlap.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
L=/home/kavinder/ARNABI_ARSH/vlm-hallu/.gpu_lock

trackB() {
  flock $L python3 -u scripts/phase170_perhead_extract.py qwen2 > logs/phase170_qwen2.log 2>&1 && touch .q_170_qwen2
  flock $L python3 -u scripts/phase172_context_graft.py qwen3   > logs/phase172_qwen3.log 2>&1 && touch .q_172_qwen3
  flock $L python3 -u scripts/phase172_context_graft.py qwen2   > logs/phase172_qwen2.log 2>&1 && touch .q_172_qwen2
  echo TRACK_B_DONE
}
trackA() {
  python3 -u scripts/phase171_head_subset.py qwen3 > logs/phase171_qwen3.log 2>&1 && touch .q_171_qwen3
  until [ -f .q_170_qwen2 ]; do sleep 30; done
  python3 -u scripts/phase171_head_subset.py qwen2 > logs/phase171_qwen2.log 2>&1 && touch .q_171_qwen2
  echo TRACK_A_DONE
}
trackB & B=$!
trackA & A=$!
wait $A $B
echo IMPORT_QUEUE2_DONE
