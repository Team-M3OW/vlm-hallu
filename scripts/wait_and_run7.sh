#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
until [ -f .q_72c_q2 ]; do sleep 60; done
sleep 10
set -x
[ -f .q_93b ] || { python3 -u scripts/phase93b_pruning_qwen2.py && touch .q_93b; }
echo QUEUE7_DONE
