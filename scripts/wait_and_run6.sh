#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
until [ -f .q_116b ] || [ -f .q_5done ]; do sleep 60; done
sleep 10
set -x
OMP_NUM_THREADS=4 python3 -u scripts/phase72a_qwen2_head.py && touch .q_72a_q2
[ -f .q_72c_q2 ] || { python3 -u scripts/phase72c_qwen2.py && touch .q_72c_q2; }
echo QUEUE6_DONE
