#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
set -x
[ -f .q_93b ]  || { python3 -u scripts/phase93b_pruning_qwen2.py        && touch .q_93b; }
[ -f .q_116b ] || { python3 -u scripts/phase116b_w_resolution_qwen2.py  && touch .q_116b; }
echo QUEUE5_DONE
