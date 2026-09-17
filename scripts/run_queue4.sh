#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
set -x
[ -f .q_108b_qwen3 ] || { python3 -u scripts/phase108b_perhead_dump.py qwen3 && touch .q_108b_qwen3; }
[ -f .q_113_qwen3 ]  || { python3 -u scripts/phase113_normwmax_endtask.py qwen3 && touch .q_113_qwen3; }
[ -f .q_109_qwen2 ]  || { python3 -u scripts/phase109_token_selection_probe.py qwen2 && touch .q_109_qwen2; }
[ -f .q_108b_qwen2 ] || { python3 -u scripts/phase108b_perhead_dump.py qwen2 && touch .q_108b_qwen2; }
[ -f .q_104b_qwen2 ] || { python3 -u scripts/phase104_locator_bakeoff.py qwen2 && touch .q_104b_qwen2; }
[ -f .q_113_qwen2 ]  || { python3 -u scripts/phase113_normwmax_endtask.py qwen2 && touch .q_113_qwen2; }
echo QUEUE4_DONE
