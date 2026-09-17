#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
set -x
[ -f .q_111_qwen3 ]  || { python3 -u scripts/phase111_pope_scope.py qwen3        && touch .q_111_qwen3; }
[ -f .q_108_qwen3 ]  || { python3 -u scripts/phase108_head_selection.py qwen3    && touch .q_108_qwen3; }
[ -f .q_111_qwen2 ]  || { python3 -u scripts/phase111_pope_scope.py qwen2        && touch .q_111_qwen2; }
[ -f .q_104b_qwen2 ] || { python3 -u scripts/phase104_locator_bakeoff.py qwen2   && touch .q_104b_qwen2; }
[ -f .q_109_qwen2 ]  || { python3 -u scripts/phase109_token_selection_probe.py qwen2 && touch .q_109_qwen2; }
[ -f .q_108_qwen2 ]  || { python3 -u scripts/phase108_head_selection.py qwen2    && touch .q_108_qwen2; }
echo QUEUE_DONE
