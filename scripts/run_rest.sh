#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "phase225_newbench.py" >/dev/null || pgrep -f "diag_textvqa_crop.py" >/dev/null; do sleep 60; done
printf "%s\n" "qwen2_7b textvqa 0" "internvl3_8b textvqa 0" "qwen3_2b docvqa 2000" "qwen2_7b docvqa 2000" "llava_ov docvqa 2000" "internvl3_8b docvqa 2000" "qwen3_2b gqa 2000" "qwen2_7b gqa 2000" "llava_ov gqa 2000" "internvl3_8b gqa 2000" > /tmp/opencode/joblist2.txt
xargs -P 2 -I{} bash -c "set -- {}; python3 scripts/phase225_newbench.py \$1 \$2 \$3 > logs/par_\$1_\$2.log 2>&1" < /tmp/opencode/joblist2.txt
echo "REST GRID DONE $(date +%H:%M)" > logs/rest_done.flag
