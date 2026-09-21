#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "python3 scripts/phase204_endtask.py" >/dev/null; do sleep 30; done
python3 scripts/phase205_tsr_probe.py qwen3 191
python3 scripts/phase205_tsr_probe.py qwen2 191
python3 scripts/phase203_patch_lens.py qwen3 191
python3 scripts/phase203_patch_lens.py qwen2 191
echo "queue done"
