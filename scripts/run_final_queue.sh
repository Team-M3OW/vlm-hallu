#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
while pgrep -f "run_205_queue" >/dev/null; do sleep 60; done
python3 scripts/phase205_tsr_probe.py qwen2 191
python3 scripts/phase204_endtask.py qwen2
python3 scripts/phase176_readout_sensitivity.py qwen2
python3 scripts/phase199_layersweep.py qwen2
echo "final queue done"
