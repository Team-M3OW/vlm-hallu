#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
F=data/phase203_patchlens_qwen3.jsonl
for i in $(seq 1 300); do
  n=$(wc -l < "$F" 2>/dev/null || echo 0)
  if [ "$n" -ge 60 ]; then echo "qwen3 done at $n; starting qwen2"; exec python3 scripts/phase203_patch_lens.py qwen2 60; fi
  if ! python3 - <<'PY'
import os,sys
alive=any('phase203_patch_lens.py' in open(f'/proc/{p}/cmdline','rb').read().decode(errors='ignore')
          for p in os.listdir('/proc') if p.isdigit() and os.path.isdir(f'/proc/{p}'))
sys.exit(0 if alive else 1)
PY
  then echo "qwen3 process gone at $n rows"; exit 1; fi
  sleep 45
done
