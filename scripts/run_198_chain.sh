#!/bin/bash
# Wait for the qwen3 leg to reach a full 191 items, then start qwen2.
# Gated on the OUTPUT FILE's line count, not a log string: a crashed run also writes to the log.
cd /home/kavinder/ARNABI_ARSH/vlm-hallu
F=data/phase198_stack_qwen3.jsonl
for i in $(seq 1 240); do
  n=$(wc -l < "$F" 2>/dev/null || echo 0)
  if [ "$n" -ge 191 ]; then
    echo "qwen3 complete at $n items; starting qwen2" 
    exec python3 scripts/phase198_stack.py qwen2
  fi
  # bail out if the qwen3 process died before finishing
  if ! python3 - <<'PY'
import os,sys
alive=any(os.path.isdir(f"/proc/{p}") and "phase198_stack.py" in open(f"/proc/{p}/cmdline","rb").read().decode(errors="ignore")
          for p in os.listdir("/proc") if p.isdigit())
sys.exit(0 if alive else 1)
PY
  then echo "qwen3 process gone at $n items -- not starting qwen2"; exit 1; fi
  sleep 30
done
echo "timed out waiting for qwen3"
