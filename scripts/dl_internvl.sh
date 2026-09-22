#!/bin/bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_CACHE=/media/kavinder/hdd2/hf_cache
python3 - <<'PY'
import os
for v in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v,None)
from huggingface_hub import snapshot_download
p=snapshot_download("OpenGVLab/InternVL3-8B-hf")
print("downloaded ->",p,flush=True)
PY
df -h /media/kavinder/hdd2 | tail -1
