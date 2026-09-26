"""Fetch Track R assets file-by-file.
snapshot_download is UNUSABLE on this host: hub 1.7.1 + tqdm 4.70.0 make tqdm.contrib.concurrent
raise "ValueError: min() iterable argument is empty" before any file is fetched. Listing and
hf_hub_download both work fine, so we loop. Retries also absorb this host's intermittent DNS
(systemd-resolved's stub dies and recovers; permanent fix: sudo systemctl restart systemd-resolved).
We take a SUBSET of episodes (static action-prediction eval needs no more) rather than 31GB.
"""
import os, sys, time, json
os.environ.setdefault("HF_HUB_CACHE","/media/kavinder/hdd2/hf_cache"); os.environ.pop("HF_TOKEN",None)
from huggingface_hub import HfApi, hf_hub_download

NEP=int(os.environ.get("NEP","60"))
SUITES=["spatial","object"]

def get(repo, f, kind, tries=12):
    for i in range(tries):
        try: return hf_hub_download(repo, f, repo_type=kind)
        except Exception as e:
            if i==tries-1: print(f"    FAIL {f}: {type(e).__name__}",flush=True); return None
            time.sleep(min(15,2*(i+1)))

for s in SUITES:
    R=f"IPEC-COMMUNITY/libero_{s}_no_noops_1.0.0_lerobot"
    for i in range(12):
        try: fs=[x.rfilename for x in HfApi().dataset_info(R).siblings]; break
        except Exception as e: print(f"  list retry {i}: {type(e).__name__}",flush=True); time.sleep(5)
    else: print("cannot list",R); continue
    meta=[f for f in fs if f.startswith("meta/")]
    eps=sorted(f for f in fs if f.startswith("data/") and f.endswith(".parquet"))[:NEP]
    cam=sorted(f for f in fs if f.startswith("videos/") and "observation.images.image/" in f)[:NEP]
    print(f"[{s}] meta={len(meta)} episodes={len(eps)} videos={len(cam)}",flush=True)
    for f in meta+eps+cam: get(R,f,"dataset")
    print(f"DONE data {s}",flush=True)

for s in SUITES:
    M=f"openvla/openvla-7b-finetuned-libero-{s}"
    for i in range(12):
        try: fs=[x.rfilename for x in HfApi().model_info(M).siblings]; break
        except Exception as e: print(f"  list retry {i}: {type(e).__name__}",flush=True); time.sleep(5)
    else: print("cannot list",M); continue
    want=[f for f in fs if f.endswith((".json",".safetensors",".bin",".txt",".py",".model"))]
    print(f"[{s}] model files={len(want)}",flush=True)
    for f in want: get(M,f,"model")
    print(f"DONE model {s}",flush=True)
print("ALL VLA ASSETS DONE",flush=True)
