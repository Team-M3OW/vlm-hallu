"""
Phase 30c: dump the FULL attention map per item, once, so every remaining read-out question can be
answered offline with zero further GPU.

WHY
---
30a killed the min/max bbox read-out (median area 1.000, n=191). 30b then showed the signal is
nonetheless present -- the GT-centre cell ranks at the 12th percentile of ~294 tokens against an
exact chance of 50th -- but that the argmax is NEVER the target (0/143, chance ~1/1733), because
the top ranks are held by attention sinks. Pooling the selected positions onto a normalised 7x7
grid shows those sinks are **positionally stable and sit at the image corners**:

    (0,6) top-right 5.1-6.3x expected, (0,0) top-left 4.4-5.7x, (6,6) bottom-right 3.8-4.2x
    the five most popular positions absorb 41-45% of all selected tokens (uniform would be 10.2%)

A content-independent, position-fixed background is **subtractable**. That is a training-free fix
and it must be tested before any learned read-out is justified. But 30b logged only the top-k
COORDINATES, so the background cannot be estimated from it -- normalising needs the attention
VALUES at every cell.

So: dump the whole map. 294 floats x 3 layers per item is trivial to store, and it makes every
candidate read-out (background-normalised peak, z-score per position, top-k after subtraction,
connected components, centroid of the residual) an offline experiment on a laptop rather than a
GPU run. This is the last forward pass this line of work should need.

NOTE ON LEAKAGE, since the background is estimated from the same 191 items: any read-out tuned on
a dataset-wide background is using the test set to build its normaliser. That is acceptable for
establishing WHETHER a fix exists, and it must be reported that way -- and if it works, the
background has to be re-estimated leave-one-out (trivial offline) before any accuracy number is
quoted. Recorded here so the caveat travels with the data.
"""
import json
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"
LAYERS = list(range(28))       # ALL layers. The 5-layer sample found L20 far better than
                               # FastV's L2 (gt_pct 0.068 vs 0.289, argmax-in-GT 6.8% vs 0.0%)
                               # but 20 was the LAST one sampled, so the peak may lie beyond it.
                               # 28 layers is ~17 MB of floats -- no reason to subsample.
B0 = 300
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print("Loading Qwen3-VL-2B (fp16, eager attention)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    img_tok_id = model.config.image_token_id
    nlayers = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") \
        else model.config.num_hidden_layers
    layers = [L for L in LAYERS if L < nlayers]
    print(f"  model has {nlayers} text layers; dumping {layers}", flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W, H = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            w, h = max(32, int(W * sc)), max(32, int(H * sc))
            if w * h > 24_000_000:
                break
            cur = img.resize((w, h), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best if best else (img, measure(img))

    done = set()
    if os.path.exists(OUT_PATH):
        for l in open(OUT_PATH):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT_PATH, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            gx0 = min(b[0] for b in ann["bbox"]) / W
            gy0 = min(b[1] for b in ann["bbox"]) / H
            gx1 = max(b[0] + b[2] for b in ann["bbox"]) / W
            gy1 = max(b[1] + b[3] for b in ann["bbox"]) / H

            small, realized = fit(img, B0)
            inp = build(small, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2
            n_img = gh * gw
            ids = inp["input_ids"][0]
            pos = (ids == img_tok_id).nonzero().flatten()
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            base = int(pos[0].item())

            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": ex["label"], "img_wh": [W, H], "realized_tokens": realized,
                   "grid": [gh, gw], "n_img_tokens": n_img,
                   "gt_box_frac": [gx0, gy0, gx1, gy1],
                   "gt_area_frac": (gx1 - gx0) * (gy1 - gy0),
                   "gt_tokens": (gx1 - gx0) * gw * (gy1 - gy0) * gh,
                   "attn": {}}
            if len(pos) != n_img:
                rec["note"] = f"token count mismatch {len(pos)} vs {n_img}"
            for L in layers:
                a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
                # store as float, rounded -- full precision is pointless for a rank/threshold study
                rec["attn"][f"L{L}"] = [round(float(v), 8) for v in a.tolist()]
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            del out
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
