"""
Phase 46: log the pass-1 `peak` scalar for HR-Bench so the adaptive policy can be tested where the
method currently LOSES.

WHY ONLY THIS
-------------
Phase 33 already ran every answer arm on HR-Bench (uniform, uniform2x, attn@0.15, attn@0.25) for all
800 rows. The only thing missing for the adaptive policy is the signal it routes on: `peak`, the max
of the ring-masked, L1-normalised, block-averaged attention map, which the localizer computes on its
way to the argmax. That is ONE forward pass per INSTANCE (200), not per row (800), because the
proposal depends on (image, question) and not on the option permutation.

WHAT IT DECIDES
---------------
On HR-Bench the deployed always-2-pass gate loses to its own compute-matched control:

    uniform@600 (compute-matched)   59.2%
    conf_route@0.15                 56.1%   -3.1pp
    TEXT+CONF (the gate)            53.4%   -5.9pp

That is the paper's single biggest weakness and it is a COST problem: the method pays 2x on 100% of
items on a benchmark where cropping usually hurts. An adaptive policy should decline most of the
time here, its cost should fall toward 1x, and its bar should fall toward uniform@300 (52.5%) --
which conf_route already beats. If instead it still loses, the adaptive fix does not rescue the
transfer result and the paper must say so.

Instance keying uses the IMAGE HASH, as in Phase 33: HR-Bench has 159 unique question strings across
200 instances, and keying on (question, category) alone once paired 212/800 rows with the WRONG
image.
"""
import base64
import hashlib
import io
import json
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase46_hrbench_peak.jsonl"
BLOCK = list(range(16, 27))
B0 = 300
MAX_MP = 24_000_000
Image.MAX_IMAGE_PIXELS = None


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("DreamMr/HR-Bench", "hr_bench_4k.parquet", repo_type="dataset")
    df = pd.read_parquet(p).reset_index(drop=True)
    df["_imghash"] = df["image"].map(
        lambda b: hashlib.md5(b.encode() if isinstance(b, str) else b).hexdigest())
    df["_inst"] = df.groupby(["question", "category", "_imghash"]).ngroup()
    assert set(df.groupby("_inst").size().unique()) == {4}
    print(f"HR-Bench 4k: {df['_inst'].nunique()} instances (hash-verified)", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    itid = model.config.image_token_id

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            w, h = max(32, int(W_ * sc)), max(32, int(H_ * sc))
            if w * h > MAX_MP:
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
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["instance"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            if int(inst) in done:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            q = grp["question"].iloc[0]
            small, _ = fit(img, B0)
            inp = build(small, q)
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2
            n_img = gh * gw
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            base = int(pos[0].item())
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            assert torch.isfinite(out.attentions[BLOCK[0]]).all(), "non-finite attention"
            acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
            for L in BLOCK:
                a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
                acc += a / (a.sum() + 1e-12)
            del out
            acc = (acc / len(BLOCK)).reshape(gh, gw)
            m = torch.full_like(acc, -1.0)
            if gh > 2 and gw > 2:
                m[1:-1, 1:-1] = acc[1:-1, 1:-1]
            else:
                m = acc.clone()
            f = m.flatten()
            # EXACTLY the features phase32 logged, so the V*Bench threshold is comparable
            valid = f[f >= 0]
            srt = torch.sort(valid, descending=True).values
            tot = float(valid.sum()) or 1.0
            pk = float(srt[0])
            pv = valid / tot
            ent = float(-(pv * (pv + 1e-12).log()).sum())
            fout.write(json.dumps({
                "instance": int(inst),
                "peak": pk,
                "peak_over_median": pk / max(float(valid.median()), 1e-12),
                "top1_frac": pk / tot,
                "top5_frac": float(srt[:5].sum()) / tot,
                "entropy": ent,
                "entropy_norm": ent / (len(valid) ** 0.0 + __import__("math").log(len(valid))),
                "n_img": n_img}) + "\n")
            fout.flush()
            n += 1
            torch.cuda.empty_cache()
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}/200] {n/el:.2f} it/s eta={(200-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
