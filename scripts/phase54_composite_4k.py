"""
Phase 54: make the method win at 4K -- SCENE + CROP COMPOSITE instead of crop-only.

WHY THE CURRENT METHOD FAILS AT 4K (Phase 53, n=800)
----------------------------------------------------
    uniform@300   52.8%      ours_attn@0.15 (crop-only)  42.6%   -17.3pp vs its bar
    uniform@1200  64.5%      rand@0.15                   35.0%   -17.8pp

A W=0.15 crop of a 4032x4032 image keeps **2.25% of the scene**. Our crop-only policy is barely
better than random placement there: when the peak is off-target -- and at 4K the localiser runs on a
300-token view where one cell spans ~233px -- everything outside the window is gone and the answer
has nothing to go on. On V*Bench the same policy works because the scene matters less and the
targets are a larger fraction of the frame.

THE FIX THE DATA ALREADY POINTS TO
----------------------------------
Phase 27 logged an arm we never built a method on: `alloc_query_2img@300` reaches **85.9% at 292
realized tokens** on V*Bench against uniform's 56.5% -- a SCENE + CROP COMPOSITE passed as two
images in ONE forward pass at a matched budget. It never discards the scene, so a bad crop costs
resolution rather than the whole question. That arm used the GT box; this file uses the ATTENTION
PEAK, making it a method.

ARMS (each = one localiser pass, which also yields the uniform@300 answer, + one composite pass)
    comp@150+150    scene at 150 tok + crop at 150 tok  -> 300 composite tokens
    comp@300+300    scene at 300 + crop at 300          -> 600 composite tokens
    comp@300+300 W=0.35                                  a larger window, since 4K peaks are coarse
    comp@150+450    scene starved, crop favoured         tests where the split should sit
Plus uniform@300 / @600 / @1200 measured in-run as the bar.

COST. The localiser pass is charged in full, and it doubles as the uniform@300 answer, so a
composite arm costs 300 + (composite tokens). The peak gate is applied OFFLINE afterwards at the
V*Bench-transferred firing rate, which pays the composite only when it fires.

WHAT WOULD FALSIFY THIS: if the composite also loses to uniform at matched tokens, then at 4K there
is no allocation policy that beats spending the budget, our method is V*Bench-only, and the paper
must say so plainly.
"""
import base64
import hashlib
import io
import json
import os
import random
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase54_composite_4k.jsonl"
BLOCK = list(range(16, 27))
B0 = 300
MAX_MP = 24_000_000
CONNECTOR = "Here is a zoomed-in crop of a region from the same image."
Image.MAX_IMAGE_PIXELS = None


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rng = random.Random(54)
    p = hf_hub_download("DreamMr/HR-Bench", "hr_bench_4k.parquet", repo_type="dataset")
    df = pd.read_parquet(p).reset_index(drop=True)
    df["_imghash"] = df["image"].map(
        lambda b: hashlib.md5(b.encode() if isinstance(b, str) else b).hexdigest())
    df["_inst"] = df.groupby(["question", "category", "_imghash"]).ngroup()
    assert set(df.groupby("_inst").size().unique()) == {4}
    print(f"HR-Bench 4k: {df['_inst'].nunique()} instances x 4 (hash-verified)", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    COST = {"p": 0, "t": 0}

    def chat(imgs, text):
        if len(imgs) == 1:
            c = [{"type": "image"}]
        else:
            c = [{"type": "image"}, {"type": "text", "text": CONNECTOR}, {"type": "image"}]
        c.append({"type": "text", "text": text})
        return pr.apply_chat_template([{"role": "user", "content": c}],
                                      tokenize=False, add_generation_prompt=True)

    def build(imgs, text):
        return pr(images=imgs, text=chat(imgs, text), return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build([img], "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.05):
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
        return best[0] if best else img

    def answer(imgs, text):
        i = build(imgs, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["p"] += 1; COST["t"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p_ = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
        return [round(float(v), 6) for v in p_.tolist()], rz

    def localize(img, text):
        small = fit(img, B0)
        inp = build([small], text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n_img = gh * gw
        base = int((inp["input_ids"][0] == itid).nonzero().flatten()[0].item())
        COST["p"] += 1; COST["t"] += n_img
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        torch.cuda.empty_cache()
        acc = (acc / len(BLOCK)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            m = acc.clone()
        f = m.flatten()
        j = int(f.argmax().item())
        return ((j % gw) + .5) / gw, ((j // gw) + .5) / gh, float(f.max())

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    def prompt(r):
        return (f"{r['question']}\n(A) {r['A']}\n(B) {r['B']}\n(C) {r['C']}\n(D) {r['D']}\n"
                "Answer with the option's letter from the given choices directly.")

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["row_id"])
    print(f"resuming: {len(done)}", flush=True)

    CONFIGS = [("comp_150_150_w15", 150, 150, 0.15),
               ("comp_300_300_w15", 300, 300, 0.15),
               ("comp_300_300_w35", 300, 300, 0.35),
               ("comp_150_450_w15", 150, 450, 0.15)]
    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            rows = [r for _, r in grp.iterrows() if int(r["index"]) not in done]
            if not rows:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            q = grp["question"].iloc[0]
            COST["p"] = 0; COST["t"] = 0
            cx, cy, peak = localize(img, q)
            loc = dict(COST)

            pre = {"uniform@300": (0, 0), "uniform@600": (0, 0), "uniform@1200": (0, 0)}
            imgs = {"uniform@300": [fit(img, B0)], "uniform@600": [fit(img, 2 * B0)],
                    "uniform@1200": [fit(img, 4 * B0)]}
            for nm, bs, bc, Wn in CONFIGS:
                imgs[nm] = [fit(img, bs), fit(window(img, cx, cy, Wn), bc)]
                pre[nm] = (loc["p"], loc["t"])

            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst),
                       "category": r["category"], "cycle": int(r["cycle_category"]),
                       "peak": peak, "img_wh": list(img.size),
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "probs": {}, "cost": {}}
                for nm, im in imgs.items():
                    COST["p"] = 0; COST["t"] = 0
                    p_, _ = answer(im, text)
                    pp, pt = pre[nm]
                    rec["probs"][nm] = p_
                    rec["cost"][nm] = {"passes": COST["p"] + pp, "tokens": COST["t"] + pt}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
            if n % 40 < 4:
                el = time.time() - t0
                print(f"  [{n}/800] {n/el:.2f} rows/s eta={(800-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
