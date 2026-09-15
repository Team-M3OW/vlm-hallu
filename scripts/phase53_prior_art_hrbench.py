"""
Phase 53: the prior-art head-to-head at matched realized budget, on HR-Bench 4k.
n = 800 rows / 200 instances, CircularEval, 4032x4032 images.

WHY THIS RUN
------------
Phase 47 ran this comparison on V*Bench and found that NO published policy beats the budget axis at
its own spend. Two things limit that table: n = 191, so every margin has a ~+-7pp CI and none is
individually significant; and V*Bench images are ~1500-2000px, which is not where tree search is
supposed to pay. HR-Bench 4k fixes both. It is 800 rows (4x the power), it is the benchmark Zoom Eye
and its relatives were built for, and its CircularEval protocol defeats option-position bias.

This is therefore the FAIREST possible venue for the strongest baseline. If Zoom Eye beats the
budget axis anywhere, it should be here, and the paper should say so.

ARMS (all charged for every image token across every pass)
    uniform@300 / @600 / @1200    the budget axis; these ARE the compute-matched bar, measured
                                  in-run on these items rather than borrowed from another run
    grounding_crop                ask the VLM for the region, crop to it, answer -- the
                                  inference-time mechanism of Chain-of-Spot / Visual CoT / DualFocus
    zoom_eye                      best-first tree search over quadrants scored by the model's own
                                  confidence, depth 2 -- Zoom Eye's core, training-free
    ours_attn@0.15                our fixed-policy crop at the ring-masked attention peak
    rand@0.15                     placement floor
    `peak` is logged so the ADAPTIVE policy is assembled offline at any firing rate, with the
    V*Bench-transferred rate as the headline.

NO ORACLE ARM: HR-Bench ships no bounding boxes. No oracle-capture fraction is computable and none
will be quoted, exactly as in Phase 33.

PROPOSALS ARE PER INSTANCE, ANSWERS PER ROW. The crop geometry depends on (image, question) and not
on the option permutation, so the expensive search runs 200 times and only the answer runs 800 times.
Cost is accounted the same way: per-instance search cost is divided across that instance's 4 rows.

INSTANCE KEYING USES THE IMAGE HASH. HR-Bench has 159 unique question strings across 200 instances;
keying on (question, category) alone once paired 212/800 rows with the WRONG image, silently.
"""
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase53_prior_art_hrbench.jsonl"
BLOCK = list(range(16, 27))
B0, W_OURS, PAD = 300, 0.15, 0.25
MAX_MP = 24_000_000
Image.MAX_IMAGE_PIXELS = None
BB = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
                r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rng = random.Random(53)
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
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                      for s in ["yes", "Yes", " yes", " Yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                     for s in ["no", "No", " no", " No"]})
    COST = {"p": 0, "t": 0}

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

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

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["p"] += 1; COST["t"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p_ = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
        return [round(float(v), 6) for v in p_.tolist()]

    def p_yes(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["p"] += 1; COST["t"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        return float(torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                                torch.logsumexp(lg[no_ids], 0)]), 0)[0])

    def generate(img, text, mx=48):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["p"] += 1; COST["t"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            o = model.generate(**i, max_new_tokens=mx, do_sample=False)
        return tok.decode(o[0][i["input_ids"].shape[-1]:], skip_special_tokens=True)

    def localize(img, text):
        small = fit(img, B0)
        inp = build(small, text)
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

    def zoom_eye(img, question, depth=2):
        probe = ("Does this image region contain the visual information needed to answer "
                 f"this question? {question} Answer yes or no.")
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
        for _ in range(depth):
            mx_, my = (x0 + x1) / 2, (y0 + y1) / 2
            best, bs = None, -1.0
            for q in [(x0, y0, mx_, my), (mx_, y0, x1, my), (x0, my, mx_, y1), (mx_, my, x1, y1)]:
                iw, ih = img.size
                c = img.crop((int(q[0] * iw), int(q[1] * ih), int(q[2] * iw), int(q[3] * ih)))
                if c.width < 16 or c.height < 16:
                    continue
                s = p_yes(fit(c, B0), probe)
                if s > bs:
                    bs, best = s, q
            if best is None:
                break
            x0, y0, x1, y1 = best
        iw, ih = img.size
        return img.crop((int(x0 * iw), int(y0 * ih), int(x1 * iw), int(y1 * ih)))

    def grounding(img, question):
        small = fit(img, B0)
        txt = generate(small, "Locate the region needed to answer this question and output its "
                              f"bounding box. {question}")
        m = BB.search(txt)
        if not m:
            return None, txt
        x0, y0, x1, y1 = [float(v) / 1000.0 for v in m.groups()]
        if x1 <= x0 or y1 <= y0:
            return None, txt
        dx, dy = (x1 - x0) * PAD, (y1 - y0) * PAD
        iw, ih = img.size
        b = (max(0, x0 - dx) * iw, max(0, y0 - dy) * ih,
             min(1, x1 + dx) * iw, min(1, y1 + dy) * ih)
        if b[2] - b[0] < 8 or b[3] - b[1] < 8:
            return None, txt
        return img.crop(tuple(int(v) for v in b)), txt

    def prompt(r):
        return (f"{r['question']}\n(A) {r['A']}\n(B) {r['B']}\n(C) {r['C']}\n(D) {r['D']}\n"
                "Answer with the option's letter from the given choices directly.")

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["row_id"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            rows = [r for _, r in grp.iterrows() if int(r["index"]) not in done]
            if not rows:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            q = grp["question"].iloc[0]

            # ---- per-instance proposals; their cost is split across the instance's 4 rows
            COST["p"] = 0; COST["t"] = 0
            cx, cy, peak = localize(img, q)
            loc_cost = dict(COST)
            COST["p"] = 0; COST["t"] = 0
            gimg, graw = grounding(img, q)
            gr_cost = dict(COST)
            COST["p"] = 0; COST["t"] = 0
            zimg = zoom_eye(img, q)
            ze_cost = dict(COST)

            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            imgs = {"uniform@300": fit(img, B0), "uniform@600": fit(img, 2 * B0),
                    "uniform@1200": fit(img, 4 * B0),
                    "ours_attn@0.15": fit(window(img, cx, cy, W_OURS), B0),
                    "rand@0.15": fit(window(img, rx, ry, W_OURS), B0),
                    "grounding_crop": fit(gimg if gimg is not None else img, B0),
                    "zoom_eye": fit(zimg, B0)}
            pre = {"uniform@300": (0, 0), "uniform@600": (0, 0), "uniform@1200": (0, 0),
                   "ours_attn@0.15": (loc_cost["p"], loc_cost["t"]),
                   "rand@0.15": (0, 0),
                   "grounding_crop": (gr_cost["p"], gr_cost["t"]),
                   "zoom_eye": (ze_cost["p"], ze_cost["t"])}

            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst),
                       "category": r["category"], "cycle": int(r["cycle_category"]),
                       "question": r["question"], "peak": peak,
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "img_wh": list(img.size), "ground_raw": (graw or "")[:200],
                       "probs": {}, "cost": {}}
                for nm, im in imgs.items():
                    COST["p"] = 0; COST["t"] = 0
                    rec["probs"][nm] = answer(im, text)
                    pp, pt = pre[nm]
                    rec["cost"][nm] = {"passes": COST["p"] + pp, "tokens": COST["t"] + pt}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
            if n % 40 < 4:
                el = time.time() - t0
                print(f"  [{n}/800] {n/el:.2f} rows/s eta="
                      f"{(800-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
