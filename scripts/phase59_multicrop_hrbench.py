"""
Phase 59: does MULTI-CROP transfer to 4K? HR-Bench `single` (the qualifying regime).

WHAT IS BEING TESTED
--------------------
Phase 58 showed on V*Bench that handing the model k candidate windows in ONE pass beats handing it
one, at the same budget: multi4 - top1 = +7.9pp [+1.0,+14.7], and on single-region questions multi4
beats the budget axis by +9.5pp [+0.8,+17.4]. The open question is whether that is a V*Bench result
or the method.

HR-Bench `single` is the right venue: 4032x4032 images, single-region questions (so §6D's coverage
condition is satisfied), 400 rows / 100 instances, CircularEval. It is also where EVERY previous
version of our method failed -- top1 crop-only is -17.3pp there, and four separate fixes reached at
best break-even. If multi-crop clears the bar here, the method transfers; if not, the V*Bench result
is scale-bound and the paper says so.

WHY MULTI-CROP MIGHT SUCCEED WHERE THE OTHERS FAILED AT 4K
----------------------------------------------------------
The 4K failure was diagnosed as proposal imprecision: the localiser's cell spans ~233px, so a single
W=0.15 window frequently misses, and at 4K a miss discards 97.75% of the scene. Multi-crop does not
need the top-1 window to be right -- it needs the target to be inside ANY of k windows. That is
exactly the failure mode, addressed without extra passes.

COST. Pass 1 (localise) also yields the uniform@300 answer, so its tokens are charged once; the
multi-crop pass adds its own. The uniform bar arms are JOINED from Phase 53 (same rows, same model,
deterministic) and `uniform@300` is recomputed here as a consistency check, asserted on the first 20.
The bar is computed on the `single` subset itself.
"""
import base64
import hashlib
import io
import json
import math
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase59_multicrop_hrbench.jsonl"
P53 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase53_prior_art_hrbench.jsonl"
BLOCK = list(range(16, 27))
B0, W, MIN_SEP = 300, 0.15, 0.20
MAX_MP = 24_000_000
CONN = "Here is another zoomed-in crop from the same image."
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
    df = df[df["category"] == "single"].copy()
    print(f"HR-Bench 4k `single`: {len(df)} rows / {df['_inst'].nunique()} instances", flush=True)

    ref = {}
    if os.path.exists(P53):
        for l in open(P53):
            d = json.loads(l)
            ref[d["row_id"]] = d
        print(f"joined {len(ref)} rows of bar arms from phase53", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    COST = {"p": 0, "t": 0}

    def chat(k, text):
        c = [{"type": "image"}]
        for _ in range(k - 1):
            c += [{"type": "text", "text": CONN}, {"type": "image"}]
        c.append({"type": "text", "text": text})
        return pr.apply_chat_template([{"role": "user", "content": c}],
                                      tokenize=False, add_generation_prompt=True)

    def build(imgs, text):
        return pr(images=imgs, text=chat(len(imgs), text), return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build([img], "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=6, tol=0.06):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            w, h = max(28, int(W_ * sc)), max(28, int(H_ * sc))
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
        return [round(float(v), 6) for v in p_.tolist()]

    def candidates(img, text, k):
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
        pts = []
        for i in torch.argsort(f, descending=True).tolist():
            if f[i].item() < 0:
                break
            px, py = ((i % gw) + .5) / gw, ((i // gw) + .5) / gh
            if all(math.hypot(px - x, py - y) >= MIN_SEP for x, y in pts):
                pts.append((px, py))
            if len(pts) == k:
                break
        while len(pts) < k:
            pts.append(pts[0])
        return pts, float(f.max())

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

    n, t0, checked = 0, time.time(), 0
    with open(OUT, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            rows = [r for _, r in grp.iterrows() if int(r["index"]) not in done]
            if not rows:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            q = grp["question"].iloc[0]
            COST["p"] = 0; COST["t"] = 0
            pts, peak = candidates(img, q, 4)
            loc = dict(COST)
            imgs = {"uniform@300": [fit(img, B0)],
                    "top1@0.15": [fit(window(img, *pts[0], W), B0)]}
            for k in (2, 3, 4):
                imgs[f"multi{k}"] = [fit(window(img, *pts[i], W), B0 // k) for i in range(k)]
            imgs["multi4_600"] = [fit(window(img, *pts[i], W), 2 * B0 // 4) for i in range(4)]
            pre = {"uniform@300": (0, 0)}
            for a in imgs:
                if a != "uniform@300":
                    pre[a] = (loc["p"], loc["t"])

            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst), "category": "single",
                       "cycle": int(r["cycle_category"]), "peak": peak,
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "img_wh": list(img.size), "probs": {}, "cost": {}}
                for nm, im in imgs.items():
                    COST["p"] = 0; COST["t"] = 0
                    rec["probs"][nm] = answer(im, text)
                    pp, pt = pre[nm]
                    rec["cost"][nm] = {"passes": COST["p"] + pp, "tokens": COST["t"] + pt}
                d = ref.get(int(r["index"]))
                if d:
                    for a in ("uniform@600", "uniform@1200"):
                        rec["probs"][a] = d["probs"][a]
                        rec["cost"][a] = d["cost"][a]
                    if checked < 20:
                        assert (max(range(4), key=lambda i: rec["probs"]["uniform@300"][i])
                                == max(range(4), key=lambda i: d["probs"]["uniform@300"][i])), \
                            "uniform@300 disagrees with phase53 -- join unsafe"
                        checked += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
            if n % 40 < 4:
                el = time.time() - t0
                print(f"  [{n}/400] {n/el:.2f} rows/s eta={(400-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done ({checked} join checks passed). Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
