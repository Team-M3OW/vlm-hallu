"""
Phase 56: COARSE-TO-FINE localisation at 4K. The last lever with real headroom.

THE DIAGNOSIS THIS ACTS ON
--------------------------
At 4K every packaging of our proposal lands at or below break-even: crop-only -17.3pp, composite
-3.5 to -7.2pp, peak-gated -0.2pp. The common cause is PROPOSAL PRECISION, not packaging. The
localiser runs on a 300-token view of a 4032x4032 image, so ONE GRID CELL SPANS ~233px. A W=0.15
window is 605px; if the coarse peak is one cell off, the target is already near the edge, and two
cells off it is gone. Meanwhile the budget axis at 4K climbs steeply (52.8 -> 59.9 -> 64.5%), so an
imprecise proposal cannot beat a bar rising that fast, however it is wrapped.

THE FIX
-------
Localise TWICE, at increasing resolution:

    pass 1  uniform@B0 on the full 4032px image   -> answer + COARSE peak   (1 cell ~= 233px)
    pass 2  crop W_MID=0.35 around the coarse peak, refit to B0, localise again
            -> answer + FINE peak                  (1 cell ~= 1411/17 ~= 83px, 2.8x finer)
    pass 3  tight W_FINE=0.15 window around the FINE peak, refit to B0 -> answer

Each pass also produces an answer, so nothing is wasted: the confidence route can pick among all
three, and the peak gate can stop after pass 1. Costs are logged per arm so every policy can be
assembled offline and charged honestly.

WHAT WOULD FALSIFY IT: if the fine peak's window does no better than the coarse one, then the
attention map at 4K does not carry usable localisation at ANY resolution we can afford, the method
is V*Bench-only, and the paper says so. That is a real possibility -- §5 showed the attention peak
lands in the GT box only 15.7% of the time even on V*Bench.

The uniform bar arms are JOINED from Phase 53 (same items, same model, deterministic decoding)
rather than recomputed; `uniform@300` is recomputed here on every row as a consistency check and the
run asserts agreement.
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase56_coarse_to_fine.jsonl"
BLOCK = list(range(16, 27))
B0, W_MID, W_FINE = 300, 0.35, 0.15
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
    print(f"HR-Bench 4k: {df['_inst'].nunique()} instances x 4", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
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

    def localize(img, text):
        """Returns (peak_x, peak_y) in FRACTIONS OF THE GIVEN IMAGE, plus the peak value and the
        pixel size of one grid cell -- which is what 'precision' means here."""
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
        cell_px = img.size[0] / gw
        return ((j % gw) + .5) / gw, ((j // gw) + .5) / gh, float(f.max()), cell_px

    def win_box(iw, ih, cx, cy, Wn):
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return (int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8)))

    def prompt(r):
        return (f"{r['question']}\n(A) {r['A']}\n(B) {r['B']}\n(C) {r['C']}\n(D) {r['D']}\n"
                "Answer with the option's letter from the given choices directly.")

    # join the bar arms from phase53
    ref = {}
    P53 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase53_prior_art_hrbench.jsonl"
    if os.path.exists(P53):
        for l in open(P53):
            d = json.loads(l)
            ref[d["row_id"]] = d
        print(f"joined {len(ref)} rows of bar arms from phase53", flush=True)

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
            IW, IH = img.size
            q = grp["question"].iloc[0]

            COST["p"] = 0; COST["t"] = 0
            cx, cy, peak_c, cell_c = localize(img, q)
            c_cost = dict(COST)
            mid_box = win_box(IW, IH, cx, cy, W_MID)
            mid = img.crop(mid_box)
            COST["p"] = 0; COST["t"] = 0
            fx, fy, peak_f, cell_f = localize(mid, q)
            f_cost = dict(COST)
            # map the fine peak back into ORIGINAL image coordinates
            gx = (mid_box[0] + fx * (mid_box[2] - mid_box[0])) / IW
            gy = (mid_box[1] + fy * (mid_box[3] - mid_box[1])) / IH

            coarse_fine = img.crop(win_box(IW, IH, cx, cy, W_FINE))
            fine_fine = img.crop(win_box(IW, IH, gx, gy, W_FINE))
            imgs = {"uniform@300": fit(img, B0),
                    "mid@0.35": fit(mid, B0),
                    "coarse_fine@0.15": fit(coarse_fine, B0),
                    "c2f_fine@0.15": fit(fine_fine, B0)}
            pre = {"uniform@300": (0, 0),
                   "mid@0.35": (c_cost["p"], c_cost["t"]),
                   "coarse_fine@0.15": (c_cost["p"], c_cost["t"]),
                   "c2f_fine@0.15": (c_cost["p"] + f_cost["p"], c_cost["t"] + f_cost["t"])}

            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst),
                       "category": r["category"], "cycle": int(r["cycle_category"]),
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "img_wh": [IW, IH],
                       "peak_coarse": peak_c, "peak_fine": peak_f,
                       "cell_px_coarse": cell_c, "cell_px_fine": cell_f,
                       "coarse_xy": [cx, cy], "fine_xy": [gx, gy],
                       "probs": {}, "cost": {}}
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
                        same = (max(range(4), key=lambda i: rec["probs"]["uniform@300"][i])
                                == max(range(4), key=lambda i: d["probs"]["uniform@300"][i]))
                        assert same, "uniform@300 disagrees with phase53 -- join is unsafe"
                        checked += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
            if n % 40 < 4:
                el = time.time() - t0
                print(f"  [{n}/800] {n/el:.2f} rows/s eta={(800-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done ({checked} join checks passed). Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
