"""
Phase 57: the SCALE SWEEP. Same items, same questions, only input resolution varies.

WHY THIS AND NOT MMBENCH
------------------------
The open gap is that the method's own win is measured on V*Bench alone (n=191, +5.4pp CI
[-1.4,+12.2] -- crosses zero). The obvious fix is a second mid-resolution benchmark with more items.
MMBench was checked first and REJECTED: its images are capped at 512px (median 512, 0% above
800px), so uniform@300 already resolves them and there is no allocation headroom. A null there
would be a benchmark-selection artifact, not evidence.

Downsampling HR-Bench's 4032x4032 images is strictly better:
  * n = 800 rows / 200 instances -- 4x V*Bench's power -- at the scale where the method works;
  * the SAME items, questions and answers at every scale, so scale is isolated with everything else
    held fixed. No cross-benchmark comparison can do that: V*Bench vs HR-Bench differ in content,
    question style and annotation as well as in resolution.
  * it turns §7's scale boundary from a TWO-POINT observation into a CURVE, which is the difference
    between "we found a limit" and "we characterised a limit".

This is a resolution ablation on a public benchmark, not a constructed dataset.

SCALES: 1008px and 2016px are run here; 4032px (native) is already measured in Phase 53/56 on the
identical rows and is joined in by the analyzer. 1008 and 2016 bracket V*Bench's ~1500-2000px.

PREDICTION, fixed before the run (§7): as scale falls, two things should move together --
  (a) the budget axis becomes LESS productive (uniform@1200 minus uniform@300 shrinks), and
  (b) the localiser's grid cell shrinks in ABSOLUTE pixels, so the proposal gets more precise.
Both favour allocation, so the allocation margin should rise monotonically as scale falls, and
should cross zero somewhere between 4032px (measured: -0.2pp) and 1008px.
If the margin does NOT move with scale, §7's account is wrong and the boundary needs a different
explanation.

ARMS per scale: uniform@300 / @600 / @1200 (the bar, measured at that scale), attn@0.15, and the
`peak` scalar so the gated policy is assembled offline at the V*Bench-transferred firing rate.
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase57_scale_sweep.jsonl"
BLOCK = list(range(16, 27))
B0, W = 300, 0.15
SCALES = [1008, 2016]
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
    print(f"HR-Bench 4k: {df['_inst'].nunique()} instances x 4; scales {SCALES}", flush=True)

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
            cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC)
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
        return ((j % gw) + .5) / gw, ((j // gw) + .5) / gh, float(f.max()), img.size[0] / gw

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
            d = json.loads(l)
            done.add((d["row_id"], d["scale"]))
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    tot = 800 * len(SCALES)
    with open(OUT, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            full = None
            for S in SCALES:
                rows = [r for _, r in grp.iterrows() if (int(r["index"]), S) not in done]
                if not rows:
                    continue
                if full is None:
                    full = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
                sc = S / max(full.size)
                img = full.resize((max(32, int(full.width * sc)), max(32, int(full.height * sc))),
                                  Image.BICUBIC)
                q = grp["question"].iloc[0]
                COST["p"] = 0; COST["t"] = 0
                cx, cy, peak, cell = localize(img, q)
                loc = dict(COST)
                imgs = {"uniform@300": fit(img, B0), "uniform@600": fit(img, 2 * B0),
                        "uniform@1200": fit(img, 4 * B0),
                        "attn@0.15": fit(window(img, cx, cy, W), B0)}
                pre = {"uniform@300": (0, 0), "uniform@600": (0, 0), "uniform@1200": (0, 0),
                       "attn@0.15": (loc["p"], loc["t"])}
                for r in rows:
                    text = prompt(r)
                    rec = {"row_id": int(r["index"]), "instance": int(inst), "scale": S,
                           "category": r["category"], "cycle": int(r["cycle_category"]),
                           "label": "ABCD".index(str(r["answer"]).strip().upper()),
                           "img_wh": list(img.size), "peak": peak, "cell_px": cell,
                           "probs": {}, "cost": {}}
                    for nm, im in imgs.items():
                        COST["p"] = 0; COST["t"] = 0
                        rec["probs"][nm] = answer(im, text)
                        pp, pt = pre[nm]
                        rec["cost"][nm] = {"passes": COST["p"] + pp, "tokens": COST["t"] + pt}
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    n += 1
                if n % 60 < 4:
                    el = time.time() - t0
                    print(f"  [{n}/{tot}] {n/el:.2f} rows/s eta={(tot-n)/max(n/el,1e-9)/60:.1f}min",
                          flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
