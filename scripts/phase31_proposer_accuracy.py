"""
Phase 31: THE ACCURACY ARM. Does the Phase 30 localizer convert into task accuracy?

Everything in Phase 30a-30d measured localization quality. This is the first experiment in that line
that answers a question about accuracy, and therefore the first that can be scored against the
pre-registered bar.

THE METHOD (training-free, data-free, budget-preserving)
--------------------------------------------------------
    pass 1    render the whole image at budget B0=300, one forward pass
    localize  mean the attention (final prompt position -> image tokens) over layers **16-26**,
              each layer L1-normalised first; mask the outermost ring of cells; take the argmax
    propose   a square window of side W centred on that cell, clipped to the image
    pass 2    re-render THAT WINDOW at the SAME realized budget B0, answer from it

No training. No auxiliary model. No dataset: the layer block is chosen by the regime boundary
measured in 4W (L0-L15 vs L16-L26 vs L27), not by outcome, and ring-masking is an architectural
prior about where sinks live, which performs the same as a leave-one-out estimated background
(4W: 0.037 vs 0.034 median gt_pct) while requiring no other images at all.

WHY W IS CHOSEN BY CROSS-CATEGORY HOLDOUT, NOT BY A SWEEP
----------------------------------------------------------
W trades containment against magnification and the two move in opposite directions (measured
offline: side 0.15 -> 32.5% containment at 44x magnification; 0.50 -> 57.1% at 4x). Picking the
best W after seeing accuracies on all 191 items would be fitting a hyperparameter on the test set --
the same error as naming L17 the "best layer" because it won an argmin over 28 noisy statistics.

V*Bench has two disjoint categories, so the holdout is free and needs no extra data:
    choose W on `direct_attributes` (115) -> report on `relative_position` (76)
    choose W on `relative_position` (76)  -> report on `direct_attributes` (115)
The headline is the **pooled held-out accuracy**: every item scored under a W selected without
seeing that item's category. The full W sweep is reported too, as sensitivity, clearly marked as
in-sample.

ARMS (all at matched realized tokens; budget gate voids any contrast with >10% spread)
--------------------------------------------------------------------------------------
    uniform@B0        whole image, one pass         -- baseline, re-run in this harness not
                                                       imported, so the gate applies to it too
    attn_prop@B0/W    the method, for each W        -- two passes
    rand_prop@B0/W    same window size, RANDOM centre -- the placement null. If the method only
                                                       matches this, the localizer contributed
                                                       nothing and the gain was magnification.
    oracle@B0         GT box + 25% pad              -- the ceiling, for the capture fraction

PRE-REGISTERED BARS (from PLAN.md, fixed before any of this was written)
------------------------------------------------------------------------
    capture = (attn_prop - uniform@300) / (93.7 - 56.5)
    > 56.5%  beats doing nothing
    > 66.0%  beats uniform@600, i.e. EARNS ITS SECOND FORWARD PASS   <- the honest floor
    > 72.9%  beats the published training-free proposer (44% oracle capture)  <- the bar

A result above 56.5% but below 66.0% is a NEGATIVE and will be reported as one: the second pass
would have been better spent on plain resolution.
"""
import json
import os
import random
import sys
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase31_proposer_accuracy.jsonl"
BLOCK = list(range(16, 27))          # the regime from 4W, chosen by boundary not by outcome
WINDOWS = [0.15, 0.25, 0.35, 0.50]
B0 = 300
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(31)
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print("Loading Qwen3-VL-2B (fp16, eager attention)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    img_tok_id = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]

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

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return int(p.argmax().item()), rz

    def localize(img, text):
        """One forward pass at B0; returns the proposed centre in fractional image coords."""
        small, _ = fit(img, B0)
        inp = build(small, text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n_img = gh * gw
        pos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
        base = int(pos[0].item())
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        acc = acc.reshape(gh, gw).clone()
        if gh > 2 and gw > 2:                   # mask the outer ring: that is where sinks live
            m = torch.full_like(acc, -1.0)
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
            acc = m
        i = int(acc.argmax().item())
        return ((i % gw) + .5) / gw, ((i // gw) + .5) / gh

    def window(img, cx, cy, W):
        iw, ih = img.size
        x0, y0 = (cx - W / 2) * iw, (cy - W / 2) * ih
        x1, y1 = (cx + W / 2) * iw, (cy + W / 2) * ih
        if x0 < 0: x0, x1 = 0, W * iw
        if y0 < 0: y0, y1 = 0, W * ih
        if x1 > iw: x0, x1 = iw - W * iw, iw
        if y1 > ih: y0, y1 = ih - W * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

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
            W_, H_ = img.size
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            dx, dy = (gx1 - gx0) * PAD, (gy1 - gy0) * PAD
            ob = (max(0, gx0 - dx), max(0, gy0 - dy), min(W_, gx1 + dx), min(H_, gy1 + dy))
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]

            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "img_wh": [W_, H_],
                   "gt_box_frac": [gx0 / W_, gy0 / H_, gx1 / W_, gy1 / H_],
                   "gt_area_frac": ((gx1 - gx0) * (gy1 - gy0)) / (W_ * H_),
                   "pred": {}, "realized_tokens": {}}

            f, _ = fit(img, B0)
            p, rz = answer(f, text); rec["pred"]["uniform"] = p; rec["realized_tokens"]["uniform"] = rz

            f, _ = fit(img.crop(tuple(int(v) for v in ob)), B0)
            p, rz = answer(f, text); rec["pred"]["oracle"] = p; rec["realized_tokens"]["oracle"] = rz

            cx, cy = localize(img, text)
            rec["peak_frac"] = [cx, cy]
            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            rec["rand_frac"] = [rx, ry]
            for Wn in WINDOWS:
                f, _ = fit(window(img, cx, cy, Wn), B0)
                p, rz = answer(f, text)
                rec["pred"][f"attn@{Wn}"] = p; rec["realized_tokens"][f"attn@{Wn}"] = rz
                f, _ = fit(window(img, rx, ry, Wn), B0)
                p, rz = answer(f, text)
                rec["pred"][f"rand@{Wn}"] = p; rec["realized_tokens"][f"rand@{Wn}"] = rz
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
