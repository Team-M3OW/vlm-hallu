"""
Phase 32: THE CONDITIONAL ALLOCATOR. Decide *whether* to reallocate, not *where*.

WHY
---
Phase 31's fixed policy is a pre-registered NEGATIVE: 60.2% held-out against a compute-matched bar
of 66.0%. But the localizer is not what failed (+22.5pp over a same-size random window,
CI [+13.6,+31.9]). What failed is applying allocation uniformly to items where it helps AND items
where it hurts:

    direct_attributes  +11.3pp [+1.7,+20.9]      relative_position  -7.9pp [-21.1,+5.3]
    smallest GT quart. +19.1pp [+2.1,+36.2]      largest GT quart. -10.4pp [-27.1,+6.2]

Mechanism (§4X, and the same one that voided region packing in §4R): cropping around a single peak
discards inter-object layout. Relational questions need it; attribute questions do not.

THE CEILING, computed offline before writing this (so it is a decision, not a hope)
-----------------------------------------------------------------------------------
    uniform                                       56.5%
    attn, fixed policy                            60.2%
    max(uniform, attn) per item -- PERFECT GATE   74.3%   <- above BOTH bars
    oracle crop                                   93.2%

The two arms agree on 130/191 items, so a gate can only ever act on the **61 disagreements**
(27 where uniform alone is right, 34 where attn alone is right). The fixed policy already takes
34/61. Reaching 66.0% needs ~45/61 (74% gate accuracy); reaching 72.9% needs ~58/61 (95%). The
ceiling clears the bars; the question is purely whether a realisable gate gets close enough.

WHAT THIS SCRIPT ADDS
---------------------
Phase 31 stored only the argmax, which makes the most natural gate impossible to evaluate. Here we
store the full 4-way probability vector for **every** arm, plus the question text, so that offline
we can test — with proper cross-validation — gates that use NO oracle information:

    conf_route   take the answer from whichever PASS is more confident (max prob).
                 Training-free, no gate model, no threshold fitted on the test set. This is the
                 strongest thing that costs nothing and it is the honest first baseline.
    conf_thresh  reallocate only when pass-1 confidence is BELOW a threshold (fitted by CV).
    text_gate    skip allocation on relational questions, detected from the question string by
                 keyword. Legitimate: the question is an input, not an annotation.
    attn_gate    gate on statistics of the attention map itself (peak height, concentration,
                 spread) -- available from pass 1 at no extra cost.
    learned      logistic regression over the above features, evaluated by cross-validation.

FORBIDDEN FEATURES, stated so they cannot creep in: the GT box, its size, and the V*Bench
`category` label. Category is an annotation; using it would reproduce §4X's post-hoc split as if it
were a method. The text gate must derive relationality from the QUESTION STRING alone.
"""
import json
import os
import random
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
BLOCK = list(range(16, 27))
WINDOWS = [0.15, 0.25, 0.35, 0.50]
B0 = 300
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(32)
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
        """Returns the FULL probability vector, not just the argmax -- that omission is what made
        Phase 31's data unable to test a confidence gate."""
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    def localize(img, text):
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
        acc = (acc / len(BLOCK)).reshape(gh, gw)
        masked = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            masked[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            masked = acc.clone()
        i = int(masked.argmax().item())
        flat = acc.flatten()
        srt, _ = torch.sort(flat, descending=True)
        s = flat.sum() + 1e-12
        pnorm = flat / s
        ent = float(-(pnorm * (pnorm + 1e-12).log()).sum())
        # gate features from the attention map -- all free, all available at inference
        feats = {
            "peak": float(masked.max()),
            "peak_over_median": float(masked.max() / (flat.median() + 1e-12)),
            "top1_frac": float(srt[0] / s),
            "top5_frac": float(srt[:5].sum() / s),
            "entropy": ent,
            "entropy_norm": ent / (torch.tensor(float(n_img)).log().item()),
            "n_img": n_img,
        }
        return ((i % gw) + .5) / gw, ((i // gw) + .5) / gh, feats

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
                   "question": text, "img_wh": [W_, H_],
                   "gt_area_frac": ((gx1 - gx0) * (gy1 - gy0)) / (W_ * H_),
                   "probs": {}, "realized_tokens": {}}

            f, _ = fit(img, B0)
            p, rz = answer(f, text)
            rec["probs"]["uniform"] = p; rec["realized_tokens"]["uniform"] = rz

            f, _ = fit(img.crop(tuple(int(v) for v in ob)), B0)
            p, rz = answer(f, text)
            rec["probs"]["oracle"] = p; rec["realized_tokens"]["oracle"] = rz

            cx, cy, feats = localize(img, text)
            rec["peak_frac"] = [cx, cy]
            rec["attn_feats"] = feats
            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            for Wn in WINDOWS:
                f, _ = fit(window(img, cx, cy, Wn), B0)
                p, rz = answer(f, text)
                rec["probs"][f"attn@{Wn}"] = p; rec["realized_tokens"][f"attn@{Wn}"] = rz
                f, _ = fit(window(img, rx, ry, Wn), B0)
                p, rz = answer(f, text)
                rec["probs"][f"rand@{Wn}"] = p; rec["realized_tokens"][f"rand@{Wn}"] = rz
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
