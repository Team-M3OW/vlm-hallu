"""
Phase 33: does the conditional allocator TRANSFER? HR-Bench 4k, nothing refitted.

WHY THIS IS THE EXPERIMENT THE PAPER NEEDS
------------------------------------------
Phase 32's gate reaches 68.6% on V*Bench and clears the compute-matched bar. Its stated main threat
(§6.2) is that the relational keyword rule **agrees with the V*Bench `category` annotation on
189/191 items (99%)**. On a benchmark whose two categories are almost perfectly keyword-separable,
the gate may be doing what the annotation would do, and nothing shows it generalises.

HR-Bench 4k breaks that confound, verified offline BEFORE any GPU was spent:

    keyword gate vs `category` concordance:   53.5%   (V*Bench: 99.0%)  -- i.e. chance
    gate fires on:                            22.5%   (V*Bench: 39.8%)

So here the rule cannot be the annotation in disguise. It also exposes a failure mode V*Bench never
contained: HR-Bench asks things like *"What is the number displayed ABOVE the entrance where the
woman is standing?"* -- an ATTRIBUTE question that uses a spatial preposition to LOCATE the target.
Cropping should help there, and the keyword rule will wrongly skip it. If the gate transfers anyway,
that is real evidence; if it fails, this is the mechanism and it is worth reporting.

NOTHING IS REFITTED. THAT IS THE POINT.
---------------------------------------
    W = 0.15                  transferred from V*Bench 5-fold CV (chosen there in every fold)
    gate rules                identical: (1) relational question -> do not reallocate
                                         (2) else answer from whichever PASS is more confident
    layer block 16-26, outer-ring mask, B0=300   all unchanged
Refitting any of these on HR-Bench would answer a different and much weaker question.
`attn@0.25` is logged as a secondary window, NOT as an alternative headline.

BARS
----
    uniform@300     baseline
    uniform@600     COMPUTE-MATCHED -- the gate spends two passes, so this is the honest floor
There is **no oracle arm**: HR-Bench ships no bounding-box annotations (columns are index, answer,
question, A-D, category, cycle_category, image). So no oracle-capture fraction is computable here,
and none will be quoted.

CIRCULAR EVAL
-------------
800 rows = 200 unique (question,image) instances x 4 option permutations, with the answer tracking
the rotation. That is HR-Bench's CircularEval protocol, built to defeat option-position bias. Both
metrics are reported: per-row accuracy, and the strict circular score (an instance counts only if
all 4 permutations are answered correctly). The strict score is the benchmark's intended metric.

Proposal geometry depends only on (image, question), never on the option permutation, so the
expensive attention pass runs once per instance (200) rather than once per row (800).
"""
import base64
import io
import json
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33_hrbench_transfer.jsonl"
BLOCK = list(range(16, 27))
W_MAIN = 0.15          # transferred from V*Bench CV -- NOT refit here
W_ALT = 0.25           # secondary, logged only
B0 = 300
MAX_MP = 24_000_000
Image.MAX_IMAGE_PIXELS = None


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("DreamMr/HR-Bench", "hr_bench_4k.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    print(f"HR-Bench 4k: {len(df)} rows", flush=True)

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

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p_ = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in p_.tolist()], rz

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
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            m = acc.clone()
        i = int(m.argmax().item())
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

    def prompt(r):
        return (f"{r['question']}\n(A) {r['A']}\n(B) {r['B']}\n(C) {r['C']}\n(D) {r['D']}\n"
                "Answer with the option's letter from the given choices directly.")

    done = set()
    if os.path.exists(OUT_PATH):
        for l in open(OUT_PATH):
            done.add(json.loads(l)["row_id"])
    print(f"resuming: {len(done)}", flush=True)

    # group by instance so the attention pass and the fitted crops are computed ONCE per image
    df = df.reset_index(drop=True)
    # GROUP BY IMAGE CONTENT TOO. HR-Bench has only 159 unique question STRINGS across 200
    # instances, so 12 questions recur with DIFFERENT images. Grouping on (question, category)
    # merged those, and since the loop takes grp["image"].iloc[0] for the whole group, 212/800
    # rows (26.5%) were answered against the WRONG IMAGE -- silently, with plausible outputs.
    # Hashing the image bytes makes the instance key exact: 200 groups x 4 cycles, verified.
    import hashlib
    df["_imghash"] = df["image"].map(
        lambda b: hashlib.md5(b.encode() if isinstance(b, str) else b).hexdigest())
    df["_inst"] = df.groupby(["question", "category", "_imghash"]).ngroup()
    _sz = df.groupby("_inst").size()
    assert set(_sz.unique()) == {4}, f"expected 4 rows per instance, got {dict(_sz.value_counts())}"
    print(f"  {df['_inst'].nunique()} instances x 4 cycles, image-hash verified", flush=True)
    n, t0 = 0, time.time()
    with open(OUT_PATH, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            rows = [r for _, r in grp.iterrows() if int(r["index"]) not in done]
            if not rows:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            base_img, _ = fit(img, B0)
            big_img, _ = fit(img, 2 * B0)
            cx, cy = localize(img, rows[0]["question"])   # question text only; options irrelevant
            crops = {W: fit(window(img, cx, cy, W), B0)[0] for W in (W_MAIN, W_ALT)}
            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst),
                       "category": r["category"], "cycle": int(r["cycle_category"]),
                       "question": r["question"],
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "img_wh": list(img.size), "peak_frac": [cx, cy],
                       "probs": {}, "realized_tokens": {}}
                for nm, im in [("uniform", base_img), ("uniform2x", big_img),
                               (f"attn@{W_MAIN}", crops[W_MAIN]), (f"attn@{W_ALT}", crops[W_ALT])]:
                    p_, rz = answer(im, text)
                    rec["probs"][nm] = p_
                    rec["realized_tokens"][nm] = rz
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n += 1
            if n % 40 < 4:
                el = time.time() - t0
                print(f"  [{n}/800] {n/el:.2f} rows/s eta={(800-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
