"""
Phase 33b: the missing control. A RANDOM-centre window of the SAME SIZE, on HR-Bench 4k.

WHAT IT DECIDES
---------------
Phase 33 found our allocator LOSES on HR-Bench (always attn@0.15 = 43.8% vs uniform 52.5%), while
simply doubling the budget WINS (uniform@600 = 59.2%). Two incompatible explanations, and Phase 33
cannot separate them because HR-Bench ships no boxes (so no oracle arm) and no random-crop arm was
run:

    (a) allocation genuinely does not pay at 4K -- budget is the better axis there; or
    (b) our LOCALIZER fails on 4032x4032 inputs, so the crops miss the evidence.
        At W=0.15 a crop is ~605px of a 4032px source: one miss loses everything.

This arm separates them, and the reading is fixed here before the run:

    attn ~= rand                      -> (b) LOCALIZER FAILURE. The attention peak carries no more
                                         information than a random point at 4K. The V*Bench
                                         localizer does not transfer; allocation is untested here.
    attn > rand, both < uniform       -> (a) ALLOCATION IS THE WRONG MOVE at this scale. The
                                         localizer still works (it beats random placement) but any
                                         crop at B0 discards more than it concentrates.
    attn > rand, attn >= uniform      -> contradicts Phase 33; investigate before believing either.

This is the same contrast that carried the V*Bench result: there, attn - rand = +22.5pp
CI [+13.6,+31.9], which is what licensed the claim that PLACEMENT (not cropping) was doing the work.
Running it here asks whether that still holds where the method fails.

DESIGN
------
Identical pipeline to Phase 33 -- same instances, same B0, same W, same fit(), same prompt, same
answer read-out -- with the crop centre drawn uniformly instead of taken from the attention peak.
Only the centre changes, so the arms differ in placement and nothing else.

The centre is drawn ONCE PER INSTANCE (not per option-permutation), matching how the attention peak
is computed once per instance, so the four CircularEval rows of an instance share a crop exactly as
they do in the attn arm. Seeded for reproducibility.

Instance keying includes the IMAGE HASH: HR-Bench has 159 unique question strings across 200
instances, and keying on (question, category) alone silently paired 212/800 rows with the WRONG
image in the first Phase 33 attempt.
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33b_rand_control.jsonl"
W_MAIN = 0.15
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
    sz = df.groupby("_inst").size()
    assert set(sz.unique()) == {4}, f"expected 4 rows/instance, got {dict(sz.value_counts())}"
    print(f"HR-Bench 4k: {len(df)} rows, {df['_inst'].nunique()} instances x 4 (hash-verified)",
          flush=True)

    print("Loading Qwen3-VL-2B (fp16)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
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

    n, t0 = 0, time.time()
    with open(OUT_PATH, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            rows = [r for _, r in grp.iterrows() if int(r["index"]) not in done]
            if not rows:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            # one centre per INSTANCE, mirroring how the attention peak is computed once per
            # instance -- so the 4 CircularEval rows share a crop exactly as in the attn arm
            rng = random.Random(9000 + int(inst))
            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            crop, _ = fit(window(img, rx, ry, W_MAIN), B0)
            for r in rows:
                p_, rz = answer(crop, prompt(r))
                fout.write(json.dumps({
                    "row_id": int(r["index"]), "instance": int(inst),
                    "category": r["category"], "cycle": int(r["cycle_category"]),
                    "label": "ABCD".index(str(r["answer"]).strip().upper()),
                    "rand_frac": [rx, ry],
                    "probs": {f"rand@{W_MAIN}": p_},
                    "realized_tokens": {f"rand@{W_MAIN}": rz}}) + "\n")
                fout.flush()
                n += 1
            if n % 80 < 4:
                el = time.time() - t0
                print(f"  [{n}/800] {n/el:.2f} rows/s eta={(800-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
