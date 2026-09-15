"""
Phase 81: RANK-ONLY ALLOCATION. The method the negatives point to.

THE PRINCIPLE, EARNED FROM FOUR FAILURES
----------------------------------------
Every attempt to REGRESS a continuous geometric quantity from internals has failed:
    per-item token budget from target size (SS14A)   -- loses to flat uniform even with a PERFECT
                                                        size oracle, 11 of 12 cells
    per-item budget from any free signal (SS14B)     -- exact-rung accuracy 9.4% vs a 35.6%
                                                        majority baseline, i.e. worse than a constant
    window size from target size (phase 78)          -- best W is 0.15 in ALL FOUR size quartiles
                                                        across a 100x area range
    attention-space reallocation (SS10B)             -- oracle-targeted recovers 16% of the crop
Meanwhile RANKING cells works on two architectures (SS14C/SS80a: 39.3->52.9%, 35.1->44.0%).

So: build a method that ONLY RANKS. No size prediction, no budget prediction, no per-item geometry.
Take the head's top-k separated cells, split a FIXED budget evenly, and show them all in ONE pass.

WHY k=4, PRICED BEFORE THE RUN
------------------------------
Union coverage of the head's top-k rises with k, but each crop gets B0/k tokens, so magnification
falls and covered targets can drop back under the SS13B cliff. Both were computed offline:

    k=1  300 tok/crop   covered 52.9%   covered AND above cliff 52.9%
    k=3  100            62.3%           60.2%
    k=4   75            66.0%           63.4%   <- optimum
    k=5   60            67.0%           62.8%   <- turns over

**+10.5pp of ANSWERABLE items at the identical total budget.** The cliff costs only 2.6pp at k=4
because V*Bench targets are tiny: a 75-token crop of a W=0.15 window still puts ~5 tokens on target
against a 0.2 threshold.

PREDICTION, fixed before the run: at SS6D's exchange rate (~52.7pp per coverage flip),
+10.5pp of coverage predicts **~+5.5pp over single-crop DCR**.

ARMS (tokens measured, never computed)
    uniform@300                  vanilla
    uniform@600                  COMPUTE-MATCHED BAR (this method spends localise + answer)
    dcr_single@300               single crop at the head's argmax          (SS14D, the incumbent)
    dcr_multi_k4                 4 crops of 75 tok at the head's top-4     <- THE METHOD
    rand_multi_k4                4 crops of 75 tok at random centres       <- CONTROL
    argmax_multi_k4              4 crops at the DEPLOYED map's top-4       <- isolates the head
    oracle@300                   GT-box crop                               ceiling

The rand and argmax multi arms are what separate "multi-crop helps" from "OUR RANKING helps".
Proposals are read from phase71a (out-of-fold); no fitting happens here.
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MAPS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"
SCORES = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase81_head_scores.json"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase81_rank_only.jsonl"
B0, W, K, MIN_SEP = 300, 0.15, 4, 0.20
CONN = "\n"
Image.MAX_IMAGE_PIXELS = None


def topk_cells(score, gh, gw, k):
    rm = np.zeros((gh, gw), dtype=bool)
    if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
    else: rm[:] = True
    s = np.where(rm.flatten(), score, -1e9)
    pts = []
    for i in np.argsort(-s):
        if s[i] < -1e8:
            break
        px, py = ((i % gw) + .5) / gw, ((i // gw) + .5) / gh
        if all(math.hypot(px - x, py - y) >= MIN_SEP for x, y in pts):
            pts.append((float(px), float(py)))
        if len(pts) == k:
            break
    while len(pts) < k and pts:
        pts.append(pts[0])
    return pts


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    S = json.load(open(SCORES))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s_, add_special_tokens=False)["input_ids"][-1]
                       for s_ in [C, f" {C}"]}) for C in "ABCD"]

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

    def fit(img, target, refine=6, tol=0.08):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    def answer(imgs, text):
        i = build(imgs, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).all(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}   k={K}, {B0//K} tok/crop", flush=True)

    n, t0 = 0, time.time()
    for ex in ds:
        qid = f"{ex['category']}/{ex['question_id']}"
        ip = os.path.join(root, ex["image"])
        if qid in done or qid not in S or not os.path.exists(ip):
            continue
        d = S[qid]
        gh, gw = d["grid"]
        img = Image.open(ip).convert("RGB")
        gt = d["gt_box_frac"]
        gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        rng = random.Random(8100 + hash(qid) % 99991)
        head_pts = topk_cells(np.asarray(d["head_score"]), gh, gw, K)
        argm_pts = topk_cells(np.asarray(d["dep_score"]), gh, gw, K)
        rand_pts = [(rng.uniform(.1, .9), rng.uniform(.1, .9)) for _ in range(K)]
        Bk = B0 // K
        arms = {
            "uniform@300":     [fit(img, B0)],
            "uniform@600":     [fit(img, 2 * B0)],
            "dcr_single@300":  [fit(window(img, *head_pts[0], W), B0)],
            "oracle@300":      [fit(img.crop((int(gt[0]*img.size[0]), int(gt[1]*img.size[1]),
                                              int(gt[2]*img.size[0]), int(gt[3]*img.size[1]))), B0)],
            "dcr_multi_k4":    [fit(window(img, cx, cy, W), Bk) for cx, cy in head_pts],
            "argmax_multi_k4": [fit(window(img, cx, cy, W), Bk) for cx, cy in argm_pts],
            "rand_multi_k4":   [fit(window(img, cx, cy, W), Bk) for cx, cy in rand_pts],
        }
        rec = {"question_id_full": qid, "category": ex["category"],
               "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
               "head_pts": head_pts, "gt_box_frac": gt,
               "probs": {}, "realized_tokens": {}}
        for nm, ims in arms.items():
            p, rz = answer(ims, ex["text"])
            rec["probs"][nm] = p
            rec["realized_tokens"][nm] = rz
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 20 == 0:
            el = time.time() - t0
            print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
