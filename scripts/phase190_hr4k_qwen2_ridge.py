"""
Phase 72c (corrected, QWEN2-VL-7B): the METHOD crossing -- second model on the second benchmark. Original header follows.

Phase 72c (corrected): TRANSFER. The V*Bench-trained re-ranking head, applied to HR-Bench 4k, nothing refitted.

WHY THIS IS THE DECIDING RUN
----------------------------
SS14D showed the head converts on V*Bench: +8.4pp over the deployed argmax, and +13.0pp over the
compute-matched bar on single-region questions. Its stated exposure was that the head is trained on
V*Bench's own GT boxes. HR-Bench 4k removes that objection completely -- it ships NO boxes, so the
head CANNOT have been fitted here even in principle, and nothing about it is touched in this file.

    trained on   V*Bench, 191 items, GT-box coverage labels      (phase72a, frozen)
    applied to   HR-Bench 4k, 200 instances x 4 permutations, 4032x4032 images
    refitted     nothing. Not W, not the layer block, not the ring mask, not B0.

HR-Bench is also the benchmark where allocation has historically LOST: SS9B measured our gated
method at -0.2pp and Zoom Eye at -15.3pp against uniform. So this is a hard test, not a friendly one.

ARMS (tokens measured, never computed)
    uniform@300      B0 baseline
    uniform@600      COMPUTE-MATCHED BAR -- every allocation arm pays localise + crop
    argmax@0.15      deployed allocator (incumbent)
    head@0.15        frozen V*Bench head re-ranks the SAME map                   <- the method
    rand@0.15        random centre, same W -- placement control
NO ORACLE ARM: HR-Bench ships no boxes, so no coverage and no oracle-capture fraction is computable
here, and none will be quoted.

CIRCULAR EVAL: 800 rows = 200 instances x 4 option permutations. An instance counts only if all 4
are correct. Both per-row and circular are reported; circular is the benchmark's own metric.
Proposal geometry depends only on (image, question), so the attention pass runs once per INSTANCE.

Instance keying includes the IMAGE HASH: HR-Bench has only 159 unique question strings across 200
instances, and keying on (question, category) alone silently paired 212/800 rows with the WRONG
image in an earlier phase.
"""
import base64
import hashlib
import io
import json
import os
import random
import time

import joblib
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
HEAD = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase72a_qwen2_head.joblib"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase190_hr4k_qwen2_ridge.jsonl"
BLOCK = list(range(15, 27))  # L15-26 of 28: same stack fraction as Qwen3-VL L16-26
B0, W = 300, 0.15
MAX_MP = 24_000_000
Image.MAX_IMAGE_PIXELS = None


def cell_features(A, gh, gw):
    """Must mirror phase70_rerank_head.build() column-for-column, or the head reads garbage."""
    nL, n = A.shape
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    M = A.reshape(nL, gh, gw)
    dep = M[BLOCK].mean(0)
    R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)).reshape(nL, gh, gw)
    pad = np.pad(dep, 1, mode="edge")
    nb = sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = (yy + .5) / gh, (xx + .5) / gw
    F = np.concatenate([
        M.reshape(nL, -1).T, R.reshape(nL, -1).T,
        nb.reshape(-1, 1), dep.reshape(-1, 1),
        fx.reshape(-1, 1), fy.reshape(-1, 1),
        np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2).reshape(-1, 1),
        np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)).reshape(-1, 1),
        (xx == gw - 1).astype(float).reshape(-1, 1),
        (yy == gh - 1).astype(float).reshape(-1, 1),
        (xx == 0).astype(float).reshape(-1, 1)], axis=1)
    return F, dep.flatten()


RIDGE_TAG = "qwen2"
import sys as _sys; _sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")

# ---- phase 190: RIDGE scorer (§21 features), fit on ALL V*Bench maps of this model; transfer to HR-Bench, nothing refit ----
_RSRC = {"qwen3": ("/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl", (16, 27)),
         "qwen2": ("/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27))}[RIDGE_TAG]
def _ridge_feats(A, gh, gw):
    NL = A.shape[0]; n = gh * gw
    LA = np.log(A + 1e-12).T; R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)).T
    b0, b1 = _RSRC[1]; dep = A[b0:b1].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0).ravel()
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)), (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo]
def _fit_ridge(lam=1.0):
    import phase70_rerank_head as _P70; _P70.W = W   # fit at the DEPLOYED window (0.15 on HR-Bench), not 0.25
    X = []; Y = []
    for l in open(_RSRC[0]):
        r = json.loads(l); gh, gw = r["grid"]; n = r["n_img_tokens"]
        if gh * gw != n: continue
        NL = len([k for k in r["attn"] if k.startswith("L")])
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)]); A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
        X.append(_ridge_feats(A, gh, gw)); Y.append(np.array([_P70.coverage(float(fx[i]), float(fy[i]), r["gt_box_frac"]) for i in range(n)]))
    X = np.vstack(X); Y = np.concatenate(Y); mu, sd = X.mean(0), X.std(0) + 1e-9; Xt = np.c_[(X - mu) / sd, np.ones(len(X))]
    A_ = Xt.T @ Xt + lam * np.eye(Xt.shape[1]); A_[-1, -1] -= lam; w = np.linalg.solve(A_, Xt.T @ Y)
    print(f"ridge fit on {len(Y)} V*Bench cells x {X.shape[1]} features at W={_P70.W} (TRANSFER: fit on all V*Bench items, applied to HR-Bench, nothing refit)", flush=True)
    return lambda A, gh, gw: np.c_[(_ridge_feats(A, gh, gw) - mu) / sd, np.ones(gh * gw)] @ w
ridge_score = _fit_ridge()

def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    bundle = joblib.load(HEAD)
    head, spec = bundle["model"], bundle["spec"]
    print(f"head: trained on {spec['train_items']} V*Bench items, "
          f"{spec['train_cells_min']}-{spec['train_cells_max']} cells/item", flush=True)

    p = hf_hub_download("DreamMr/HR-Bench", "hr_bench_4k.parquet", repo_type="dataset")
    df = pd.read_parquet(p).reset_index(drop=True)
    df["_imghash"] = df["image"].map(
        lambda b: hashlib.md5(b.encode() if isinstance(b, str) else b).hexdigest())
    df["_inst"] = df.groupby(["question", "category", "_imghash"]).ngroup()
    sz = df.groupby("_inst").size()
    assert set(sz.unique()) == {4}, f"expected 4 rows/instance, got {dict(sz.value_counts())}"
    print(f"HR-Bench 4k: {len(df)} rows, {df['_inst'].nunique()} instances (hash-verified)",
          flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.06):
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
        return best if best else (img, measure(img))

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).all(), "non-finite logits -- check dtype"
        pv = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in pv.tolist()], rz

    def propose(img, text):
        """One pass at B0. Returns the argmax cell AND the head's cell from the SAME map."""
        small, _ = fit(img, B0)
        inp = build(small, text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n = gh * gw
        base = int((inp["input_ids"][0] == itid).nonzero().flatten()[0].item())
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        A = np.stack([out.attentions[L][0, :, -1, base:base + n].float().mean(0).cpu().numpy()
                      for L in range(len(out.attentions))])
        del out
        torch.cuda.empty_cache()
        assert np.isfinite(A).all(), "non-finite attention"
        F, dep = cell_features(A, gh, gw)
        assert F.shape[1] == spec["n_features"], f"feature drift {F.shape[1]}"
        rm = np.zeros((gh, gw), dtype=bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        rmf = rm.flatten()
        s = head.predict(F)
        sr = ridge_score(A, gh, gw)
        cell = lambda i: (((i % gw) + .5) / gw, ((i // gw) + .5) / gh)
        return (cell(int(np.argmax(np.where(rmf, s, -1e9)))),
                cell(int(np.argmax(np.where(rmf, dep, -1e9)))), n,
                cell(int(np.argmax(np.where(rmf, sr, -1e9)))))

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

    n_, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for inst, grp in df.groupby("_inst"):
            rows = [r for _, r in grp.iterrows() if int(r["index"]) not in done]
            if not rows:
                continue
            img = Image.open(io.BytesIO(base64.b64decode(grp["image"].iloc[0]))).convert("RGB")
            # CORRECTED (phase 116/117): 72b localised on the BARE QUESTION with the four options
            # stripped, unlike every other phase in this project (phase 71 passes the full prompt).
            # On identical rows that cost the crop arm 19.3pp -- argmax@0.15 42.7% -> 62.0% -- and the
            # argmax cell was identical on only 44.7% of items. Attention is question-conditioned
            # (SS14R) and the read-out depends on which query token is read (phase 48), so stripping
            # the options removes most of the text the localiser conditions on.
            hcell, acell, ncell, rcell = propose(img, prompt(rows[0]))
            rng = random.Random(7200 + int(inst))
            rcx, rcy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            arms = {
                "uniform@300": fit(img, B0)[0],
                "uniform@600": fit(img, 2 * B0)[0],
                "argmax@0.15": fit(window(img, *acell, W), B0)[0],
                "head@0.15":   fit(window(img, *hcell, W), B0)[0],
                "ridge@0.15":  fit(window(img, *rcell, W), B0)[0],
                "rand@0.15":   fit(window(img, rcx, rcy, W), B0)[0],
            }
            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst),
                       "category": r["category"], "cycle": int(r["cycle_category"]),
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "head_cell": list(hcell), "argmax_cell": list(acell), "ridge_cell": list(rcell),
                       "n_cells": int(ncell), "probs": {}, "realized_tokens": {}}
                for nm, im in arms.items():
                    pv, rz = answer(im, text)
                    rec["probs"][nm] = pv
                    rec["realized_tokens"][nm] = rz
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_ += 1
            if n_ % 40 < 4:
                el = time.time() - t0
                print(f"  [{n_}/800] {n_/el:.2f} rows/s eta={(800-n_)/max(n_/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
