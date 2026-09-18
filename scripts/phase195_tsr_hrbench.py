"""
Phase 195: TSR ON HR-BENCH -- recheck the crop-free resolution reallocation on a DIFFERENT BENCHMARK, n=400/stratum.
TSR (§26) is currently V*Bench-only: Qwen3-VL-2B +6.3 ✔ pooled / +10.5 ✔ cross-instance; Qwen2-VL-7B +2.6 n.s.
(that checkpoint has zero cross-instance resolution headroom). HR-Bench inverts the asymmetry -- measured on disk
(phase 54/56/57, n=800-1600): 600->1200 tokens is worth +4.9 to +7.5 ✔ on SINGLE-instance and only +1.8 to +2.1 n.s.
on CROSS. So HR-Bench tests the same mechanism where the headroom lives in the other stratum.
METHOD (no crop, question-type agnostic): encode at 900 tokens; full width through L0-L16; at L16 keep the top 10%
of visual tokens by last-prompt-token attention (mean L12-L16); run L17-L27 narrow.
    token-layers = 900*17 + 90*11 = 16,290  <=  bar 600*28 = 16,800   (97%)
ARMS  uniform@300 | uniform@600 (BAR) | tsr900 | uniform@900 (DIAGNOSTIC, 150% of bar: decomposes resolution vs pruning)
PRE-REGISTERED
    P1  tsr900 - bar, pooled, CI clear of zero          S1  tsr900 - uniform@900 not significantly negative (pruning free)
    Reported per stratum; HR-Bench ships no boxes, so there is no coverage or oracle arm (as in §72c/§155).
    Prediction from the headroom above: a gain on SINGLE, ~nothing on CROSS -- the mirror of V*Bench-Qwen3.
usage: phase195_tsr_hrbench.py <4k|8k> <qwen3|qwen2>
"""
import base64
import hashlib
import io
import json, sys
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

WHICH_BENCH, WHICH_MODEL = sys.argv[1], sys.argv[2]
MODEL_ID = {"qwen3":"Qwen/Qwen3-VL-2B-Instruct","qwen2":"Qwen/Qwen2-VL-7B-Instruct"}[WHICH_MODEL]
HEAD = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase72a_head.joblib"
OUT = f"/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase195_tsr_hr{WHICH_BENCH}_{WHICH_MODEL}.jsonl"
BLOCK = list(range(16, 27))
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


NL_TSR, PRUNE_L, KEEP_TSR, E_TSR = 28, 16, 0.10, 900
import transformers.models.qwen3_vl.modeling_qwen3_vl as _QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as _QM2
def _mk(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks=QM.repeat_kv(key,module.num_key_value_groups); vs=QM.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query,ks.transpose(2,3))*scaling
        if attention_mask is not None: w=w+attention_mask[:,:,:,:ks.shape[-2]]
        b=getattr(module,"_prune_bias",None)
        if b is not None and b.shape[-1]==w.shape[-1]: w=w+b.to(w.dtype).view(1,1,1,-1)
        w=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype)
        return torch.matmul(w,vs).transpose(1,2).contiguous(), w
    return patched
_QM3.eager_attention_forward=_mk(_QM3); _QM2.eager_attention_forward=_mk(_QM2)

def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    print(f"TSR on HR-Bench {WHICH_BENCH} / {WHICH_MODEL}: encode@{E_TSR}, prune to {KEEP_TSR:.0%} at L{PRUNE_L}", flush=True)

    p = hf_hub_download("DreamMr/HR-Bench", f"hr_bench_{WHICH_BENCH}.parquet", repo_type="dataset")
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

    layers = model.model.language_model.layers
    def _clear():
        for l in layers:
            if hasattr(l.self_attn, "_prune_bias"): del l.self_attn._prune_bias
    def answer(img, text, prune=False):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device); _clear(); tl = rz * NL_TSR
        if prune:
            pos = (i["input_ids"][0] == itid).nonzero().flatten(); base, n = int(pos[0]), int(len(pos))
            with torch.no_grad(): o = model(**i, output_attentions=True)
            A = np.stack([o.attentions[L][0, :, -1, base:base + n].float().mean(0).cpu().numpy()
                          for L in range(PRUNE_L - 4, PRUNE_L + 1)]); del o
            A = A / np.maximum(A.sum(1, keepdims=True), 1e-12); sc = A.mean(0)
            keep = max(1, int(round(KEEP_TSR * n)))
            drop = (base + np.argsort(-sc)[keep:]).tolist()
            b = torch.zeros(i["input_ids"].shape[1], device=model.device)
            b[torch.as_tensor(drop, device=model.device)] = -1e4
            for li in range(PRUNE_L + 1, NL_TSR): layers[li].self_attn._prune_bias = b
            tl = n * (PRUNE_L + 1) + keep * (NL_TSR - 1 - PRUNE_L)
        with torch.no_grad(): lg = model(**i).logits[0, -1].float()
        _clear()
        assert torch.isfinite(lg).all(), "non-finite logits"
        pv = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        del i; torch.cuda.empty_cache()
        return [round(float(v), 6) for v in pv.tolist()], rz, tl

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
            plan = {"uniform@300": (fit(img, B0)[0], False), "uniform@600": (fit(img, 2 * B0)[0], False),
                    "tsr900": (fit(img, E_TSR)[0], True), "uniform@900": (fit(img, E_TSR)[0], False)}
            for r in rows:
                text = prompt(r)
                rec = {"row_id": int(r["index"]), "instance": int(inst),
                       "category": r["category"], "cycle": int(r["cycle_category"]),
                       "label": "ABCD".index(str(r["answer"]).strip().upper()),
                       "probs": {}, "realized_tokens": {}, "token_layers": {}}
                for nm, (im, pz) in plan.items():
                    pv, rz, tl = answer(im, text, pz)
                    rec["probs"][nm] = pv
                    rec["realized_tokens"][nm] = rz
                    rec["token_layers"][nm] = tl
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
