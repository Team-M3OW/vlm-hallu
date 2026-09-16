"""
Phase 82: unblock the THIRD architecture. Extract per-layer attention maps from LLaVA models,
which phase 74 skipped because their image tokens do not form a clean grid.

WHY THEY WERE SKIPPED, AND WHY THAT WAS RIGHT
---------------------------------------------
LLaVA-OneVision and LLaVA-NeXT insert a learned `image_newline` embedding at the end of every row
(SS5's own finding), so `n_img_tokens` is not gh*gw and any grid inferred by square-rooting it is
wrong. Phase 74 asserted the grid and skipped rather than guess -- correct behaviour, and the reason
it wrote 0 items.

THE FIX, BORROWED FROM TWO PHASES THAT ALREADY SOLVED IT
    phase 41: locate separators by matching the model's OWN `image_newline` parameter against the
              merged input embeddings (max-abs difference < 1e-2). No unpad logic, no assumptions
              about tiling.
    phase 43: keep only MODAL-LENGTH segments as rows. Treating every inter-separator run as a row
              once swallowed LLaVA's separator-free base image as "row 0", put 59.8% of positions in
              a bucket capped at ~5%, and manufactured a spurious 3.0x result.

So: find separators, split into segments, take the modal segment length as the row width, keep only
the modal-length segments (the anyres tiles' raster), and rebuild a rectangular grid from those.
Anything that does not resolve to a clean rectangle is skipped and COUNTED, so partial coverage is
visible rather than silent.

WHAT THIS BUYS
--------------
REPLICATION_LEDGER's top claim -- "averaging across depth dilutes the read-out" -- currently rests on
two Qwen models, i.e. one family and one tokenisation scheme. LLaVA is a different family with an
EXPLICIT row separator. If it replicates there, the claim spans two families and two schemes.
"""
import json
import os
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODELS = [("onevision", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
          ("llavanext", "llava-hf/llava-v1.6-vicuna-7b-hf")]
B0 = 300
Image.MAX_IMAGE_PIXELS = None


def newline_vec(model):
    for path in (("model", "image_newline"), ("image_newline",),
                 ("model", "model", "image_newline")):
        o = model
        for p in path:
            o = getattr(o, p, None)
            if o is None:
                break
        if isinstance(o, torch.nn.Parameter):
            return o.data
    return None


def grid_from_separators(isnl):
    """Modal-length segments are the rows. Returns (gh, gw, keep_indices) or None."""
    segs, cur = [], []
    for i, f in enumerate(isnl):
        if f:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        segs.append(cur)
    if not segs:
        return None
    lens = Counter(len(s) for s in segs)
    modal = lens.most_common(1)[0][0]
    rows = [s for s in segs if len(s) == modal]
    if modal < 4 or len(rows) < 4:
        return None
    return len(rows), modal, [i for s in rows for i in s]


def run(tag, mid, out_path):
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    if os.path.exists(out_path) and sum(1 for _ in open(out_path)) >= 150:
        print(f"  {tag}: already extracted"); return
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(mid)
    itid = model.config.image_token_id
    nl = newline_vec(model)
    if nl is None:
        print(f"  {tag}: image_newline not found -- cannot segment. SKIPPING."); return
    # The image features are injected DURING forward, not at the embedding layer: every image
    # token id is the same placeholder, so get_input_embeddings() returns an IDENTICAL vector for
    # all of them (verified: min = median = max distance to image_newline). Separators are only
    # visible in the MERGED embeddings, so capture those with a pre-hook on the language model.
    CAP = {"emb": None}

    def _pre(mod, args, kwargs):
        e = kwargs.get("inputs_embeds")
        if e is None and args:
            e = next((a for a in args if torch.is_tensor(a) and a.dim() == 3), None)
        if e is not None:
            CAP["emb"] = e.detach()
        return None

    lm = getattr(getattr(model, "model", model), "language_model", None) or model.model
    hook = lm.register_forward_pre_hook(_pre, with_kwargs=True)
    nL = model.config.get_text_config().num_hidden_layers
    print(f"  {tag}: {nL} layers, image_newline dim {tuple(nl.shape)}", flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def ntok(img):
        return int((build(img, "x")["input_ids"][0] == itid).sum())

    def fit(img, target, refine=6, tol=0.10):
        W_, H_ = img.size
        sc = (target / max(ntok(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(56, int(W_ * sc)), max(56, int(H_ * sc))), Image.BICUBIC)
            r = ntok(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    n, skip, t0 = 0, 0, time.time()
    with open(out_path, "a") as fout:
        for ex in ds:
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap):
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            IW, IH = img.size
            gt = [min(b[0] for b in ann["bbox"]) / IW, min(b[1] for b in ann["bbox"]) / IH,
                  max(b[0] + b[2] for b in ann["bbox"]) / IW,
                  max(b[1] + b[3] for b in ann["bbox"]) / IH]
            try:
                inp = build(fit(img, B0), ex["text"])
            except Exception:
                skip += 1; continue
            ids = inp["input_ids"][0]
            pos = (ids == itid).nonzero().flatten()
            if len(pos) < 32:
                skip += 1; continue
            base, n_img = int(pos[0].item()), int(len(pos))
            inp = inp.to(model.device)
            CAP["emb"] = None
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            if CAP["emb"] is None:
                skip += 1; del out; torch.cuda.empty_cache(); continue
            emb = CAP["emb"][0, base:base + n_img].float()
            d = (emb - nl.float().to(emb.device)).abs().max(-1).values
            isnl = (d < 1e-2).cpu().tolist()
            g = grid_from_separators(isnl)
            if g is None:
                if skip < 2:
                    print(f"    grid fail: {int(sum(isnl))} seps of {n_img} tokens", flush=True)
                skip += 1; del out; torch.cuda.empty_cache(); continue
            gh, gw, keep = g
            A = np.stack([out.attentions[L][0, :, -1, base:base + n_img].float().mean(0).cpu().numpy()
                          for L in range(len(out.attentions))])
            del out; torch.cuda.empty_cache()
            A = A[:, keep]                      # drop separators, keep the raster
            if A.shape[1] != gh * gw or not np.isfinite(A).all():
                skip += 1; continue
            fout.write(json.dumps({
                "question_id_full": f"{ex['category']}/{ex['question_id']}",
                "category": ex["category"], "grid": [gh, gw], "n_img_tokens": gh * gw,
                "n_raw_tokens": n_img, "n_separators": int(sum(isnl)),
                "gt_box_frac": gt,
                "attn": {f"L{i}": [round(float(v), 8) for v in A[i]] for i in range(A.shape[0])}
            }) + "\n")
            fout.flush()
            n += 1
            if n % 40 == 0:
                print(f"    [{n}] {n/(time.time()-t0):.2f} it/s (skipped {skip})", flush=True)
    hook.remove()
    del model; torch.cuda.empty_cache()
    print(f"  {tag}: wrote {n}, skipped {skip} -> {out_path}", flush=True)


if __name__ == "__main__":
    for tag, mid in MODELS:
        print(f"\n=== {mid} ===", flush=True)
        try:
            run(tag, mid, f"/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase82_{tag}.jsonl")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print("\ndone", flush=True)
