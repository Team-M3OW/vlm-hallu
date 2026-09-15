"""
Phase 77: EVIDENCE-GROUNDED CONTRASTIVE DECODING. A decoding-time fix, no crop, no re-encoding.

THE IDEA
--------
Run the model normally, then again with the evidence region MASKED OUT of attention, and push the
logits away from the masked version:

        logits_final = logits_full + alpha * (logits_full - logits_masked)

Both passes are at B0 = 300 realized tokens. Nothing is re-encoded, no pixels are added, the image
is never cut. The compute bar is therefore uniform@600, and it is measured in-run.

WHY THIS IS NOT ONE OF THE SIX INTERVENTIONS THAT ALREADY FAILED
----------------------------------------------------------------
SS10B tried to AMPLIFY attention on the evidence cells and got +5.8pp with a GROUND-TRUTH target --
16% of what the crop buys -- because the target is sub-token and there is nothing there to amplify.
Contrast is a different operation: it does not need the region to be attended more, only to
CONTRIBUTE something, and it then extrapolates that contribution. SS10A's nulls also used the raw
argmax (39.3% coverage) or random cells; the head (52.9%, SS14C) did not exist yet.

ODDS, STATED HONESTLY BEFORE THE RUN: SS10A found that biasing attention away from random or
interior IMAGE tokens is indistinguishable from doing nothing, while doing it to PROMPT tokens costs
-17.3pp. If individual image tokens contribute nothing measurable, the contrast term is ~0 and every
arm collapses to baseline. That is the most likely outcome and it would be a clean seventh null.

ARMS (all one image, B0 tokens, 2 LM passes each; alpha swept)
    baseline           no intervention
    cd_oracle          mask the GT-box cells                  <- CEILING. If this is flat, the
                                                                 method is dead regardless of the
                                                                 localiser, and we stop.
    cd_head            mask the Phase 71a head's region       <- the method
    cd_argmax          mask the deployed argmax's region      <- incumbent localiser
    cd_rand            mask a random equal-sized region       <- CONTROL. cd_head must beat this.
    mask_only_head     the masked pass alone, no contrast     <- shows whether masking does anything

BUILT-IN FALSIFICATION: the contrast must be LARGER on items where the head's window covers the
evidence than where it misses. If the gain does not track coverage it is not about evidence, and
the arm is void whatever its accuracy.

PRIOR ART -- must be checked before any novelty claim. Contrastive decoding for VLMs is crowded:
VCD (distorted image), ICD (perturbed instruction), Pensieve (retrieved images), RITUAL
(transformed images). What is not obviously present is grounding the contrast in a LEARNED
LOCALISER over the model's own attention. Treat as unverified until LITERATURE_GAPS says otherwise.
"""
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROP = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71a_head_proposals.json"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase77_contrastive_decode.jsonl"
B0, W = 300, 0.15
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
Image.MAX_IMAGE_PIXELS = None
_orig = QM.eager_attention_forward


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    k = QM.repeat_kv(key, module.num_key_value_groups)
    v = QM.repeat_kv(value, module.num_key_value_groups)
    a = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        a = a + attention_mask[:, :, :, : k.shape[-2]]
    b = getattr(module, "_mask_bias", None)
    if b is not None and b.shape[-1] == a.shape[-1]:
        a = a + b.to(a.dtype).view(1, 1, 1, -1)
    a = torch.nn.functional.softmax(a, dim=-1, dtype=torch.float32).to(query.dtype)
    a = torch.nn.functional.dropout(a, p=dropout, training=module.training)
    return torch.matmul(a, v).transpose(1, 2).contiguous(), a


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    layers = model.model.language_model.layers

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
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best

    def clear():
        for l in layers:
            if hasattr(l.self_attn, "_mask_bias"):
                del l.self_attn._mask_bias

    def logits(inp, drop=None):
        clear()
        if drop is not None and len(drop):
            n = inp["input_ids"].shape[1]
            b = torch.zeros(n, device=model.device)
            b[torch.as_tensor(sorted(drop), device=model.device)] = -1e4
            for l in layers:
                l.self_attn._mask_bias = b
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        clear()
        assert torch.isfinite(lg).all(), "non-finite logits"
        return torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids])

    def cells_in(cx, cy, gh, gw, base):
        """token indices whose cell centre falls inside the W-window centred at (cx,cy)"""
        x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
        if x0 < 0: x0, x1 = 0.0, W
        if y0 < 0: y0, y1 = 0.0, W
        if x1 > 1: x0, x1 = 1 - W, 1.0
        if y1 > 1: y0, y1 = 1 - W, 1.0
        out = []
        for i in range(gh * gw):
            fx, fy = ((i % gw) + .5) / gw, ((i // gw) + .5) / gh
            if x0 <= fx <= x1 and y0 <= fy <= y1:
                out.append(base + i)
        return out

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    for ex in ds:
        qid = f"{ex['category']}/{ex['question_id']}"
        ip = os.path.join(root, ex["image"])
        if qid in done or qid not in props or not os.path.exists(ip):
            continue
        img = Image.open(ip).convert("RGB")
        small, rz = fit(img, B0)
        inp = build(small, ex["text"]).to(model.device)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        pos = (inp["input_ids"][0] == itid).nonzero().flatten()
        base = int(pos[0].item())
        pp = props[qid]
        gt = pp["gt_box_frac"]
        rng = random.Random(7700 + hash(qid) % 99991)
        regions = {
            "oracle": cells_in((gt[0]+gt[2])/2, (gt[1]+gt[3])/2, gh, gw, base),
            "head": cells_in(*pp["head"], gh, gw, base),
            "argmax": cells_in(*pp["argmax"], gh, gw, base),
            "rand": cells_in(rng.uniform(.1, .9), rng.uniform(.1, .9), gh, gw, base),
        }
        full = logits(inp)
        rec = {"question_id_full": qid, "category": ex["category"],
               "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
               "head_cov": pp["head_cov"], "argmax_cov": pp["argmax_cov"],
               "realized_tokens": int(rz), "n_masked": {k: len(v) for k, v in regions.items()},
               "probs": {}, "contrast_l1": {}}
        rec["probs"]["baseline"] = [round(float(v), 6)
                                    for v in torch.softmax(full, 0).tolist()]
        for nm, cells in regions.items():
            m = logits(inp, cells)
            rec["contrast_l1"][nm] = round(float((full - m).abs().sum()), 6)
            if nm == "head":
                rec["probs"]["mask_only_head"] = [round(float(v), 6)
                                                  for v in torch.softmax(m, 0).tolist()]
            for a in ALPHAS:
                if a == 0.0:
                    continue
                rec["probs"][f"cd_{nm}@{a}"] = [
                    round(float(v), 6) for v in torch.softmax(full + a * (full - m), 0).tolist()]
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 20 == 0:
            el = time.time() - t0
            print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
