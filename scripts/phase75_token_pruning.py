"""
Phase 75: does the LAYER-CONTRAST insight generalise to a DIFFERENT TASK -- visual token pruning?

THE BET
-------
FastV and the visual-token-pruning literature rank image tokens by attention received AT ONE LAYER
(FastV uses K=2) and drop the low-ranked ones. SS14F showed, for localisation, that:

    no single layer is a good ranker           (gt_pct 0.456/0.413/0.409 vs 0.500 chance)
    the deployed block MEAN is barely better    (39.3% vs best single layer 41.9%)
    a learned SIGNED combination is much better (45.5% linear, 52.9% full head)
    because some layers point at DISTRACTORS and a mean can only ADD, never SUBTRACT

If that is a property of VLM attention rather than of our task, then ranking tokens for PRUNING by
a signed layer combination should beat ranking them by any single layer -- on a different task, a
different metric, and against an established baseline. That would lift the finding out of this
paper's own problem.

If it does NOT transfer, the finding is specific to "which cell contains the queried evidence", which
is a narrower but still honest claim. Either way it is worth knowing.

ARMS (all prune the SAME NUMBER of tokens, so cost is matched by construction)
    none                 no pruning -- the upper bound
    rand                 random tokens dropped -- the control that makes the others readable
    layerK               rank by attention at layer K alone            (FastV-style)
    blockmean            rank by the mean over layers 16-26            (our deployed read-out)
    linear               rank by the SS14F learned signed combination  (the hypothesis)

KEEP FRACTIONS: 10%, 25%, 50% of image tokens.

MECHANISM: pruning is implemented as a large negative additive bias on the pruned image-token
columns of the attention logits, applied from layer K onward -- functionally what FastV does. The
ranking pass and the answer pass are both counted for every arm, so no arm is cheaper than another
and the contrast is about RANKING QUALITY only, never cost.

The SS14F weights were fitted to predict crop-window coverage on V*Bench, NOT to predict pruning
quality. Using them unchanged is the honest test of transfer; refitting them here would answer a
different and much weaker question.
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
WEIGHTS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase73_layer_structure.json"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase75_token_pruning.jsonl"
BLOCK = list(range(16, 27))
FASTV_K = 2
B0 = 300
KEEP = [0.10, 0.25, 0.50]
Image.MAX_IMAGE_PIXELS = None

_orig = QM.eager_attention_forward


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states = QM.repeat_kv(key, module.num_key_value_groups)
    value_states = QM.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    b = getattr(module, "_prune_bias", None)
    if b is not None and b.shape[-1] == attn_weights.shape[-1]:
        attn_weights = attn_weights + b.to(attn_weights.dtype).view(1, 1, 1, -1)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32
                                               ).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    out = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return out, attn_weights


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    W = np.asarray(json.load(open(WEIGHTS))["weights"], dtype=float)
    print(f"SS14F weights loaded: {len(W)} layers, "
          f"{int((W>0).sum())} positive / {int((W<0).sum())} negative", flush=True)

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
    print(f"{len(layers)} decoder layers", flush=True)

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
            if hasattr(l.self_attn, "_prune_bias"):
                del l.self_attn._prune_bias

    def run(inp, drop=None, from_layer=FASTV_K, want_attn=False):
        clear()
        if drop is not None and len(drop):
            n = inp["input_ids"].shape[1]
            b = torch.zeros(n, device=model.device)
            b[torch.as_tensor(drop, device=model.device)] = -1e4
            for li in range(from_layer, len(layers)):
                layers[li].self_attn._prune_bias = b
        with torch.no_grad():
            out = model(**inp, output_attentions=want_attn)
        lg = out.logits[0, -1].float()
        assert torch.isfinite(lg).all(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        A = None
        if want_attn:
            A = np.stack([out.attentions[L][0, :, -1, :].float().mean(0).cpu().numpy()
                          for L in range(len(out.attentions))])
        del out
        torch.cuda.empty_cache()
        clear()
        return [round(float(v), 6) for v in p.tolist()], A

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    rng = random.Random(75)
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            if qid in done or not os.path.exists(ip):
                continue
            img = Image.open(ip).convert("RGB")
            small, _ = fit(img, B0)
            inp = build(small, ex["text"]).to(model.device)
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            base, ntok = int(pos[0].item()), int(len(pos))
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])

            p_none, A = run(inp, want_attn=True)
            Ai = A[:, base:base + ntok]
            Ai = Ai / np.maximum(Ai.sum(1, keepdims=True), 1e-12)
            scores = {
                "layerK": Ai[FASTV_K],
                "blockmean": Ai[BLOCK].mean(0),
                "linear": (W[:Ai.shape[0], None] * Ai).sum(0),
                "rand": np.array([rng.random() for _ in range(ntok)]),
            }
            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "n_img": ntok, "probs": {"none": p_none}}
            for kf in KEEP:
                k = max(1, int(round(kf * ntok)))
                for nm, s in scores.items():
                    drop = (base + np.argsort(-s)[k:]).tolist()
                    p_, _ = run(inp, drop=drop)
                    rec["probs"][f"{nm}@{kf}"] = p_
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
