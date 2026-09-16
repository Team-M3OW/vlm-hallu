"""
Phase 94: TWO-STAGE PRUNING. Keep most of the compute saving AND the late-read accuracy?

THE TENSION
-----------
FastV reads at layer 2 because pruning early is what saves compute: dropping 90% of visual tokens at
L2 of 28 avoids that work for 26 layers (~93% of visual-token compute). Phases 87/93 showed early
reads are catastrophic -- -23pp on POPE, -32pp on MMBench, both far below random. Reading at L16
instead is accurate but saves only ~43%.

HYPOTHESIS: you do not need a good ranking to discard OBVIOUS background. Early attention may be
useless for choosing the best 10% yet adequate for dropping the worst 50%.

    stage 1 at L2    keep the top 50%   -- coarse cut where the signal is weak
    stage 2 at L16   re-rank survivors, cut to 10%

ARMS (all end at 10% keep, so final cost is identical)
    none | late_only (L16) | layer2_only (FastV) | two_stage_50 | two_stage_25 | two_stage_rand50

The RANDOM first-stage control is the one that matters: if a random 50% cut at L2 followed by a late
re-rank matches the attention-guided version, then stage 1 is doing nothing and the honest method is
"drop half at random early, rank late" -- which needs no early attention at all.

PRE-REGISTERED
    two_stage_50 ~= late_only AND > rand50 -> early attention IS usable coarsely; frontier moves
    two_stage_50 ~= rand50                 -> early attention adds nothing; random first cut is the
                                              method, and that is a cleaner result
    two_stage_50 << late_only              -> the early cut destroys what the late stage needs
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase94_twostage.jsonl"
EARLY, LATE_L = 2, 16
BLOCK = list(range(16, 27))
B0, FINAL_KEEP = 300, 0.10
Image.MAX_IMAGE_PIXELS = None


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    k = QM.repeat_kv(key, module.num_key_value_groups)
    v = QM.repeat_kv(value, module.num_key_value_groups)
    a = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        a = a + attention_mask[:, :, :, : k.shape[-2]]
    spec = getattr(module, "_stage", None)
    if spec is not None:
        li = getattr(module, "_layer_idx", 99)
        for frm, bias in spec:
            if li >= frm and bias.shape[-1] == a.shape[-1]:
                a = a + bias.to(a.dtype).view(1, 1, 1, -1)
    a = torch.nn.functional.softmax(a, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(a, v).transpose(1, 2).contiguous(), a


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok, itid = pr.tokenizer, model.config.image_token_id
    opt = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                   for s in [C, f" {C}"]}) for C in "ABCD"]
    layers = model.model.language_model.layers
    for i, l in enumerate(layers):
        l.self_attn._layer_idx = i

    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False,
                                                          add_generation_prompt=True),
                  return_tensors="pt")

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
        return best[0]

    def clear():
        for l in layers:
            if hasattr(l.self_attn, "_stage"):
                del l.self_attn._stage

    def ask(inp, stages):
        clear()
        n = inp["input_ids"].shape[1]
        spec = []
        for frm, drop in stages:
            if drop is None or len(drop) == 0:
                continue
            b = torch.zeros(n, device=model.device)
            b[torch.as_tensor(sorted(drop), device=model.device)] = -1e4
            spec.append((frm, b))
        if spec:
            for l in layers:
                l.self_attn._stage = spec
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        clear()
        assert torch.isfinite(lg).all()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["qid"])
    print(f"resuming: {len(done)}   stage1 L{EARLY} -> stage2 L{LATE_L}, final keep {FINAL_KEEP}",
          flush=True)
    rng = random.Random(94)
    n, t0 = 0, time.time()
    for ex in ds:
        qid = f"{ex['category']}/{ex['question_id']}"
        ip = os.path.join(root, ex["image"])
        if qid in done or not os.path.exists(ip):
            continue
        inp = build(fit(Image.open(ip).convert("RGB"), B0), ex["text"]).to(model.device)
        pos = (inp["input_ids"][0] == itid).nonzero().flatten()
        base, ntok = int(pos[0].item()), int(len(pos))
        clear()
        with torch.no_grad():
            o = model(**inp, output_attentions=True)
        A = np.stack([o.attentions[L][0, :, -1, base:base + ntok].float().mean(0).cpu().numpy()
                      for L in range(len(o.attentions))])
        del o
        torch.cuda.empty_cache()
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        early, late = A[EARLY], A[BLOCK].mean(0)
        kf = max(1, int(round(FINAL_KEEP * ntok)))
        idx = lambda a: (base + np.asarray(a)).tolist()

        rec = {"qid": qid, "category": ex["category"], "n_img": ntok,
               "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
               "probs": {}}
        rec["probs"]["none"] = ask(inp, [])
        rec["probs"]["late_only"] = ask(inp, [(LATE_L, idx(np.argsort(-late)[kf:]))])
        rec["probs"]["layer2_only"] = ask(inp, [(EARLY, idx(np.argsort(-early)[kf:]))])
        for tag, frac, score in [("two_stage_50", 0.50, early), ("two_stage_25", 0.25, early),
                                 ("two_stage_rand50", 0.50, None)]:
            k1 = max(kf, int(round(frac * ntok)))
            order = (np.array(sorted(range(ntok), key=lambda _: rng.random()))
                     if score is None else np.argsort(-score))
            keep1 = order[:k1]
            surv = keep1[np.argsort(-late[keep1])]
            rec["probs"][tag] = ask(inp, [(EARLY, idx(order[k1:])), (LATE_L, idx(surv[kf:]))])
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 25 == 0:
            print(f"  [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
