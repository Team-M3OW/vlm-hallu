"""
Phase 93: does the layer-2 pruning collapse hold on GENERAL VQA, or only on small-object search?

THE OBJECTION
-------------
SS14L and phase 87 are measured on V*Bench -- a small-object SEARCH benchmark where the answer hinges
on one tiny region, so question-conditioned late-layer attention is exactly what should matter and
early-layer attention exactly what should fail. That is the benchmark most likely to flatter the
claim. The pruning literature evaluates on general VQA, where the answer often does NOT depend on
locating one small thing.

BENCHMARKS (both load from local cache; the machine's disk is 99% full so nothing new downloads)
    POPE         9000 rows, yes/no object PRESENCE. Not a search task at all -- the object is
                 usually large or absent. The strongest available test of the objection.
    MMBench_EN   4377 rows, 4-way multiple choice, general vision-language reasoning.

⚠ MMBench was rejected earlier in this project for ALLOCATION experiments (512px images leave no
allocation headroom). That objection does not apply to pruning, which does not add pixels.

PRE-REGISTERED READING, fixed before the run
    layer-2 collapses on BOTH      -> the claim is about pruning read-outs generally
    layer-2 is fine on both        -> the claim is SCOPED to small-object search and the paper's
                                      framing must shrink to match
    mixed                          -> report per benchmark, make no general claim

ARMS: none | rand | layer2 (FastV default) | late block -- all pruning the SAME token count at 10%
keep, so cost is matched by construction. Scored from option logits, so no generation and results
are deterministic.
"""
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase93_pruning_general.jsonl"
BLOCK, FASTV_K, B0, KEEP, N_PER = list(range(16, 27)), 2, 300, 0.10, 200
Image.MAX_IMAGE_PIXELS = None


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
    return torch.matmul(a, v).transpose(1, 2).contiguous(), a


def main():
    from datasets import load_dataset
    benches = []
    try:
        d = load_dataset("lmms-lab/POPE", "default", split="test")
        rows = []
        for ex in d.shuffle(seed=93).select(range(min(4 * N_PER, len(d)))):
            if ex.get("image") is None:
                continue
            rows.append({"img": ex["image"], "q": ex["question"],
                         "opts": ["yes", "no"], "gold": 0 if str(ex["answer"]).lower().startswith("y") else 1})
            if len(rows) >= N_PER:
                break
        benches.append(("POPE", rows))
    except Exception as e:
        print(f"POPE skip: {e}", flush=True)
    try:
        d = load_dataset("lmms-lab/MMBench_EN", "default", split="dev")
        rows = []
        for ex in d.shuffle(seed=93).select(range(min(4 * N_PER, len(d)))):
            if ex.get("image") is None or not ex.get("A") or not ex.get("B"):
                continue
            o = [ex.get(c) for c in "ABCD"]
            n = sum(1 for x in o if x)
            if n < 2 or str(ex.get("answer", "")).upper() not in "ABCD"[:n]:
                continue
            q = ex["question"] + "".join(f"\n({c}) {ex[c]}" for c in "ABCD"[:n] if ex.get(c))
            rows.append({"img": ex["image"], "q": q, "opts": list("ABCD"[:n]),
                         "gold": "ABCD".index(str(ex["answer"]).upper())})
            if len(rows) >= N_PER:
                break
        benches.append(("MMBench", rows))
    except Exception as e:
        print(f"MMBench skip: {e}", flush=True)
    for t, r in benches:
        print(f"  {t}: {len(r)} items", flush=True)
    if not benches:
        return

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok, itid = pr.tokenizer, model.config.image_token_id
    layers = model.model.language_model.layers

    def ids_for(o):
        return sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in [o, f" {o}",
                       o.capitalize(), f" {o.capitalize()}"]})

    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False,
                                                          add_generation_prompt=True),
                  return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.08):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target):
                best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol:
                break
            sc *= (target/r) ** 0.5
        return best[0]

    def clear():
        for l in layers:
            if hasattr(l.self_attn, "_mask_bias"):
                del l.self_attn._mask_bias

    def ask(inp, oid, drop=None):
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
        assert torch.isfinite(lg).all()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in oid]), 0)
        return int(torch.argmax(p))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["bench"], r["idx"]))
    print(f"resuming: {len(done)}", flush=True)
    rng = random.Random(93)
    n, t0 = 0, time.time()
    for tag, rows in benches:
        for i, ex in enumerate(rows):
            if (tag, i) in done:
                continue
            try:
                oid = [ids_for(o) for o in ex["opts"]]
                inp = build(fit(ex["img"].convert("RGB"), B0),
                            ex["q"] + "\nAnswer with the option's letter from the given choices directly."
                            if len(ex["opts"]) > 2 else ex["q"] + "\nAnswer yes or no."
                            ).to(model.device)
                pos = (inp["input_ids"][0] == itid).nonzero().flatten()
                base, ntok = int(pos[0].item()), int(len(pos))
                clear()
                with torch.no_grad():
                    o = model(**inp, output_attentions=True)
                A = np.stack([o.attentions[L][0, :, -1, base:base+ntok].float().mean(0).cpu().numpy()
                              for L in range(len(o.attentions))])
                del o; torch.cuda.empty_cache()
                A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
                k = max(1, int(round(KEEP * ntok)))
                rec = {"bench": tag, "idx": i, "gold": ex["gold"], "n_img": ntok,
                       "pred": {"none": ask(inp, oid)}}
                for nm, s in [("layer2", A[FASTV_K]), ("late", A[BLOCK].mean(0)),
                              ("rand", np.array([rng.random() for _ in range(ntok)]))]:
                    rec["pred"][nm] = ask(inp, oid, (base + np.argsort(-s)[k:]).tolist())
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n += 1
                if n % 40 == 0:
                    print(f"  [{tag} {i+1}/{len(rows)}] {n/(time.time()-t0):.2f} it/s", flush=True)
            except Exception as e:
                print(f"  {tag}[{i}] {type(e).__name__}: {str(e)[:50]}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
