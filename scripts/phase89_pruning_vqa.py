"""
Phase 89: does the layer-2 pruning collapse hold on the benchmarks the PRUNING LITERATURE uses?

THE OBJECTION THIS ANSWERS
--------------------------
SS14L is measured on V*Bench, which is the benchmark most likely to flatter it: a small-object
SEARCH benchmark where the answer hinges on one tiny region, so question-conditioned late-layer
attention is exactly what matters and early-layer attention is exactly what should fail. FastV and
its successors are evaluated on general VQA -- POPE, GQA, TextVQA, ScienceQA -- where the answer
often does NOT depend on locating one small thing, and where an early-layer read may be perfectly
adequate for deciding which tokens are redundant.

PRE-REGISTERED READING, fixed before the run
    layer-2 collapses on >=2 of these benchmarks  -> the claim is about pruning read-outs generally
    layer-2 is fine on general VQA                -> the claim is SCOPED to small-object search,
                                                     and the abstract overreaches
    mixed                                         -> report per benchmark; make no general claim

Open-ended answers are scored by normalised exact match against the reference set (the standard
protocol for these benchmarks); POPE and ScienceQA are effectively closed-set.
"""
import json
import re
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase89_pruning_vqa.jsonl"
BLOCK = list(range(16, 27))
KS = [2]
B0, KEEP = 300, 0.10
N_PER = 150
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


def norm(t):
    t = str(t).lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())


def load_benches():
    from datasets import load_dataset
    specs = [("POPE", "lmms-lab/POPE", "default", "test"),
             ("GQA", "lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
             ("TextVQA", "lmms-lab/textvqa", "default", "validation"),
             ("ScienceQA-IMG", "lmms-lab/ScienceQA", "ScienceQA-IMG", "test")]
    out = []
    for tag, repo, cfg, split in specs:
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True)
            rows = []
            for ex in ds:
                img, q = ex.get("image"), ex.get("question") or ""
                if img is None or not q:
                    continue
                a = ex.get("answer")
                if isinstance(a, list):
                    gold = [norm(x) for x in a if x]
                elif isinstance(a, int) and ex.get("choices"):
                    gold = [norm(ex["choices"][a])]
                elif a is not None:
                    gold = [norm(a)]
                else:
                    continue
                if not gold or not any(gold):
                    continue
                rows.append({"image": img, "q": q, "gold": gold})
                if len(rows) >= N_PER:
                    break
            if rows:
                out.append((tag, rows)); print(f"  {tag}: {len(rows)}", flush=True)
        except Exception as e:
            print(f"  {tag}: SKIP {type(e).__name__}: {str(e)[:60]}", flush=True)
    return out


def main():
    benches = load_benches()
    if not benches:
        print("no benchmarks loaded"); return
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    itid = model.config.image_token_id
    layers = model.model.language_model.layers

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

    def gen(inp, drop=None):
        clear()
        if drop is not None and len(drop):
            n = inp["input_ids"].shape[1]
            b = torch.zeros(n, device=model.device)
            b[torch.as_tensor(sorted(drop), device=model.device)] = -1e4
            for l in layers:
                l.self_attn._mask_bias = b
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=8, do_sample=False,
                               pad_token_id=pr.tokenizer.eos_token_id)
        clear()
        return norm(pr.tokenizer.decode(o[0, inp["input_ids"].shape[1]:],
                                        skip_special_tokens=True))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["bench"], r["idx"]))
    print(f"resuming: {len(done)}", flush=True)
    rng = random.Random(89)
    n, t0 = 0, time.time()
    for tag, rows in benches:
        for i, ex in enumerate(rows):
            if (tag, i) in done:
                continue
            try:
                inp = build(fit(ex["image"].convert("RGB"), B0),
                            ex["q"] + "\nAnswer the question using a single word or phrase."
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
                       "pred": {"none": gen(inp)}}
                for nm, sc in [("layer2", A[FASTV_K]), ("late", A[BLOCK].mean(0)),
                               ("rand", np.array([rng.random() for _ in range(ntok)]))]:
                    rec["pred"][nm] = gen(inp, (base + np.argsort(-sc)[k:]).tolist())
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n += 1
                if n % 25 == 0:
                    print(f"  [{tag} {i+1}/{len(rows)}] {n/(time.time()-t0):.2f} it/s", flush=True)
            except Exception as e:
                print(f"  {tag}[{i}] {type(e).__name__}: {str(e)[:50]}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
