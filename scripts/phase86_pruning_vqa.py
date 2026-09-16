"""
Phase 86: does the layer-2 pruning collapse hold on the benchmarks the PRUNING LITERATURE uses?

THE OBJECTION THIS ANSWERS
--------------------------
SS14L measured the pruning result on V*Bench, which is a SMALL-OBJECT SEARCH benchmark: the answer
hinges on one tiny region, so question-conditioned late-layer attention is exactly what matters and
early-layer attention is exactly what should fail. FastV and its successors are evaluated on general
VQA -- GQA, TextVQA, POPE, ScienceQA -- where the answer often does NOT depend on locating one small
thing, and where early-layer attention may be perfectly adequate for deciding which tokens are
redundant.

So the claim "the field prunes at the wrong depth" is currently supported only on the benchmark most
likely to flatter it. If the effect vanishes on general VQA, the honest claim shrinks to "on
small-object search, prune late" -- which is a much narrower contribution and must be stated as such.

PRE-REGISTERED READING, fixed before the run
    layer-2 collapses on >=2 of these benchmarks   -> the claim is about pruning read-outs generally
    layer-2 is fine on general VQA                 -> the claim is SCOPED to small-object search and
                                                      the abstract must be rewritten
    mixed                                          -> report per-benchmark; no general claim

ARMS (identical to SS14L; every arm prunes the SAME token count so cost is matched by construction)
    none | rand | layerK (FastV default K=2) | blockmean (late) | linear (SS14F weights)

BENCHMARKS -- chosen because the pruning literature evaluates on them, not because they suit us.
POPE is yes/no object hallucination; GQA is compositional; TextVQA needs OCR of small text (so it is
the one where we EXPECT the late read-out to matter); ScienceQA-IMG is diagram reasoning.
Open-ended answers are scored by normalised exact match against the reference set, which is the
standard protocol for these benchmarks.
"""
import json
import os
import random
import re
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase86_pruning_vqa.jsonl"
BLOCK, FASTV_K, B0, KEEP = list(range(16, 27)), 2, 300, [0.10, 0.25]
N_PER = 200
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
    return torch.matmul(a, v).transpose(1, 2).contiguous(), a


def norm(t):
    t = str(t).lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())


def load_benches():
    from datasets import load_dataset
    out = []
    specs = [("POPE", "lmms-lab/POPE", "default", "test"),
             ("GQA", "lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
             ("TextVQA", "lmms-lab/textvqa", "default", "validation"),
             ("ScienceQA-IMG", "lmms-lab/ScienceQA", "ScienceQA-IMG", "test")]
    for tag, repo, cfg, split in specs:
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True)
            rows = []
            for ex in ds:
                img = ex.get("image")
                q = ex.get("question") or ex.get("hint") or ""
                ans = ex.get("answer") or ex.get("answers")
                if img is None or not q:
                    continue
                if isinstance(ans, list):
                    gold = [norm(a) for a in ans if a]
                elif ans is None:
                    ch = ex.get("choices")
                    if ch is None:
                        continue
                    gold = [norm(ch[int(ex["answer"])])] if isinstance(ex.get("answer"), int) else None
                    if gold is None:
                        continue
                else:
                    gold = [norm(ans)]
                if not gold or not any(gold):
                    continue
                rows.append({"bench": tag, "image": img, "q": q, "gold": gold})
                if len(rows) >= N_PER:
                    break
            if rows:
                out.append((tag, rows))
                print(f"  {tag}: {len(rows)} items", flush=True)
        except Exception as e:
            print(f"  {tag}: SKIP ({type(e).__name__}: {str(e)[:70]})", flush=True)
    return out


def main():
    W = np.asarray(json.load(open(WEIGHTS))["weights"], dtype=float)
    print("loading benchmarks...", flush=True)
    benches = load_benches()
    if not benches:
        print("no benchmarks loaded; abort", flush=True)
        return
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
            if hasattr(l.self_attn, "_mask_bias"):
                del l.self_attn._mask_bias

    def gen(inp, drop=None, maxnew=8):
        clear()
        if drop is not None and len(drop):
            n = inp["input_ids"].shape[1]
            b = torch.zeros(n, device=model.device)
            b[torch.as_tensor(sorted(drop), device=model.device)] = -1e4
            for l in layers:
                l.self_attn._mask_bias = b
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=maxnew, do_sample=False,
                               pad_token_id=pr.tokenizer.eos_token_id)
        clear()
        txt = pr.tokenizer.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return norm(txt)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done.add((r["bench"], r["idx"]))
    print(f"resuming: {len(done)}", flush=True)
    rng = random.Random(86)
    t0, n = time.time(), 0
    for tag, rows in benches:
        for i, ex in enumerate(rows):
            if (tag, i) in done:
                continue
            try:
                img = ex["image"].convert("RGB")
                small = fit(img, B0)
                prompt = ex["q"] + "\nAnswer the question using a single word or phrase."
                inp = build(small, prompt).to(model.device)
                pos = (inp["input_ids"][0] == itid).nonzero().flatten()
                base, ntok = int(pos[0].item()), int(len(pos))
                clear()
                with torch.no_grad():
                    o = model(**inp, output_attentions=True)
                A = np.stack([o.attentions[L][0, :, -1, base:base+ntok].float().mean(0).cpu().numpy()
                              for L in range(len(o.attentions))])
                del o; torch.cuda.empty_cache()
                A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
                rank = {"layerK": A[FASTV_K], "blockmean": A[BLOCK].mean(0),
                        "linear": (W[:A.shape[0], None] * A).sum(0),
                        "rand": np.array([rng.random() for _ in range(ntok)])}
                rec = {"bench": tag, "idx": i, "gold": ex["gold"], "n_img": ntok,
                       "pred": {"none": gen(inp)}}
                for kf in KEEP:
                    k = max(1, int(round(kf * ntok)))
                    for nm, s in rank.items():
                        rec["pred"][f"{nm}@{kf}"] = gen(
                            inp, drop=(base + np.argsort(-s)[k:]).tolist())
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n += 1
                if n % 25 == 0:
                    el = time.time() - t0
                    print(f"  [{tag} {i+1}/{len(rows)}] {n/el:.2f} it/s", flush=True)
            except Exception as e:
                print(f"  {tag}[{i}] error {type(e).__name__}: {str(e)[:60]}", flush=True)
                continue
    print("Done.", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
