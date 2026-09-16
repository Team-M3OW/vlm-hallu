"""
Phase 87: is the pruning collapse specific to LAYER 2, or is it a DEPTH EFFECT?

THE OBJECTION THIS ANSWERS
--------------------------
"The critique targets FastV's default depth specifically; it would strengthen the paper to test
whether more recent pruning methods (post-FastV) already avoid layer-2 reads, to clarify whether
this is a live problem in current best-practice or mainly historical."

We cannot run every successor method, but we can answer the question underneath it: prune by
attention read at EVERY depth and report the whole curve. That says which depths are safe and which
are not, so any method -- existing or future -- can be placed on it by its read depth alone. It also
distinguishes two possibilities the single-K comparison cannot:

    (a) layer 2 is anomalously bad (a quirk of that layer)
    (b) pruning damage falls monotonically with read depth (a property of the stack)

(b) would make this a live design constraint rather than a historical criticism of one default.

PRE-REGISTERED: if damage is monotone in depth, a method reading anywhere in the first third is
exposed, and "read late" is a general prescription. If only K=2 is bad, the finding is narrower and
we say so.

ARMS: rank by attention at layer K for K in {0,1,2,4,6,8,10,12,14,16,20,24,26}, plus the deployed
late block and a random control, all at 10% keep and all pruning the SAME token count.
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase87_prune_depth.jsonl"
BLOCK = list(range(16, 27))
KS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 26]
B0, KEEP = 300, 0.10
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

    def ask(inp, drop=None):
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
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["qid"])
    print(f"resuming: {len(done)}  |  sweeping K over {KS}", flush=True)
    rng = random.Random(87)
    n, t0 = 0, time.time()
    for ex in ds:
        qid = f"{ex['category']}/{ex['question_id']}"
        ip = os.path.join(root, ex["image"])
        if qid in done or not os.path.exists(ip):
            continue
        img = Image.open(ip).convert("RGB")
        inp = build(fit(img, B0), ex["text"]).to(model.device)
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
        rec = {"qid": qid, "category": ex["category"], "n_img": ntok,
               "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
               "probs": {"none": ask(inp)}}
        for K in KS:
            rec["probs"][f"L{K}"] = ask(inp, (base + np.argsort(-A[K])[k:]).tolist())
        rec["probs"]["late_block"] = ask(inp, (base + np.argsort(-A[BLOCK].mean(0))[k:]).tolist())
        rs = np.array([rng.random() for _ in range(ntok)])
        rec["probs"]["rand"] = ask(inp, (base + np.argsort(-rs)[k:]).tolist())
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 20 == 0:
            print(f"  [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
