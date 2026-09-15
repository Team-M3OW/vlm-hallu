"""
Phase 50: SUPPRESS THE SERIALIZATION SINK INSIDE THE FORWARD PASS. One pass, same budget, no crop.

WHY THIS IS THE RIGHT METHOD AND THE EARLIER ONES WERE NOT
----------------------------------------------------------
Every method so far reallocated PIXELS -- crop, zoom, re-rank candidates -- and every one of them
paid for it in forward passes. Phase 47 showed why that is a losing trade: the uniform curve rises
56.5 -> 63.9 -> 71.7% at 1x/2x/4x, so any method spending k passes must beat uniform@300k, and no
published policy does. The compute bar, not the idea, is what kills crop-based allocation.

This intervention does not touch pixels and does not add a pass. §7A established that 25% of the
last token's attention over image tokens lands on the TRAILING ROW BOUNDARY of the raster
serialization -- the last column, at 3.4-4.5x its fair share, on four architectures, present from
L0, and sitting on the learned `image_newline` in models that have one. That mass is spent on a
position that carries no image content. Ring-masking removes it from our READ-OUT, but the model
itself still spends it.

So: suppress it IN THE FORWARD PASS with an additive bias on the attention logits of those key
positions, let softmax redistribute the freed mass over the remaining tokens, and measure accuracy.
Cost is IDENTICAL to baseline -- one pass, same realized tokens -- so this is a pure paired
comparison with no compute bar to clear. That is the whole point.

CONTROLS, because "ablating tokens changes the answer" is not a finding
----------------------------------------------------------------------
    rand_cols     bias the SAME NUMBER of randomly chosen image tokens. If sink suppression only
                  works as well as this, the effect is generic perturbation, not the sink.
    first_col     bias the FIRST column -- enriched on Qwen3-VL (2.0x) but NOT universal (0.9x on
                  Qwen2-VL, §7C). A different prediction for a different position.
    interior      bias an equal-sized block of INTERIOR columns: real image content, no sink.
                  If this helps as much, the account is wrong.
    text_tokens   bias an equal number of PROMPT tokens: tests whether any mass redistribution helps.

A SWEEP, NOT A TUNED POINT: bias strength b is swept and the whole curve is reported, so a single
flattering value cannot be selected after the fact.
"""
import json
import os
import random
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase50_sink_steering.jsonl"
B0 = 300
BIASES = [-1.0, -2.0, -4.0, -8.0]
Image.MAX_IMAGE_PIXELS = None

# ---- patch eager attention to accept a per-module additive bias over KEY positions
_orig = QM.eager_attention_forward


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states = QM.repeat_kv(key, module.num_key_value_groups)
    value_states = QM.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    b = getattr(module, "_sink_bias", None)
    if b is not None:                       # (T,) additive, applied to every query row
        attn_weights = attn_weights + b.to(attn_weights.dtype).view(1, 1, 1, -1)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1,
                                               dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    out = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return out, attn_weights


QM.eager_attention_forward = patched


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(50)
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
    attns = [l.self_attn for l in model.model.language_model.layers]
    print(f"{len(attns)} attention modules patched", flush=True)

    def set_bias(v):
        for a in attns:
            a._sink_bias = v

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    def run(inp, bias_vec):
        set_bias(bias_vec)
        try:
            with torch.no_grad():
                lg = model(**inp).logits[0, -1].float()
        finally:
            set_bias(None)
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    # ---- self-test: the patch must be inert at bias 0 and must bite at bias<0
    _img = fit(Image.new("RGB", (640, 480), (120, 130, 140)), B0)
    _i = build(_img, "What is this?").to(model.device)
    a = run(_i, None)
    b = run(_i, torch.zeros(_i["input_ids"].shape[-1], device=model.device))
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-3, "zero bias must be a no-op"
    _pos = (_i["input_ids"][0] == itid).nonzero().flatten()
    v = torch.zeros(_i["input_ids"].shape[-1], device=model.device)
    v[_pos[:20]] = -8.0
    c = run(_i, v)
    assert max(abs(x - y) for x, y in zip(a, c)) > 1e-3, "bias had no effect -- patch not active"
    print(f"  self-test OK: zero-bias no-op, -8 on 20 tokens moves p by "
          f"{max(abs(x-y) for x,y in zip(a,c)):.4f}", flush=True)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            text = ex["text"]
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            small = fit(img, B0)
            inp = build(small, text)
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2
            ids = inp["input_ids"][0].tolist()
            pos = [i for i, v_ in enumerate(ids) if v_ == itid]
            if len(pos) != gh * gw:
                continue
            T = len(ids)
            inp = inp.to(model.device)
            dev = model.device

            def vec(idxs, b):
                v = torch.zeros(T, device=dev)
                v[list(idxs)] = b
                return v

            last_col = [pos[r * gw + (gw - 1)] for r in range(gh)]
            first_col = [pos[r * gw + 0] for r in range(gh)]
            mid = gw // 2
            interior = [pos[r * gw + mid] for r in range(gh)]
            k = len(last_col)
            rand_cols = rng.sample(pos, k)
            text_pos = [i for i in range(T) if i not in set(pos)]
            text_tok = rng.sample(text_pos, min(k, len(text_pos)))

            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "grid": [gh, gw], "n_biased": k, "n_img": len(pos),
                   "realized_tokens": len(pos), "probs": {}}
            rec["probs"]["baseline"] = run(inp, None)
            for nm, idxs in [("last_col", last_col), ("first_col", first_col),
                             ("interior", interior), ("rand_cols", rand_cols),
                             ("text_tokens", text_tok)]:
                for b in BIASES:
                    rec["probs"][f"{nm}@{b}"] = run(inp, vec(idxs, b))
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
