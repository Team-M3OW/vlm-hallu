"""
Phase 51: ALLOCATE IN ATTENTION SPACE INSTEAD OF PIXEL SPACE. One pass, same budget, no crop.

THE QUESTION
------------
The paper's finding is that moving visual tokens to the evidence beats adding tokens, by >=26x. All
of that was done in PIXEL space -- crop, refit, re-run -- and Phase 47 showed why pixel-space
allocation struggles to pay: it costs forward passes, and the uniform curve (56.5 -> 63.9 -> 71.7%
at 1x/2x/4x) prices out anything spending more than about one extra pass.

The internal analogue costs nothing: rather than giving the evidence more PIXELS, give its tokens
more ATTENTION WEIGHT. Same image, same realized tokens, one pass, additive bias on the attention
logits of the cells covering the target.

Phase 50 tested the destructive half of this (suppress the sink) and found nothing: biasing away
the trailing row boundary is indistinguishable from biasing away random or interior image tokens,
while biasing away prompt tokens costs ~19pp. So image-token attention is not a scarce resource in
the direction of removal. This file tests the constructive direction.

THE DECISIVE ARM IS THE ORACLE
------------------------------
    amp_oracle   amplify exactly the cells whose centres fall inside the GT box.
                 This is the attention-space counterpart of the oracle CROP, which scores 92.7%
                 against uniform's 56.5% at the same 300 tokens.

    amp_oracle approaches the oracle crop -> attention-space allocation WORKS. The method becomes
                 one pass at B0 with no crop, the compute bar collapses to uniform@300, and the
                 remaining problem is localisation, which we have already characterised.
    amp_oracle ~= baseline                -> allocation MUST happen in pixel space; attention
                 reweighting cannot substitute for resolution. That is a clean, strong negative and
                 it explains why every method in this literature crops: the information simply is
                 not in the token embeddings at low resolution, so no amount of attention recovers
                 it. It would also retire the whole attention-steering direction with evidence.

Either outcome is decisive, which is why this is worth the run.

ARMS (all one pass, identical realized tokens)
    baseline                     no intervention
    amp_oracle@+b                amplify GT-box cells                       (ceiling)
    amp_top{1,5,15}@+b           amplify the top-k cells of the ring-masked attention map (method)
    amp_rand@+b                  amplify a random equal-sized cell set      (control)
    amp_win@+b                   amplify the cells inside the deployed W=0.15 window at the peak
Bias strength is SWEPT; the whole curve is reported.
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
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase51_attn_alloc.jsonl"
BLOCK = list(range(16, 27))
B0, W = 300, 0.15
AMPS = [1.0, 2.0, 4.0, 8.0]
Image.MAX_IMAGE_PIXELS = None


def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    key_states = QM.repeat_kv(key, module.num_key_value_groups)
    value_states = QM.repeat_kv(value, module.num_key_value_groups)
    aw = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask
    b = getattr(module, "_sink_bias", None)
    if b is not None:
        aw = aw + b.to(aw.dtype).view(1, 1, 1, -1)
    aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    aw = torch.nn.functional.dropout(aw, p=dropout, training=module.training)
    return torch.matmul(aw, value_states).transpose(1, 2).contiguous(), aw


QM.eager_attention_forward = patched


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(51)
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

    def logits(inp, v):
        set_bias(v)
        try:
            with torch.no_grad():
                lg = model(**inp).logits[0, -1].float()
        finally:
            set_bias(None)
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt_ids]), 0)
        return [round(float(x), 6) for x in p.tolist()]

    def attnmap(inp, gh, gw, base):
        n_img = gh * gw
        set_bias(None)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        torch.cuda.empty_cache()
        m = (acc / len(BLOCK)).reshape(gh, gw)
        o = torch.full_like(m, -1.0)
        if gh > 2 and gw > 2:
            o[1:-1, 1:-1] = m[1:-1, 1:-1]
        else:
            o = m.clone()
        return o.flatten()

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
            IW, IH = img.size
            gt = [min(b[0] for b in ann["bbox"]) / IW, min(b[1] for b in ann["bbox"]) / IH,
                  max(b[0] + b[2] for b in ann["bbox"]) / IW,
                  max(b[1] + b[3] for b in ann["bbox"]) / IH]
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
            base_tok = pos[0]
            f = attnmap(inp, gh, gw, base_tok)

            def cells(pred):
                return [pos[r * gw + c] for r in range(gh) for c in range(gw)
                        if pred(((c + .5) / gw), ((r + .5) / gh))]

            oracle = cells(lambda x, y: gt[0] <= x <= gt[2] and gt[1] <= y <= gt[3])
            if not oracle:
                cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
                oracle = [pos[min(gh - 1, int(cy * gh)) * gw + min(gw - 1, int(cx * gw))]]
            order = torch.argsort(f, descending=True).tolist()
            top = {k: [pos[i] for i in order[:k]] for k in (1, 5, 15)}
            j = order[0]
            px, py = ((j % gw) + .5) / gw, ((j // gw) + .5) / gh
            win = cells(lambda x, y: abs(x - px) <= W / 2 and abs(y - py) <= W / 2) or [pos[j]]
            rnd = rng.sample(pos, len(oracle))

            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "grid": [gh, gw], "n_oracle": len(oracle), "n_img": len(pos),
                   "realized_tokens": len(pos), "probs": {}}
            rec["probs"]["baseline"] = logits(inp, None)

            def vec(idxs, b):
                v = torch.zeros(T, device=dev)
                v[list(idxs)] = b
                return v

            for nm, idxs in [("amp_oracle", oracle), ("amp_top1", top[1]), ("amp_top5", top[5]),
                             ("amp_top15", top[15]), ("amp_win", win), ("amp_rand", rnd)]:
                for b in AMPS:
                    rec["probs"][f"{nm}@{b}"] = logits(inp, vec(idxs, b))
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
