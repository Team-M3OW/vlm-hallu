"""
Phase 85: produce FIGURE DATA FROM ACTUAL INFERENCE, not from illustration.

Every panel the paper shows should be something the model really did on a real image:
what it attended to, which tokens survived pruning, what crop it was actually fed, and what it
actually answered. This script runs all of that and saves it, so the figure code draws measurements
rather than drawings.

FOR EACH SELECTED ITEM
    attention          the full per-layer map from one real forward pass at B0=300
    pruning masks      the exact token sets kept by layer-2 ranking and by a late read-out at
                       10% keep -- the same sets SS14L scored
    crop images        the actual pixels handed to the model for each proposal, refit to B0
    answers            the model's real prediction and confidence for: full image, layer-2-pruned,
                       late-pruned, standard-read-out crop, learned-read-out crop
Nothing is recomputed at figure time and nothing is hand-placed.

Items are chosen to span outcomes rather than to flatter: the selection requires that the standard
read-out MISSES and the learned one COVERS, which is the contrast the paper claims -- but the
answers are whatever the model gives, including where our own arm is wrong.
"""
import json
import os
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROP = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71a_head_proposals.json"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase85_figure_inference.json"
IMGDIR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/paper/figs/crops"
BLOCK, FASTV_K, B0, W, KEEP = list(range(16, 27)), 2, 300, 0.15, 0.10
N_ITEMS = 14
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


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    os.makedirs(IMGDIR, exist_ok=True)
    P = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    E = {f"{e['category']}/{e['question_id']}": e for e in ds}
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok, itid = pr.tokenizer, model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
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

    def ask(img, text, drop=None):
        clear()
        inp = build(img, text).to(model.device)
        if drop is not None and len(drop):
            n = inp["input_ids"].shape[1]
            b = torch.zeros(n, device=model.device)
            b[torch.as_tensor(sorted(drop), device=model.device)] = -1e4
            for l in layers:
                l.self_attn._mask_bias = b
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        clear()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt_ids]), 0)
        p = [round(float(v), 4) for v in p.tolist()]
        return {"probs": p, "pred": "ABCD"[int(np.argmax(p))], "conf": round(max(p), 3)}

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    picks = [q for q in P if P[q]["head_cov"] >= .5 > P[q]["argmax_cov"]][:N_ITEMS]  # pool; each figure selects from it by a rule stated in its caption
    print(f"{len(picks)} items selected (standard read-out misses, learned covers)", flush=True)
    out = []
    for qid in picks:
        ex, d = E[qid], P[qid]
        img = Image.open(os.path.join(root, ex["image"])).convert("RGB")
        gold = "ABCD"[ex["label"]] if isinstance(ex["label"], int) else str(ex["label"])
        small = fit(img, B0)
        inp = build(small, ex["text"]).to(model.device)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        pos = (inp["input_ids"][0] == itid).nonzero().flatten()
        base, n = int(pos[0].item()), int(len(pos))
        clear()
        with torch.no_grad():
            o = model(**inp, output_attentions=True)
        A = np.stack([o.attentions[L][0, :, -1, base:base + n].float().mean(0).cpu().numpy()
                      for L in range(len(o.attentions))])
        del o; torch.cuda.empty_cache()
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)

        k = max(1, int(round(KEEP * n)))
        rank = {"layer2": A[FASTV_K], "late": A[BLOCK].mean(0)}
        masks, ans = {}, {"full": ask(small, ex["text"])}
        for nm, s in rank.items():
            keep = np.argsort(-s)[:k]
            masks[nm] = sorted(int(x) for x in keep)
            ans[f"pruned_{nm}"] = ask(small, ex["text"], drop=(base + np.argsort(-s)[k:]).tolist())

        crops = {}
        for nm, key in [("standard", "argmax"), ("learned", "head")]:
            c = fit(window(img, *d[key], W), B0)
            fn = f"{qid.replace('/', '_')}_{nm}.png"
            c.save(os.path.join(IMGDIR, fn))
            crops[nm] = fn
            ans[f"crop_{nm}"] = ask(c, ex["text"])
        img.resize((min(img.size[0], 900), int(img.size[1] * min(1, 900 / img.size[0])))
                   ).save(os.path.join(IMGDIR, f"{qid.replace('/', '_')}_full.png"))

        out.append({"qid": qid, "question": ex["text"], "gold": gold,
                    "grid": [gh, gw], "n_tokens": n, "keep_k": k,
                    "gt_box_frac": d["gt_box_frac"], "img_wh": list(img.size),
                    "head": d["head"], "argmax": d["argmax"],
                    "attn_late": [round(float(v), 7) for v in A[BLOCK].mean(0)],
                    "attn_layer2": [round(float(v), 7) for v in A[FASTV_K]],
                    "masks": masks, "crops": crops, "answers": ans})
        a = out[-1]["answers"]
        print(f"  {qid}: gold {gold} | full {a['full']['pred']} | "
              f"prune-L2 {a['pruned_layer2']['pred']} | prune-late {a['pruned_late']['pred']} | "
              f"crop-std {a['crop_standard']['pred']} | crop-ours {a['crop_learned']['pred']}",
              flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} and {IMGDIR}", flush=True)


if __name__ == "__main__":
    QM.eager_attention_forward = patched
    main()
