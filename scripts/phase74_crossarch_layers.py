"""
Phase 74: does the LAYER-CONTRAST finding hold on other architectures?

WHAT SS14F ESTABLISHED ON Qwen3-VL-2B
--------------------------------------
    deployed block mean            39.3%   top-1 evidence coverage
    best single layer (L17)        41.9%   -- so the mean is not merely the wrong layers
    learned LINEAR reweighting     45.5%   -- and the learned weights have BOTH SIGNS
    full head                      52.9%
The final layer's gt_pct is 0.529, WORSE than the 0.500 chance level: it is anti-correlated with the
target and the deployed read-out adds it in at +1 like every other layer. The claim is that a mean
can only add, so it cancels layers that point at distractors.

If that is a property of VLM attention read-outs it must replicate. If it is a Qwen3-VL-2B quirk,
the paper's mechanism section is one checkpoint deep and must say so. Both outcomes are worth the
run, which is why it is worth doing before writing the section.

MODELS: Qwen2-VL-7B-Instruct and LLaVA-OneVision-7B -- a different generation of the same family,
and a different family entirely (different tokenisation, different merge, explicit image_newline).

WHAT IS MEASURED (identical pipeline to SS14F, nothing refit)
    per-layer gt_pct and top-1 coverage
    the deployed block mean, rescaled to each model's depth
    the best single layer
    a learned LINEAR reweighting, out-of-fold and grouped by item
    whether the learned weights take both signs, and whether the LAST layer is negative

Only the V*Bench attention maps are extracted here; all analysis reuses phase73's code path so the
two cannot drift.
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODELS = ["Qwen/Qwen2-VL-7B-Instruct", "llava-hf/llava-onevision-qwen2-7b-ov-hf"]
B0 = 300
Image.MAX_IMAGE_PIXELS = None


def extract(model_id, out_path):
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    if os.path.exists(out_path) and sum(1 for _ in open(out_path)) >= 150:
        print(f"  {out_path} already has data; skipping extraction", flush=True)
        return
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(model_id)
    itid = model.config.image_token_id
    MS = getattr(pr.image_processor, "merge_size", 1)
    nL = model.config.text_config.num_hidden_layers
    print(f"  {model_id}: {nL} layers, merge {MS}", flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        i = build(img, "x")
        if "image_grid_thw" in i:
            return int(sum(g[1] * g[2] // (MS * MS) for g in i["image_grid_thw"].tolist()))
        return int((i["input_ids"][0] == itid).sum())

    def fit(img, target, refine=6, tol=0.08):
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

    n, t0 = 0, time.time()
    with open(out_path, "a") as fout:
        for ex in ds:
            ap = os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json"
            if not os.path.exists(ap):
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(os.path.join(root, ex["image"])).convert("RGB")
            IW, IH = img.size
            gt = [min(b[0] for b in ann["bbox"]) / IW, min(b[1] for b in ann["bbox"]) / IH,
                  max(b[0] + b[2] for b in ann["bbox"]) / IW,
                  max(b[1] + b[3] for b in ann["bbox"]) / IH]
            small, _ = fit(img, B0)
            inp = build(small, ex["text"])
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            if len(pos) < 16:
                continue
            base, ntok = int(pos[0].item()), int(len(pos))
            if "image_grid_thw" in inp:
                g = inp["image_grid_thw"][0].tolist()
                gh, gw = g[1] // MS, g[2] // MS
            else:
                gh = gw = int(round(ntok ** 0.5))
            if gh * gw != ntok:
                continue                       # separator tokens present; skip rather than guess
            with torch.no_grad():
                out = model(**inp.to(model.device), output_attentions=True)
            A = np.stack([out.attentions[L][0, :, -1, base:base + ntok].float().mean(0).cpu().numpy()
                          for L in range(len(out.attentions))])
            del out
            torch.cuda.empty_cache()
            if not np.isfinite(A).all():
                continue
            fout.write(json.dumps({
                "question_id_full": f"{ex['category']}/{ex['question_id']}",
                "category": ex["category"], "grid": [gh, gw], "n_img_tokens": ntok,
                "gt_box_frac": gt,
                "attn": {f"L{i}": [round(float(v), 8) for v in A[i]] for i in range(A.shape[0])}
            }) + "\n")
            fout.flush()
            n += 1
            if n % 40 == 0:
                print(f"    [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    del model
    torch.cuda.empty_cache()
    print(f"  wrote {n} items -> {out_path}", flush=True)


if __name__ == "__main__":
    for mid in MODELS:
        tag = mid.split("/")[-1].replace("-", "_")
        print(f"\n=== {mid} ===", flush=True)
        try:
            extract(mid, f"/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase74_{tag}.jsonl")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print("\nextraction done", flush=True)
