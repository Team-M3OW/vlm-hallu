"""
Phase 76: THE GATE FOR SINGLE-PASS FOVEATION. Can the VISION TOWER localise on its own?

WHY THIS DECIDES THE ARCHITECTURE DIRECTION
-------------------------------------------
SS14D/E give a working re-ranker, but it reads the LANGUAGE MODEL's decoder attention, which needs a
full LM forward. So any method built on it is inherently two-pass -- look, then crop and look again --
and SS2 showed that is exactly why crop policies lose to simply spending the budget.

The vision tower runs BEFORE the LM. If its own attention localises the target, resolution can be
allocated non-uniformly at ENCODE time and the whole thing costs:

    2 vision encodes + 1 LM forward        (foveated)
    2 vision encodes + 2 LM forwards       (crop-and-re-encode, what everyone does)

On a 2B model the LM forward dominates, so this removes the cost that has sunk every allocation
method we have measured. It is also NOT cropping -- the image is never cut, only sampled unevenly --
which matters given the standing constraint against preprocessing.

    vision-tower attention localises      -> single-pass foveation is buildable; proceed.
    it does not                           -> the direction is dead at its first step and no
                                             engineering should be spent on it. Report and stop.

WHAT IS MEASURED
----------------
Qwen3-VL-2B's vision tower is 24 layers, 16 heads, patch 16, spatial merge 2. There is no CLS token,
so per-patch salience is the attention each patch RECEIVES, averaged over queries and heads -- the
standard substitute, and the same quantity ram's Vision-CLS approximates with a class token.

For every vision layer we record the full per-patch map, so the SS14F analysis (best single layer,
block mean, learned signed combination, does the last layer anti-correlate) runs unchanged on the
vision tower. That also answers option 3 for the vision side: whether depth disagreement is a
property of transformers generally or of the LM read-out specifically.

Heads are kept SEPARATE for the first 191 items so the head-level version of the same question can
be asked without a second extraction.
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

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase76_vision_tower.jsonl"
B0 = 300
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    visual = model.model.visual
    MS = model.config.vision_config.spatial_merge_size
    print(f"vision tower: {len(visual.blocks)} blocks, merge {MS}", flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // (MS * MS)
                       for g in build(img, "x")["image_grid_thw"].tolist()))

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

    # The vision block does `hidden_states + self.attn(...)` and returns ONE tensor, so
    # output_attentions never reaches it and a forward hook captures nothing. The weights exist
    # only inside the module-level attention function, so patch that -- the technique SS10A/SS10B
    # used on the LM -- and REDUCE inside the patch: raw vision weights are ~1200x1200x16 per
    # layer, ~2GB over 24 layers, which will not fit alongside the model.
    import transformers.models.qwen3_vl.modeling_qwen3_vl as QM
    CAP = {"maps": [], "on": False}
    _orig_eager = QM.eager_attention_forward

    def cap_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = _orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kw)
        if CAP["on"] and type(module).__name__ == "Qwen3VLVisionAttention" and w is not None:
            t = w.detach().float()
            while t.dim() > 3:
                t = t[0]
            CAP["maps"].append(t.mean(0).mean(0).cpu().numpy())   # over heads, then queries
        return out, w

    QM.eager_attention_forward = cap_eager

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0, failed = 0, time.time(), 0
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if qid in done or not os.path.exists(ap):
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            IW, IH = img.size
            gt = [min(b[0] for b in ann["bbox"]) / IW, min(b[1] for b in ann["bbox"]) / IH,
                  max(b[0] + b[2] for b in ann["bbox"]) / IW,
                  max(b[1] + b[3] for b in ann["bbox"]) / IH]
            small, _ = fit(img, B0)
            inp = build(small, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            PH, PW = g[1], g[2]                       # pre-merge patch grid
            gh, gw = PH // MS, PW // MS               # merged grid (matches LM tokens)
            CAP["maps"].clear(); CAP["on"] = True
            with torch.no_grad():
                model(**inp.to(model.device))
            CAP["on"] = False
            if not CAP["maps"]:
                failed += 1
                if failed <= 2:
                    print("  !! no vision attention captured -- hook signature mismatch", flush=True)
                if failed >= 5:
                    print("  aborting: vision attention is not exposed on this model", flush=True)
                    break
                continue
            maps = {}
            for li, recv in enumerate(CAP["maps"]):
                if recv.shape[-1] != PH * PW:
                    maps = {}
                    break
                # pool the pre-merge patch grid down to the merged grid the LM actually sees
                r = recv.reshape(gh, MS, gw, MS).mean(axis=(1, 3))
                maps[f"L{li}"] = [round(float(v), 8) for v in r.flatten()]
            CAP["maps"].clear()
            torch.cuda.empty_cache()
            if not maps:
                failed += 1
                continue
            fout.write(json.dumps({
                "question_id_full": qid, "category": ex["category"],
                "grid": [gh, gw], "n_img_tokens": gh * gw, "patch_grid": [PH, PW],
                "gt_box_frac": gt, "attn": maps}) + "\n")
            fout.flush()
            n += 1
            if n % 40 == 0:
                print(f"  [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    QM.eager_attention_forward = _orig_eager
    print(f"Done. {n} items, {failed} failures -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
