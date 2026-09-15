"""
Phase 68 (probe): is there ANY sub-patch signal to ensemble? Cheap, decisive, run before any method.

THE IDEA UNDER TEST
-------------------
A sub-token target (§13B: below ~0.2 merged tokens) sits below the Nyquist limit of the ViT patch
grid, so it is averaged into one patch and its detail aliases away. If the image is shifted by a
FRACTION OF A PATCH, the target falls on a different patch boundary and a different projection of it
survives -- so ensembling over offsets might recover sub-patch detail, the way multi-frame
super-resolution does. Non-crop, training-free.

WHAT THIS PROBE DECIDES, BEFORE ANY METHOD IS BUILT
---------------------------------------------------
    answers are IDENTICAL across offsets -> the encoder is effectively shift-invariant at sub-patch
        scale, there is no independent information to ensemble, and the idea is DEAD. No further
        run needed.
    answers VARY across offsets          -> there is variance. Then the question is whether it is
        SIGNAL (ensembling raises accuracy) or NOISE (ensembling does nothing / hurts), which the
        same data answers.

Crucially the variance test is cheap and conclusive in the dead direction, so it is run first.
Budget accounting is deferred: ensembling k offsets costs k passes, so any method would face a
uniform@300k bar. That only matters if the probe survives.

DESIGN
------
Shift by fractions of one MERGED-TOKEN cell (Qwen merges 2x2 patches of 16px -> a 32px cell at the
resized scale). Offsets are applied by padding+cropping the resized image so the CONTENT is
unchanged and only its alignment to the patch grid moves. Realized token count is held fixed and
asserted, so this is not a budget change in disguise.
"""
import json
import os
import statistics as st
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase68_jitter.jsonl"
B0 = 300
OFFSETS = [(0, 0), (8, 0), (16, 0), (0, 8), (0, 16), (8, 8), (16, 16), (24, 24)]
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]

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

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    def shift(img, dx, dy):
        """Move the CONTENT relative to the patch grid, holding the canvas size fixed.
        Pad by (dx,dy) using edge replication, then crop back to the original size."""
        w, h = img.size
        if dx == 0 and dy == 0:
            return img
        canvas = Image.new("RGB", (w + dx, h + dy))
        canvas.paste(img.crop((0, 0, min(dx, w), h)).resize((dx, h)) if dx else img, (0, 0))
        canvas.paste(img, (dx, dy))
        return canvas.crop((0, 0, w, h))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)};  offsets={OFFSETS} (merged cell = 32px at the resized scale)",
          flush=True)

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
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            area = ((gx1 - gx0) * (gy1 - gy0)) / (IW * IH)
            text = ex["text"]
            base = fit(img, B0)
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str)
                   else int(ex["label"]),
                   "tokens_on_target": area * B0, "probs": {}, "tokens": {}}
            for dx, dy in OFFSETS:
                p, rz = answer(shift(base, dx, dy), text)
                rec["probs"][f"{dx},{dy}"] = p
                rec["tokens"][f"{dx},{dy}"] = rz
            # the budget must NOT move -- otherwise this is an upscale in disguise
            ts = list(rec["tokens"].values())
            assert max(ts) - min(ts) <= 0.05 * min(ts), f"token count moved: {ts}"
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
