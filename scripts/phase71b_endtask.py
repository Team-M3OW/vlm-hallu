"""
Phase 71b: THE END-TASK MEASUREMENT. Does better proposal quality become better accuracy?

Phase 70 raised top-1 coverage 39.3% -> 52.9% out-of-fold. Coverage is a proposal-quality metric,
and SS14C stated plainly that the accuracy gain was unmeasured. This measures it.

THE PREDICTION, FIXED BEFORE THE RUN
------------------------------------
SS6D measured the two coverage strata: a window that MISSES costs -15.6pp [-26.0,-5.2], one that
fully covers gains +37.1pp [+24.2,+50.0] -- a 52.7pp swing. Moving 13.6% of items across that
boundary predicts

        0.136 x 52.7 = +7.2pp   for head@0.15 over argmax@0.15.

A result far below that means coverage is not the mediator SS6D claims; far above means something
other than coverage changed. Both are informative, which is why the number is written down first.

THE BAR
-------
The head is FREE: it re-ranks attention that pass 1 already computed, so head and argmax cost
exactly the same -- one localisation pass at B0 plus one crop pass at B0. The honest bar is
therefore uniform@600, not uniform@300, and both are run in-line on the same items rather than
quoted from an earlier phase. Token counts are MEASURED from image_grid_thw, never computed; any
arm drifting >10% from its target voids its contrast.

ARMS
    uniform@300        B0 baseline
    uniform@600        COMPUTE-MATCHED BAR for every two-pass arm
    argmax@0.15        the deployed allocator (incumbent)
    head@0.15          the same allocator with the Phase 70 head's OOF proposal   <- the method
    rand@0.15          random centre, same W -- the control that licensed the original claim
    oracle@0.15        window centred on the GT box -- the ceiling AT THIS WINDOW SIZE

Proposals are read from phase71a (out-of-fold, grouped by item). No fitting happens in this file.
Loaded in bfloat16: a bf16 checkpoint in fp16 gave all-NaN logits that argmax'd to index 0 and read
as a clean negative (SS8). Logits are asserted finite before anything is written.
"""
import json
import os
import random
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROP = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71a_head_proposals.json"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71b_endtask.jsonl"
B0, W = 300, 0.15
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(PROP))
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

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).all(), "non-finite logits -- check dtype"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            if qid in done or qid not in props:
                continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip):
                continue
            pp = props[qid]
            img = Image.open(ip).convert("RGB")
            gt = pp["gt_box_frac"]
            gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
            rng = random.Random(7100 + hash(qid) % 100000)
            rcx, rcy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]

            arms = {
                "uniform@300": fit(img, B0)[0],
                "uniform@600": fit(img, 2 * B0)[0],
                "argmax@0.15": fit(window(img, *pp["argmax"], W), B0)[0],
                "head@0.15":   fit(window(img, *pp["head"], W), B0)[0],
                "rand@0.15":   fit(window(img, rcx, rcy, W), B0)[0],
                "oracle@0.15": fit(window(img, gcx, gcy, W), B0)[0],
            }
            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "head_cov": pp["head_cov"], "argmax_cov": pp["argmax_cov"],
                   "probs": {}, "realized_tokens": {}}
            for nm, im in arms.items():
                p, rz = answer(im, text)
                rec["probs"][nm] = p
                rec["realized_tokens"][nm] = rz
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
