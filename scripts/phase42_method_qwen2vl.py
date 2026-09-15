"""
Phase 42: does the METHOD and the COVERAGE MECHANISM transfer to another architecture?
Qwen2-VL-7B-Instruct on V*Bench. Nothing is refitted.

WHY THIS RUN
------------
Everything in §3-§5 -- the coverage dose-response, the sink-masked localizer, the confidence gate as
a coverage detector -- is measured on Qwen3-VL-2B. The exchange rate (§2) is multi-architecture, but
the mechanism and the method are not, and that is the main scope limit on the paper.

Qwen2-VL-7B is the right second model: a DIFFERENT generation of the vision stack, 3.5x the
parameters, and a clean `image_grid_thw` so the localizer needs no new bookkeeping. Phase 41 already
shows its sink is columnar too (LAST COL 4.4x, rows 1.3x/0.6x), so the mask is justified here on its
own measured grounds rather than by assumption.

NOTHING IS REFITTED -- that is the point:
    W in {0.15, 0.25}        transferred from Qwen3-VL's V*Bench CV
    relative layer block     [0.55L, 0.95L], the architecture-agnostic form of L16-26 of 28
    outer-ring mask, B0=300  unchanged
Refitting any of these on Qwen2-VL would answer a weaker question.

ARMS (all budget-gated at MEASURED realized tokens)
---------------------------------------------------
    uniform      whole image @ 300                      baseline
    oracle       crop to the padded GT box @ 300        the ceiling, and the coverage=100% case
    attn@W       crop at the ring-masked attention peak
    rand@W       crop of the same size at a random centre -- the placement control
`peak_frac` and the GT box are logged so that COVERAGE, the §3 mediator, can be recomputed offline
on this architecture exactly as it was on Qwen3-VL. The proposal itself never sees a box.

WHAT WOULD FALSIFY THE PAPER'S GENERALITY
-----------------------------------------
    attn ~= rand                      -> the localizer does not transfer; the mechanism is
                                         Qwen3-VL-specific and §4 must say so.
    coverage dose-response absent     -> coverage is not the mediator here, and §3's claim to be
                                         about VLMs rather than about one VLM fails.
    conf(attn) does not predict coverage -> the gate is not a coverage detector in general, and
                                         §5.3's explanation of why the method works is local.
Each of these is checked by phase42_analyze.py against the Qwen3-VL numbers.
"""
import json
import math
import os
import random
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase42_qwen2vl_method.jsonl"
WINDOWS = [0.15, 0.25]
B0 = 300
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(42)
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print(f"Loading {MODEL_ID} (fp16, eager)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    L = model.config.get_text_config().num_hidden_layers
    BLOCK = list(range(int(0.55 * L), int(0.95 * L) + 1))
    MS = getattr(pr.image_processor, "merge_size", 2)
    print(f"  {L} layers; relative block {BLOCK[0]}-{BLOCK[-1]}; merge {MS}", flush=True)
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        i = build(img, "x")
        return int(sum(g[1] * g[2] // (MS * MS) for g in i["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            w, h = max(32, int(W_ * sc)), max(32, int(H_ * sc))
            cur = img.resize((w, h), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best if best else (img, measure(img))

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // (MS * MS) for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        # HARD GUARD. Qwen2-VL-7B is a bfloat16 checkpoint and silently produces all-NaN logits in
        # fp16. The first run of this file did exactly that: every arm returned NaN, argmax fell
        # through to index 0, and since 71/191 labels are "A" every arm scored an identical 37.2%
        # with CI [+0.0,+0.0] -- which the analyzer then read as "the localizer does not transfer".
        # A dtype bug must crash here, not turn into a negative result.
        assert torch.isfinite(lg).any(), "non-finite logits -- wrong dtype for this checkpoint"
        assert torch.isfinite(p).all(), "non-finite answer distribution -- refusing to log NaN"
        return [round(float(v), 6) for v in p.tolist()], rz

    def localize(img, text):
        small, _ = fit(img, B0)
        inp = build(small, text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // MS, g[2] // MS
        n_img = gh * gw
        pos = (inp["input_ids"][0] == itid).nonzero().flatten()
        base = int(pos[0].item())
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        assert torch.isfinite(out.attentions[BLOCK[0]]).all(), "non-finite attention"
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for li in BLOCK:
            a = out.attentions[li][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        torch.cuda.empty_cache()
        acc = (acc / len(BLOCK)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            m = acc.clone()
        i = int(m.argmax().item())
        return ((i % gw) + .5) / gw, ((i // gw) + .5) / gh

    def window(img, cx, cy, W):
        iw, ih = img.size
        x0, y0 = (cx - W / 2) * iw, (cy - W / 2) * ih
        x1, y1 = (cx + W / 2) * iw, (cy + W / 2) * ih
        if x0 < 0: x0, x1 = 0, W * iw
        if y0 < 0: y0, y1 = 0, W * ih
        if x1 > iw: x0, x1 = iw - W * iw, iw
        if y1 > ih: y0, y1 = ih - W * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    done = set()
    if os.path.exists(OUT_PATH):
        for l in open(OUT_PATH):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT_PATH, "a") as fout:
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
            W_, H_ = img.size
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            dx, dy = (gx1 - gx0) * PAD, (gy1 - gy0) * PAD
            ob = (max(0, gx0 - dx), max(0, gy0 - dy), min(W_, gx1 + dx), min(H_, gy1 + dy))
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]
            try:
                cx, cy = localize(img, text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)

            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "question": text, "img_wh": [W_, H_],
                   "gt_box_frac": [gx0 / W_, gy0 / H_, gx1 / W_, gy1 / H_],
                   "peak_frac": [cx, cy], "rand_frac": [rx, ry],
                   "probs": {}, "realized_tokens": {}}
            cells = [("uniform", fit(img, B0)[0]),
                     ("oracle", fit(img.crop(tuple(int(v) for v in ob)), B0)[0])]
            for Wn in WINDOWS:
                cells.append((f"attn@{Wn}", fit(window(img, cx, cy, Wn), B0)[0]))
                cells.append((f"rand@{Wn}", fit(window(img, rx, ry, Wn), B0)[0]))
            for nm, im in cells:
                p, rz = answer(im, text)
                rec["probs"][nm] = p
                rec["realized_tokens"][nm] = rz
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
