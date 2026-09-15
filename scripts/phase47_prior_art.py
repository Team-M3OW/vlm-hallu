"""
Phase 47: PRIOR-ART HEAD-TO-HEAD at a matched realized token budget, with honest pass accounting.

WHY THIS COMPARISON HAS NOT BEEN RUN
------------------------------------
Every crop/zoom method reports accuracy against a baseline that spends FEWER tokens than it does.
V*/SEAL, Zoom Eye, Chain-of-Spot, Visual CoT and DualFocus all add passes, add pixels, or both, and
none reports what the same budget buys if you simply spend it on a larger uniform image. §1's whole
point is that this makes the methods incomparable. This file fixes that.

WHAT IS AND IS NOT BEING COMPARED
---------------------------------
These are REIMPLEMENTATIONS OF THE PUBLISHED POLICIES ON ONE COMMON BACKBONE (Qwen3-VL-2B), not the
released checkpoints. That is deliberate and it cuts both ways:

  + It isolates the POLICY from the backbone. SEAL ships a fine-tuned vicuna-7B; comparing our
    method on Qwen3-VL against their model would confound the search policy with the base model,
    and whichever way it came out the result would be uninterpretable.
  - It cannot reproduce gains that come from their FINE-TUNING rather than their search. Methods
    that train the model to emit regions (Chain-of-Spot, Visual CoT, DualFocus) are represented by
    their inference-time mechanism only, running on a backbone that was not trained for it.

This is stated as a limitation, not hidden. The claim is about POLICIES at matched budget.

THE ARMS
--------
    uniform@300 / @600 / @1200    the axis everyone ignores; the honest bars at 1x / 2x / 4x
    grounding_crop                ask the VLM for the target's bounding box, crop to it, answer.
                                  This is the inference-time mechanism of the Chain-of-Spot /
                                  Visual CoT / DualFocus family. Qwen3-VL emits `bbox_2d` natively
                                  (normalised 0-1000), so the backbone supports it without training.
    zoom_eye                      best-first tree search over image regions scored by the model's
                                  own confidence that the region answers the question, depth 2 over
                                  quadrants. This is Zoom Eye's core and it is training-free.
    ours_attn@0.15                our fixed-policy crop at the ring-masked attention peak
    ours_adaptive                 our peak-gated adaptive policy (Phase 45)
    oracle / rand                 ceiling and floor

COST IS REPORTED IN BOTH CURRENCIES
-----------------------------------
`passes` (image prefills) and `image_tokens` (summed realized tokens over every pass). Tree search
is cheap per pass and expensive in total, which is exactly what a matched-budget comparison is for.
A generate call prefills the image once, so it counts as one image pass; its decode steps are text-
only and are noted separately rather than folded in.

Every crop is fitted to B0 by the same measured-token loop used everywhere else, so no arm gets a
resolution advantage at equal token count.
"""
import json
import math
import os
import random
import re
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase47_prior_art.jsonl"
BLOCK = list(range(16, 27))
B0 = 300
PAD = 0.25
W_OURS = 0.15
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(47)
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
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                      for s in ["yes", "Yes", " yes", " Yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                     for s in ["no", "No", " no", " No"]})

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
        return best if best else (img, measure(img))

    COST = {"passes": 0, "tokens": 0}

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["passes"] += 1
        COST["tokens"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    def p_yes(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["passes"] += 1
        COST["tokens"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        return float(torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                                torch.logsumexp(lg[no_ids], 0)]), 0)[0])

    def generate(img, text, maxnew=48):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["passes"] += 1
        COST["tokens"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            o = model.generate(**i, max_new_tokens=maxnew, do_sample=False)
        return tok.decode(o[0][i["input_ids"].shape[-1]:], skip_special_tokens=True)

    def localize(img, text):
        small, _ = fit(img, B0)
        inp = build(small, text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n_img = gh * gw
        pos = (inp["input_ids"][0] == itid).nonzero().flatten()
        base = int(pos[0].item())
        COST["passes"] += 1
        COST["tokens"] += n_img
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        acc = (acc / len(BLOCK)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            m = acc.clone()
        f = m.flatten()
        j = int(f.argmax().item())
        return ((j % gw) + .5) / gw, ((j // gw) + .5) / gh, float(f.max())

    def window(img, cx, cy, W):
        iw, ih = img.size
        x0, y0 = (cx - W / 2) * iw, (cy - W / 2) * ih
        x1, y1 = (cx + W / 2) * iw, (cy + W / 2) * ih
        if x0 < 0: x0, x1 = 0, W * iw
        if y0 < 0: y0, y1 = 0, W * ih
        if x1 > iw: x0, x1 = iw - W * iw, iw
        if y1 > ih: y0, y1 = ih - W * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    BB = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
                    r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")

    def grounding_crop(img, target, question):
        """Chain-of-Spot / Visual-CoT / DualFocus family: ask the model where to look, then crop."""
        small, _ = fit(img, B0)
        q = (f"Locate the {target} in the image and output its bounding box."
             if target else
             f"Locate the region needed to answer this question and output its bounding box. {question}")
        txt = generate(small, q)
        m = BB.search(txt)
        if not m:
            return None, txt
        v = [float(x) for x in m.groups()]
        # Qwen emits coordinates normalised to 0-1000
        x0, y0, x1, y1 = [c / 1000.0 for c in v]
        if x1 <= x0 or y1 <= y0:
            return None, txt
        dx, dy = (x1 - x0) * PAD, (y1 - y0) * PAD
        iw, ih = img.size
        box = (max(0, (x0 - dx)) * iw, max(0, (y0 - dy)) * ih,
               min(1, (x1 + dx)) * iw, min(1, (y1 + dy)) * ih)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None, txt
        return img.crop(tuple(int(b) for b in box)), txt

    def zoom_eye(img, question, depth=2):
        """Zoom Eye's core: best-first tree search over regions, scored by the model's own
        confidence that the region contains what the question needs. Training-free."""
        probe = ("Does this image region contain the visual information needed to answer "
                 f"this question? {question} Answer yes or no.")
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
        for _ in range(depth):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            quads = [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]
            best, bs = None, -1.0
            for q in quads:
                iw, ih = img.size
                c = img.crop((int(q[0] * iw), int(q[1] * ih), int(q[2] * iw), int(q[3] * ih)))
                if c.width < 16 or c.height < 16:
                    continue
                cf, _ = fit(c, B0)
                s = p_yes(cf, probe)
                if s > bs:
                    bs, best = s, q
            if best is None:
                break
            x0, y0, x1, y1 = best
        iw, ih = img.size
        return img.crop((int(x0 * iw), int(y0 * ih), int(x1 * iw), int(y1 * ih)))

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
            W_, H_ = img.size
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            dx, dy = (gx1 - gx0) * PAD, (gy1 - gy0) * PAD
            ob = (max(0, gx0 - dx), max(0, gy0 - dy), min(W_, gx1 + dx), min(H_, gy1 + dy))
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]
            tgt = (ann.get("target_object") or [None])[0]
            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "question": text, "img_wh": [W_, H_],
                   "gt_box_frac": [gx0 / W_, gy0 / H_, gx1 / W_, gy1 / H_],
                   "target": tgt, "probs": {}, "cost": {}}

            def arm(name, fn):
                COST["passes"] = 0
                COST["tokens"] = 0
                p = fn()
                rec["probs"][name] = p
                rec["cost"][name] = {"passes": COST["passes"], "tokens": COST["tokens"]}

            arm("uniform@300", lambda: answer(fit(img, B0)[0], text))
            arm("uniform@600", lambda: answer(fit(img, 2 * B0)[0], text))
            arm("uniform@1200", lambda: answer(fit(img, 4 * B0)[0], text))
            arm("oracle", lambda: answer(fit(img.crop(tuple(int(v) for v in ob)), B0)[0], text))
            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            arm("rand@0.15", lambda: answer(fit(window(img, rx, ry, W_OURS), B0)[0], text))

            def ours():
                cx, cy, pk = localize(img, text)
                rec["peak"] = pk
                rec["peak_frac"] = [cx, cy]
                return answer(fit(window(img, cx, cy, W_OURS), B0)[0], text)
            arm("ours_attn@0.15", ours)

            def ground():
                c, txt = grounding_crop(img, tgt, text)
                rec["ground_raw"] = txt[:200]
                if c is None:
                    return answer(fit(img, B0)[0], text)     # graceful fallback, cost counted
                return answer(fit(c, B0)[0], text)
            arm("grounding_crop", ground)

            arm("zoom_eye", lambda: answer(fit(zoom_eye(img, text), B0)[0], text))

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
