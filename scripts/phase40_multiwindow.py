"""
Phase 40: MULTI-WINDOW allocation. The intervention §3.4 lacks.

WHY THIS RUN AND NOT A BIGGER WINDOW
------------------------------------
§3 establishes evidence-set COVERAGE as the mediator of the sign of the allocation effect, but only
as a mediator: the one interventional test so far (the window-size sweep) came out null, and the
exogenous-placement instrument fired on 11/191 items. Causation is the open stage.

Phase 39D says which intervention is worth running. For the 96 zero-coverage items -- the bulk of
the negative mass -- the gap from the window EDGE to the GT box has median 0.231 of image width,
p25 0.120, p90 0.586; only 6.2% are within 0.05, and a W=0.35 window at the same centre rescues
just 20%. These are FAR misses. Enlarging the window cannot fix them (consistent with the null W
sweep). Splitting the budget across two separated attention peaks can, and it is the only available
intervention that MOVES COVERAGE WHILE HOLDING THE REALIZED BUDGET FIXED.

Coverage makes no reference to CONTIGUITY. That is the falsifiable part: if coverage is what
matters, two windows covering X% should behave like one window covering X%.

THE ARMS (all budget-gated to the same TOTAL realized tokens, ~B0=300)
----------------------------------------------------------------------
    uniform     whole image @ 300                          the baseline
    one         peak-1 window W=0.15 @ 300                  the deployed single-window method
    two_diff    peak-1 + peak-2 windows, 150 tokens each    the hypothesis
    two_same    peak-1 + peak-1 (DUPLICATED), 150 each      ** THE CONTROL THAT DECIDES THIS **
    two_rand    peak-1 + a RANDOM window, 150 each          placement control for window 2

`two_same` is the point of the design. Against it, `two_diff` holds constant: the number of images
(2), the total realized budget (300), the per-image resolution (150), and the two-image prompt
format including its connector sentence. The ONLY thing that varies is whether the second window
shows a DIFFERENT PLACE. Any other comparison confounds "a second region" with "half the resolution"
or with "the model handles two-image prompts differently", and those confounds are exactly what
sank the naive reading of the W sweep.

Peak 2 is the highest-scoring cell at least MIN_SEP (0.25 of image diagonal, in normalised coords)
from peak 1, on the same ring-masked L16-26 block mean. No ground truth is used anywhere in the
proposal; boxes are logged only so coverage can be computed offline afterwards.

PRE-REGISTERED READING, fixed here before the run
-------------------------------------------------
    two_diff > two_same  (paired, CI excludes 0)   -> a genuinely different second region ADDS
                                                      accuracy at fixed budget. Coverage is doing
                                                      causal work; §3 is upgraded from mediator.
    two_diff ~= two_same                           -> the second window adds nothing beyond format
                                                      and resolution. The coverage account FAILS as
                                                      an intervention and stays correlational.
                                                      Report as a negative; do not rescue it.
    two_diff < two_same                            -> a second location actively hurts; the
                                                      contiguity assumption the account denies is
                                                      real. Report and investigate.
Secondary: two_diff vs one tests whether multi-window beats the deployed method outright, and
two_diff vs two_rand repeats the placement contrast that decided V*Bench and HR-Bench.

Counts are MEASURED from the processor's own image_grid_thw, never computed. The budget gate voids
any contrast whose arms spread >10%.
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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase40_multiwindow.jsonl"
BLOCK = list(range(16, 27))
W_MAIN = 0.15
B0 = 300
MIN_SEP = 0.25
CONNECTOR = "Here is another zoomed-in crop of a region from the same image."
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(40)
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print("Loading Qwen3-VL-2B (fp16, eager attention)...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    img_tok_id = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]

    def chat(imgs, text):
        if len(imgs) == 1:
            content = [{"type": "image"}]
        else:
            content = [{"type": "image"}, {"type": "text", "text": CONNECTOR}, {"type": "image"}]
        content.append({"type": "text", "text": text})
        return pr.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=False, add_generation_prompt=True)

    def build(imgs, text):
        return pr(images=imgs, text=chat(imgs, text), return_tensors="pt")

    def measure(img):
        i = build([img], "x")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

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

    def answer(imgs, text):
        i = build(imgs, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    def two_peaks(img, text):
        small, _ = fit(img, B0)
        inp = build([small], text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n_img = gh * gw
        pos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
        base = int(pos[0].item())
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
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]      # ring mask: the columnar sinks live there (§4.1)
        else:
            m = acc.clone()
        flat = m.flatten()
        order = torch.argsort(flat, descending=True).tolist()
        pts = []
        for i in order:
            if flat[i].item() < 0:
                break
            px, py = ((i % gw) + .5) / gw, ((i // gw) + .5) / gh
            if all(math.hypot(px - x, py - y) >= MIN_SEP for x, y in pts):
                pts.append((px, py))
            if len(pts) == 2:
                break
        while len(pts) < 2:                       # tiny grids may admit no separated second peak
            pts.append(pts[0])
        return pts

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
            # NOTE: the rng draw happens AFTER every skip, so the stream stays reconstructible
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            W_, H_ = img.size
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]

            (p1x, p1y), (p2x, p2y) = two_peaks(img, text)
            rx, ry = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)

            full, _ = fit(img, B0)
            w1_full, _ = fit(window(img, p1x, p1y, W_MAIN), B0)
            w1_half, _ = fit(window(img, p1x, p1y, W_MAIN), B0 // 2)
            w2_half, _ = fit(window(img, p2x, p2y, W_MAIN), B0 // 2)
            wr_half, _ = fit(window(img, rx, ry, W_MAIN), B0 // 2)

            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "question": text, "img_wh": [W_, H_],
                   "gt_box_frac": [gx0 / W_, gy0 / H_, gx1 / W_, gy1 / H_],
                   "peak1": [p1x, p1y], "peak2": [p2x, p2y], "rand": [rx, ry],
                   "peaks_separated": bool(math.hypot(p1x - p2x, p1y - p2y) >= MIN_SEP),
                   "probs": {}, "realized_tokens": {}}
            for nm, imgs in [("uniform", [full]),
                             ("one", [w1_full]),
                             ("two_diff", [w1_half, w2_half]),
                             ("two_same", [w1_half, w1_half]),
                             ("two_rand", [w1_half, wr_half])]:
                p, rz = answer(imgs, text)
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
