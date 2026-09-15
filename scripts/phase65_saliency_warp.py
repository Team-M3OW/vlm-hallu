"""
Phase 65: WARP INSTEAD OF CROP. Reallocate tokens without discarding the scene.

WHY
---
Cropping is a STEP FUNCTION over the token budget: every token inside the window, none outside. At
W=0.15 that discards **97.75% of the image**, which is exactly the failure mode we measured --
multi-region questions break (§6D), and at 4K crop-only is -17.3pp against the budget axis because a
miss throws the scene away.

A saliency-guided WARP is the continuous version of the same operation. Sample the image
non-uniformly -- densely where the model's own attention points, sparsely elsewhere -- so the target
is magnified and the periphery is COMPRESSED RATHER THAN REMOVED. Same token count, one image, one
extra pass, nothing discarded.

Verified on a synthetic target (0.39% of the image): lambda=0 reproduces it exactly, lambda=0.5
magnifies it to 2.69%, lambda=0.9 to 6.03% (15.5x), with the full intensity range of the scene still
present at every setting.

CONSTRUCTION
------------
Separable inverse-CDF resampling. The attention map (ring-masked block mean L16-26, Gaussian
smoothed) is marginalised to x and y; each marginal is mixed with the uniform density by lambda,
given a floor so the periphery can never collapse, integrated to a CDF, and inverted to give source
coordinates for each output pixel. Bilinear interpolation. lambda = 0 is the identity, so the arm
family contains its own null.

ARMS (answer pass always B0=300; attention-dependent arms also pay the localiser pass)
    uniform@300 / @600 / @1200    the bar, measured in-run
    oracle                        crop to the GT box -- the ceiling
    top1@0.15                     single-crop, the deployed policy
    multi4                        4 windows in one pass, the current best (§11A)
    warp@{0.3,0.5,0.7,0.9}        this file's proposal, same cost as top1

WHAT WOULD FALSIFY IT: if warping does not beat cropping at equal cost, then discarding the scene
was not what was costing us, and the crop-based framing was right. The lambda sweep also has to be
well-behaved -- if only one value works, that is a tuning artifact, not a method.
"""
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase65_saliency_warp.jsonl"
BLOCK = list(range(16, 27))
B0, W, MIN_SEP, PAD = 300, 0.15, 0.20, 0.25
LAMBDAS = [0.3, 0.5, 0.7, 0.9]
FLOOR = 0.15
CONN = "Here is another zoomed-in crop from the same image."
Image.MAX_IMAGE_PIXELS = None


def axis_src(s1d, n_out, lam, floor=FLOOR):
    s = np.clip(np.asarray(s1d, dtype=np.float64), 0, None)
    s = s / max(s.sum(), 1e-12)
    u = np.ones_like(s) / len(s)
    s = (1 - lam) * u + lam * s
    s = (1 - floor) * s + floor * u
    cdf = np.concatenate([[0.0], np.cumsum(s)])
    cdf /= cdf[-1]
    return np.interp(np.linspace(0, 1, n_out), cdf, np.linspace(0.0, 1.0, len(cdf)))


def warp_image(img, sal, lam, out_size):
    """Resample the FULL-RESOLUTION source onto an output grid of `out_size`.

    The output must be built at the TARGET size, not at the source size. Writing it the other way
    round allocates two float64 meshgrids at full resolution (up to 5759x1440 here, ~1.1 GB per item
    across four lambdas) and the run was OOM-killed after 6 items. Sampling straight to the output
    grid is also the correct operation: the warp's job is to decide WHERE the output pixels read
    from, and the output only ever needs to be the size that yields B0 tokens.
    """
    Wd, Ht = img.size
    ow, oh = out_size
    xs = (axis_src(sal.sum(0), ow, lam) * (Wd - 1)).astype(np.float32)
    ys = (axis_src(sal.sum(1), oh, lam) * (Ht - 1)).astype(np.float32)
    XX, YY = np.meshgrid(xs, ys)                      # oh x ow, not Ht x Wd
    a = np.asarray(img, dtype=np.float32)
    out = np.stack([map_coordinates(a[..., c], [YY, XX], order=1, mode="nearest")
                    for c in range(3)], -1)
    del XX, YY
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
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
    COST = {"p": 0, "t": 0}

    def chat(k, text):
        c = [{"type": "image"}]
        for _ in range(k - 1):
            c += [{"type": "text", "text": CONN}, {"type": "image"}]
        c.append({"type": "text", "text": text})
        return pr.apply_chat_template([{"role": "user", "content": c}],
                                      tokenize=False, add_generation_prompt=True)

    def build(imgs, text):
        return pr(images=imgs, text=chat(len(imgs), text), return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build([img], "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=6, tol=0.06):
        Wd, Ht = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(Wd * sc)), max(28, int(Ht * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    def answer(imgs, text):
        i = build(imgs, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["p"] += 1; COST["t"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    def saliency(img, text, k=4):
        small = fit(img, B0)
        inp = build([small], text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n_img = gh * gw
        base = int((inp["input_ids"][0] == itid).nonzero().flatten()[0].item())
        COST["p"] += 1; COST["t"] += n_img
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        torch.cuda.empty_cache()
        m = (acc / len(BLOCK)).reshape(gh, gw)
        r = torch.full_like(m, -1.0)
        if gh > 2 and gw > 2:
            r[1:-1, 1:-1] = m[1:-1, 1:-1]
        else:
            r = m.clone()
        f = r.flatten()
        pts = []
        for i in torch.argsort(f, descending=True).tolist():
            if f[i].item() < 0:
                break
            px, py = ((i % gw) + .5) / gw, ((i // gw) + .5) / gh
            if all(math.hypot(px - x, py - y) >= MIN_SEP for x, y in pts):
                pts.append((px, py))
            if len(pts) == k:
                break
        while len(pts) < k:
            pts.append(pts[0])
        sal = np.clip(r.cpu().numpy(), 0, None)          # ring-masked, non-negative
        sal = gaussian_filter(sal, 1.0)
        return sal, pts

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
            dx, dy = (gx1 - gx0) * PAD, (gy1 - gy0) * PAD
            ob = (max(0, gx0 - dx), max(0, gy0 - dy), min(IW, gx1 + dx), min(IH, gy1 + dy))
            text = ex["text"]
            area = ((gx1 - gx0) * (gy1 - gy0)) / (IW * IH)
            COST["p"] = 0; COST["t"] = 0
            sal, pts = saliency(img, text)
            loc = dict(COST)
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str)
                   else int(ex["label"]),
                   "tokens_on_target": area * B0, "probs": {}, "cost": {}}

            def arm(nm, imgs, extra):
                COST["p"] = 0; COST["t"] = 0
                rec["probs"][nm] = answer(imgs, text)
                rec["cost"][nm] = {"passes": COST["p"] + extra[0], "tokens": COST["t"] + extra[1]}

            for nm, bgt in (("uniform@300", B0), ("uniform@600", 2*B0), ("uniform@1200", 4*B0)):
                arm(nm, [fit(img, bgt)], (0, 0))
            arm("oracle", [fit(img.crop(tuple(int(v) for v in ob)), B0)], (0, 0))
            arm("top1@0.15", [fit(window(img, *pts[0], W), B0)], (loc["p"], loc["t"]))
            arm("multi4", [fit(window(img, *pts[i], W), B0 // 4) for i in range(4)],
                (loc["p"], loc["t"]))
            # learn the output dimensions ONCE from the uniform fit, then warp straight to them:
            # identical token count to uniform@300, no full-resolution intermediate.
            ow, oh = fit(img, B0).size
            for lam in LAMBDAS:
                arm(f"warp@{lam}", [warp_image(img, sal, lam, (ow, oh))], (loc["p"], loc["t"]))
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
