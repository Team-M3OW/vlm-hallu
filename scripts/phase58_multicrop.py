"""
Phase 58: MULTI-CROP IN ONE PASS. k candidate windows as k images, B0/k tokens each.

THE ARITHMETIC THAT MAKES THIS THE REMAINING SHOT
-------------------------------------------------
The method's binding constraint is proposal quality, not packaging or detection:

    oracle crop (perfect placement)            92.7%   vs uniform@300's 56.5%
    our argmax window: mean coverage            40.5%   peak inside GT on only 15.7% of items
    ORACLE among the map's own top-5 windows    61.5%   coverage
    ORACLE among top-10                         66.1%

So the ranking already contains far better windows than the top-1 rule uses. Every previous attempt
to exploit that paid a forward pass per candidate, and Phase 47 showed the budget axis outruns that
(56.5 -> 63.9 -> 71.7% at 1x/2x/4x).

It does not have to cost passes. The processor accepts SEVERAL IMAGES IN ONE FORWARD PASS, so k
windows at B0/k tokens each is ONE pass at the SAME total budget. No candidate selection is needed:
the model sees all k simultaneously and can answer from whichever contains the evidence.

Resolution is still gained, by a lot. A W=0.15 window is 2.25% of the image; at B0/4 = 75 tokens
that is 33 tokens per % of area against uniform@300's 3 -- an 11x density gain per window, four
windows over.

ARMS (pass 1 = uniform@B0, which yields an answer AND the ranked candidates, so its 300 tokens are
charged once; the multi-crop pass adds its own)
    uniform@300 / @600 / @1200      the bar, measured in-run
    top1@0.15                       the deployed single-window policy (this file's control)
    multi2 / multi3 / multi4        k separated windows, B0/k tokens each, ONE pass
    multi4_600                      k=4 at 150 tokens each -- the same idea at 2x budget
    multi3_scene                    scene at B0/2 plus 3 windows at B0/6 -- keeps context too
`peak` is logged so every arm can additionally be peak-gated offline.

WHAT WOULD FALSIFY IT: if multi-crop does not beat top1 at equal total tokens, then the extra
candidates are not usable by the model even when handed to it directly, and proposal quality cannot
be bought this way either.
"""
import json
import math
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase58_multicrop.jsonl"
BLOCK = list(range(16, 27))
B0, W, MIN_SEP, PAD = 300, 0.15, 0.20, 0.25
Image.MAX_IMAGE_PIXELS = None
CONN = "Here is another zoomed-in crop from the same image."


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

    def candidates(img, text, k):
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
        acc = (acc / len(BLOCK)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            m = acc.clone()
        f = m.flatten()
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
        return pts, float(f.max())

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    def cov(cx, cy, Wn, gt):
        x0, x1, y0, y1 = cx - Wn / 2, cx + Wn / 2, cy - Wn / 2, cy + Wn / 2
        if x0 < 0: x0, x1 = 0., Wn
        if y0 < 0: y0, y1 = 0., Wn
        if x1 > 1: x0, x1 = 1 - Wn, 1.
        if y1 > 1: y0, y1 = 1 - Wn, 1.
        gx0, gy0, gx1, gy1 = gt
        return (max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0))
                / max((gx1 - gx0) * (gy1 - gy0), 1e-9))

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
            COST["p"] = 0; COST["t"] = 0
            pts, peak = candidates(img, text, 4)
            loc = dict(COST)
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str)
                   else int(ex["label"]),
                   "gt_box_frac": gt, "peak": peak, "cands": pts,
                   "cand_cov": [cov(px, py, W, gt) for px, py in pts],
                   "probs": {}, "cost": {}}

            def arm(nm, imgs, extra):
                COST["p"] = 0; COST["t"] = 0
                rec["probs"][nm] = answer(imgs, text)
                rec["cost"][nm] = {"passes": COST["p"] + extra[0], "tokens": COST["t"] + extra[1]}

            for nm, b in (("uniform@300", B0), ("uniform@600", 2 * B0), ("uniform@1200", 4 * B0)):
                arm(nm, [fit(img, b)], (0, 0))
            arm("oracle", [fit(img.crop((int(gt[0]*IW), int(gt[1]*IH),
                                         int(gt[2]*IW), int(gt[3]*IH))), B0)], (0, 0))
            arm("top1@0.15", [fit(window(img, *pts[0], W), B0)], (loc["p"], loc["t"]))
            for k in (2, 3, 4):
                ims = [fit(window(img, *pts[i], W), B0 // k) for i in range(k)]
                arm(f"multi{k}", ims, (loc["p"], loc["t"]))
            arm("multi4_600", [fit(window(img, *pts[i], W), 2 * B0 // 4) for i in range(4)],
                (loc["p"], loc["t"]))
            arm("multi3_scene", [fit(img, B0 // 2)] +
                [fit(window(img, *pts[i], W), B0 // 6) for i in range(3)], (loc["p"], loc["t"]))
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
