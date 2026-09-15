"""
Phase 49: CHEAP CANDIDATE RE-RANKING -- the method the mechanism points to.

THE BINDING CONSTRAINT, ESTABLISHED
-----------------------------------
Oracle placement scores 92.7% in ONE pass at 300 tokens against uniform's 56.5%, so the ceiling at
1x cost is enormous and the whole gap is proposal quality. Three routes to a better proposal have
now been measured and closed:

  adaptive window size   capped: the peak lands in the GT box on only 15.7% of items and the median
                         miss is 0.170 of image width, so no window size rescues an off-target peak.
  mass-contour boxes     cover 87-100% only by occupying 50-100% of the image, i.e. no zoom at all.
  a different query token the deployed `last` read-out is already the BEST of seven (Phase 48);
                         attention from the noun naming the target is near chance (gt_pct 0.367).

What is NOT closed: the GT cell ranks in the top 3.4% of cells (gt_pct 0.037) while the ARGMAX is
usually a distractor. Oracle selection among the map's own top-k reaches coverage 55.9% (k=3),
61.5% (k=5), 66.1% (k=10) against the argmax's 40.5%. The ranking already contains the answer; the
top-1 rule throws it away.

WHY THIS HAS TO BE CHEAP
------------------------
Phase 47 showed extra passes are a losing trade: the uniform curve rises 56.5 -> 63.9 -> 71.7% at
1x/2x/4x, so a 4-pass re-ranker must clear 71.7%. Scoring k candidates at FULL budget costs k
passes and prices itself out. So candidates are scored at B0/k tokens each -- all k together cost
ONE pass worth of tokens -- and only the winner is re-run at full budget.

    pass 1   uniform@B0 with attention   -> an answer, and the top-k separated candidates
    pass 2   k crops at B0/k tokens each -> pick the most confident  (k passes, B0 tokens total)
    pass 3   winner at full B0           -> the answer
    total    300 + 300 + 300 = 900 tokens, so the honest bar is uniform@900, NOT uniform@300.

THE RISK, STATED BEFORE RUNNING: the confidence signal that detects coverage (AUROC 0.835) was
measured on FULL-budget crops. At B0/k tokens a candidate may be too coarse for confidence to mean
anything, in which case selection degrades to random and the method spends 900 tokens for nothing.
`select_random` and `select_oracle` arms are run alongside to measure exactly that, so the outcome
is interpretable either way.

ARMS
----
    uniform@300/@600/@900/@1200   the budget axis; @900 is THIS method's compute-matched bar
    top1@300                      the deployed single-candidate policy
    select_conf                   k=3 candidates scored at B0/3, winner re-run at B0   <- the method
    select_random                 same cost, winner chosen at random -- the control that matters
    select_oracle                 same cost, winner chosen by true coverage -- the ceiling
    lowres_best                   answer directly from the best B0/3 candidate (no pass 3, 600 tok)
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

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase49_candidate_select.jsonl"
BLOCK = list(range(16, 27))
B0, K, W = 300, 3, 0.15
MIN_SEP = 0.20
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(49)
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

    COST = {"p": 0, "t": 0}

    def answer(img, text):
        i = build(img, text)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        COST["p"] += 1; COST["t"] += rz
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    def candidates(img, text, k):
        small, _ = fit(img, B0)
        inp = build(small, text)
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
        order = torch.argsort(f, descending=True).tolist()
        pts = []
        for i in order:
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

    def covf(cx, cy, Wn, gt):
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
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]
            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "gt_box_frac": gt, "probs": {}, "cost": {}, "cand_cov": []}

            def arm(nm, fn):
                COST["p"] = 0; COST["t"] = 0
                rec["probs"][nm] = fn()
                rec["cost"][nm] = {"passes": COST["p"], "tokens": COST["t"]}

            for nm, b in [("uniform@300", B0), ("uniform@600", 2 * B0),
                          ("uniform@900", 3 * B0), ("uniform@1200", 4 * B0)]:
                arm(nm, lambda b=b: answer(fit(img, b)[0], text))
            arm("oracle", lambda: answer(fit(img.crop((int(gt[0]*IW), int(gt[1]*IH),
                                                       int(gt[2]*IW), int(gt[3]*IH))), B0)[0], text))

            COST["p"] = 0; COST["t"] = 0
            pts, pk = candidates(img, text, K)
            rec["peak"] = pk
            rec["cands"] = pts
            rec["cand_cov"] = [covf(px, py, W, gt) for px, py in pts]
            base_cost = dict(COST)

            # score the k candidates at B0/K tokens each -- all k together cost ONE pass of tokens
            lowp, lowconf = [], []
            for px, py in pts:
                c = fit(window(img, px, py, W), max(40, B0 // K))[0]
                p = answer(c, text)
                lowp.append(p); lowconf.append(max(p))
            sel_cost = dict(COST)

            def finish(j, nm):
                COST["p"], COST["t"] = sel_cost["p"], sel_cost["t"]
                px, py = pts[j]
                p = answer(fit(window(img, px, py, W), B0)[0], text)
                rec["probs"][nm] = p
                rec["cost"][nm] = {"passes": COST["p"], "tokens": COST["t"]}

            finish(max(range(K), key=lambda j: lowconf[j]), "select_conf")
            finish(rng.randrange(K), "select_random")
            finish(max(range(K), key=lambda j: rec["cand_cov"][j]), "select_oracle")

            j = max(range(K), key=lambda j: lowconf[j])
            rec["probs"]["lowres_best"] = lowp[j]
            rec["cost"]["lowres_best"] = {"passes": sel_cost["p"], "tokens": sel_cost["t"]}
            rec["sel_conf_idx"] = j

            COST["p"], COST["t"] = base_cost["p"], base_cost["t"]
            p = answer(fit(window(img, *pts[0], W), B0)[0], text)
            rec["probs"]["top1@300"] = p
            rec["cost"]["top1@300"] = {"passes": COST["p"], "tokens": COST["t"]}

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
