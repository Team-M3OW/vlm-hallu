"""
Phase 30: the METHOD. A training-free, budget-preserving, attention-guided allocator.

WHY THIS EXISTS
---------------
Every allocation arm in this paper so far is an ORACLE: it crops the ground-truth box. Phase 29
quantified what that oracle is worth in practice -- a published proposer (ViRGo) captures only
**31-44% of the oracle gain**. This phase turns our oracle into a method and measures it against
that number. If it clears 44% it is a contribution; if it does not, that is a reportable negative
about how much of the allocation headroom is actually reachable without supervision.

THE SIGNAL WE ARE STANDING ON (Phase 22, same model: Qwen3-VL-2B, eager attention)
---------------------------------------------------------------------------------
Phase 22 pre-registered that attention-guided selection would retain the target BELOW the keep rate
for small objects (the mechanism it proposed for the ACL 2502.11501 finding). **That prediction was
refuted, and in the useful direction.** Measured retention minus keep rate at r=0.25:

    median tokens_on_object     L2        L4        L8
              0.26           +34.8pp   +33.8pp   +31.7pp
              1.03           +31.8pp   +34.9pp   +21.2pp
              4.19           +23.9pp   +26.7pp   +18.4pp
             18.14           +10.8pp    +8.7pp    +2.0pp
            104.86            +3.1pp    +1.2pp    +0.0pp

`random` and `uniform` sit at the keep rate to within 1.6pp, so the null is calibrated by
construction and no fitting is involved. Attention marks the target far above chance, and **most
strongly for the smallest objects** -- exactly the regime where allocation matters and exactly the
regime V*Bench is made of. So the localizer is already inside the model; it costs one cheap forward
pass to read out.

THIS SCRIPT IS THE CHEAP PRE-CHECK, NOT THE FULL EXPERIMENT
-----------------------------------------------------------
The full arm matrix is five conditions x 191 items. Before paying for it, there is one way the
method dies that a single forward pass per item can detect: **if attention is diffuse, the bounding
box of the top-r attention tokens is most of the image**, the crop degenerates to the uniform
baseline, and the method cannot gain anything no matter how the downstream arms are run. Phase 22
measured retention (are the target's tokens ranked highly?) but never measured the AREA of the
region those tokens span, which is what a proposer actually has to pay for.

So: one pass per item at the floor budget, read layer-L attention, emit the proposal box, and score
it against the GT box on the three quantities that decide whether the method is viable:

    containment  = |proposal ∩ GT| / |GT|     -- does the proposal keep the evidence?
    area_frac    = |proposal| / |image|       -- is it actually cheaper than looking everywhere?
    gain_headroom= containment / area_frac    -- evidence kept per unit of budget spent; 1.0 is
                                                 what you get by proposing the whole image

PRE-REGISTERED, before looking at any output
--------------------------------------------
* Layer **L2**. Not tuned: it is FastV's own published choice, and Phase 22 shows L2 is at or near
  the best rank quality in the small-object strata. Taking the argmax over {2,4,8} post hoc would
  be fitting the selector to the test set.
* Keep rate **r=0.25**, the middle of Phase 22's three rates.
* Proposal = axis-aligned bounding box of the top-`r` image tokens by attention, in grid
  coordinates, mapped to pixels FRACTIONALLY (cell j of grid_w spans [j/grid_w, (j+1)/grid_w] * W).
  Deliberately no PATCH constant: the project's PATCH is 14 while Qwen3-VL is really 16, harmless
  where it cancels but NOT harmless in a forward grid->pixel mapping. Fractions cancel it properly.

DECISION RULE, fixed now
------------------------
* median containment >= 0.5 AND median area_frac <= 0.5  -> viable, run the full arm matrix.
* median area_frac > 0.8                                 -> proposal is ~the whole image; the method
                                                            is dead, report as a negative with this
                                                            number as the reason.
* containment high but area_frac high too                -> attention is diffuse; a bbox is the wrong
                                                            read-out, and the honest report is that
                                                            the rank signal of Phase 22 does not
                                                            convert into a REGION.
"""
import json
import os
import sys
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30_proposal_quality.jsonl"
LAYERS = [2, 4, 8]          # L2 is the pre-registered one; 4/8 logged for the sensitivity table only
KEEP_RATES = [0.10, 0.25, 0.50]   # 0.25 pre-registered; others logged, not selected on
B0 = 300                    # the budget the allocation arms use throughout this paper
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print("Loading Qwen3-VL-2B (fp16, eager attention -- needed for output_attentions)...",
          flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    img_tok_id = model.config.image_token_id

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        o = build(img, "x")
        return int(sum(g[1] * g[2] // 4 for g in o["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        """Scale the image so the processor realizes ~`target` merged tokens.

        Qwen3-VL is a continuous-budget model (Phase 28: dynamic range 124x), so plain proportional
        refinement converges here and the geometric ladder the tiled models need is unnecessary.
        Best-seen is kept so a non-monotone step never makes the result worse than where it started.
        """
        W, H = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            w, h = max(32, int(W * sc)), max(32, int(H * sc))
            if w * h > 24_000_000:
                break
            cur = img.resize((w, h), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best if best else (img, measure(img))

    done = set()
    if os.path.exists(OUT_PATH):
        for l in open(OUT_PATH):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    with open(OUT_PATH, "a") as fout:
        for ex in ds:
            rel = ex["image"]
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, rel)
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            # GT union box, in fractional image coords -- same definition as every other phase
            gx0 = min(b[0] for b in ann["bbox"]) / W
            gy0 = min(b[1] for b in ann["bbox"]) / H
            gx1 = max(b[0] + b[2] for b in ann["bbox"]) / W
            gy1 = max(b[1] + b[3] for b in ann["bbox"]) / H

            small, realized = fit(img, B0)
            inp = build(small, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2          # merged-token grid
            n_img = gh * gw
            ids = inp["input_ids"][0]
            pos = (ids == img_tok_id).nonzero().flatten()
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            base = int(pos[0].item())

            rec = {"question_id_full": qid, "category": ex["category"],
                   "img_wh": [W, H], "realized_tokens": realized,
                   "grid": [gh, gw], "n_img_tokens": n_img,
                   "gt_box_frac": [gx0, gy0, gx1, gy1],
                   "gt_area_frac": (gx1 - gx0) * (gy1 - gy0),
                   "proposal": {}}
            if len(pos) != n_img:
                rec["note"] = f"token count mismatch {len(pos)} vs {n_img}"
            for L in LAYERS:
                a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
                order = torch.argsort(a, descending=True).tolist()
                for r in KEEP_RATES:
                    k = max(1, int(round(r * n_img)))
                    top = order[:k]
                    rows = [t // gw for t in top]
                    cols = [t % gw for t in top]
                    # fractional grid -> fractional image; no PATCH constant anywhere
                    px0, px1 = min(cols) / gw, (max(cols) + 1) / gw
                    py0, py1 = min(rows) / gh, (max(rows) + 1) / gh
                    ix0, iy0 = max(px0, gx0), max(py0, gy0)
                    ix1, iy1 = min(px1, gx1), min(py1, gy1)
                    inter = max(0., ix1 - ix0) * max(0., iy1 - iy0)
                    gt_a = max((gx1 - gx0) * (gy1 - gy0), 1e-12)
                    pa = (px1 - px0) * (py1 - py0)
                    rec["proposal"][f"L{L}@{r}"] = {
                        "box_frac": [px0, py0, px1, py1],
                        "area_frac": pa,
                        "containment": inter / gt_a,
                        "headroom": (inter / gt_a) / max(pa, 1e-12),
                        "attn_mass_in_gt": float(
                            a[[t for t in range(n_img)
                               if gx0 <= ((t % gw) + .5) / gw <= gx1
                               and gy0 <= ((t // gw) + .5) / gh <= gy1]].sum() / (a.sum() + 1e-9)),
                    }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            del out
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
