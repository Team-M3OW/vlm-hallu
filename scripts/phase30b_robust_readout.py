"""
Phase 30b: is the Phase 22 attention signal actually USABLE as a localizer at B0=300, and if so
what read-out recovers it?

WHAT 30a ESTABLISHED (pre-registered, and it stands as a negative)
------------------------------------------------------------------
The min/max bounding box of the top-r attention tokens is the WHOLE IMAGE on essentially every
item, at every layer and every keep rate. So that read-out cannot drive an allocator.

WHY, and why that is not the same as "attention is diffuse"
-----------------------------------------------------------
Attention is the opposite of diffuse. On a 7x10=70-token grid, 50% of the image-attention mass sits
on the **top 2 tokens** (max/median ratio 155x at L2). The problem is that the high-attention
tokens are not CO-LOCATED: the top-2 box is compact but by the top-6 the box spans the full grid.
That is the attention-sink / register-token pattern -- a few positions absorb large attention
largely independent of content. A min/max bbox is an extreme-order statistic, so three scattered
sinks destroy it no matter how good the ranking is.

    CAUTION, and the reason this script re-measures rather than reusing those numbers: the figures
    above come from a 7x10 grid produced by a 0.05 rescale in a one-off diagnostic. The actual
    experiment runs at B0=300 -> ~15x20. That is 4x more cells, and both the NUMBER of sinks and
    the area they span are unmeasured there. Nothing from the 70-token grid is carried forward.

THE QUESTION THAT DECIDES EVERYTHING, asked first
--------------------------------------------------
`attn_mass_in_gt` came back 0.0000 median in 30a. That is NOT evidence of failure: at B0=300 a
V*Bench target spans **0.04-2.4 merged tokens** (measured; most under one), so the set of grid cells
whose centre falls inside the GT box is frequently EMPTY and the statistic is undefined rather than
low. It has to be replaced with something that is always defined:

    gt_rank  = the attention rank (1 = highest) of the single grid cell CONTAINING THE GT CENTRE
    gt_pct   = that rank as a percentile of n_img (0.0 = top, 0.5 = chance, 1.0 = worst)

    * gt_pct near the top      -> ranking transferred; sinks are a removable nuisance and a
                                  robust read-out should work.
    * gt_pct near 0.5 (chance) -> Phase 22's RePOPE signal DID NOT TRANSFER to V*Bench at 300
                                  tokens, and no read-out repair rescues it. That is the finding,
                                  and it is the transfer risk flagged as uncertain before starting.

THE READ-OUT BEING TESTED (one, chosen by mechanism, not by trying four)
------------------------------------------------------------------------
**Percentile box**: take the top-k tokens by attention, then take the 10th-90th percentile of their
row and column coordinates instead of min/max. Rationale: it is a rank-order statistic, matching the
nature of the signal itself, and it discards a fixed small fraction of coordinates by construction
rather than by tuning a threshold against the outcome.

Deliberately NOT used: attention-weighted centroid / covariance. Any mass-weighted statistic is
dominated by precisely the tokens that must be discarded, because the sinks carry the mass (50% in
two tokens). Deliberately NOT used as the primary: argmax-and-grow, because it stakes everything on
one token and whether the argmax is the target or a sink is exactly what `gt_rank` is measuring.

Everything needed to score ANY other read-out offline is logged (`topk_rc`), so no further forward
pass is needed to try alternatives.

STATUS: POST-HOC. The pre-registered headline remains 30a's min/max result. This is a repair
attempt and is reported as one.
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
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30b_robust_readout.jsonl"
LAYERS = [2, 4, 8]
KEEP_RATES = [0.10, 0.25]
PCTL = 10          # percentile box keeps the central 10th-90th of top-k coordinates
B0 = 300
Image.MAX_IMAGE_PIXELS = None


def pctl_box(rows, cols, gh, gw, p=PCTL):
    """Central (100-2p)% box over the top-k coordinates, in fractional image coords."""
    def lohi(v):
        v = sorted(v)
        n = len(v)
        if n <= 2:
            return v[0], v[-1]
        lo = v[max(0, int(round((p / 100) * (n - 1))))]
        hi = v[min(n - 1, int(round((1 - p / 100) * (n - 1))))]
        return lo, hi
    r0, r1 = lohi(rows)
    c0, c1 = lohi(cols)
    return c0 / gw, r0 / gh, (c1 + 1) / gw, (r1 + 1) / gh


def score(box, gt):
    px0, py0, px1, py1 = box
    gx0, gy0, gx1, gy1 = gt
    ix0, iy0 = max(px0, gx0), max(py0, gy0)
    ix1, iy1 = min(px1, gx1), min(py1, gy1)
    inter = max(0., ix1 - ix0) * max(0., iy1 - iy0)
    gt_a = max((gx1 - gx0) * (gy1 - gy0), 1e-12)
    pa = max((px1 - px0) * (py1 - py0), 1e-12)
    return {"box_frac": [px0, py0, px1, py1], "area_frac": pa,
            "containment": inter / gt_a, "headroom": (inter / gt_a) / pa}


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print("Loading Qwen3-VL-2B (fp16, eager attention)...", flush=True)
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
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
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
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            gx0 = min(b[0] for b in ann["bbox"]) / W
            gy0 = min(b[1] for b in ann["bbox"]) / H
            gx1 = max(b[0] + b[2] for b in ann["bbox"]) / W
            gy1 = max(b[1] + b[3] for b in ann["bbox"]) / H
            gt = (gx0, gy0, gx1, gy1)
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2

            small, realized = fit(img, B0)
            inp = build(small, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // 2, g[2] // 2
            n_img = gh * gw
            ids = inp["input_ids"][0]
            pos = (ids == img_tok_id).nonzero().flatten()
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            base = int(pos[0].item())

            # the grid cell containing the GT centre -- always defined, unlike attn_mass_in_gt
            gr = min(gh - 1, max(0, int(cy * gh)))
            gc = min(gw - 1, max(0, int(cx * gw)))
            gt_cell = gr * gw + gc

            rec = {"question_id_full": qid, "category": ex["category"],
                   "img_wh": [W, H], "realized_tokens": realized, "grid": [gh, gw],
                   "n_img_tokens": n_img, "gt_box_frac": list(gt),
                   "gt_area_frac": (gx1 - gx0) * (gy1 - gy0),
                   "gt_tokens": (gx1 - gx0) * gw * (gy1 - gy0) * gh,
                   "gt_cell_rc": [gr, gc], "layers": {}}
            if len(pos) != n_img:
                rec["note"] = f"token count mismatch {len(pos)} vs {n_img}"
            for L in LAYERS:
                a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
                order = torch.argsort(a, descending=True).tolist()
                rank_of = {t: i for i, t in enumerate(order)}
                top1 = order[0]
                t1r, t1c = top1 // gw, top1 % gw
                d = {
                    # THE DECIDING NUMBER: where does the GT-centre cell rank?
                    "gt_rank": rank_of[gt_cell] + 1,
                    "gt_pct": (rank_of[gt_cell] + 1) / n_img,
                    "top1_rc": [t1r, t1c],
                    # normalised distance from the argmax cell to the GT centre
                    "top1_dist": (((t1c + .5) / gw - cx) ** 2 + ((t1r + .5) / gh - cy) ** 2) ** .5,
                    "top1_in_gt": bool(gx0 <= (t1c + .5) / gw <= gx1
                                       and gy0 <= (t1r + .5) / gh <= gy1),
                    "concentration": float(a.max() / (a.median() + 1e-12)),
                    "readouts": {},
                }
                for r in KEEP_RATES:
                    k = max(1, int(round(r * n_img)))
                    top = order[:k]
                    rows = [t // gw for t in top]
                    cols = [t % gw for t in top]
                    d["readouts"][f"minmax@{r}"] = score(
                        (min(cols) / gw, min(rows) / gh, (max(cols) + 1) / gw, (max(rows) + 1) / gh),
                        gt)
                    d["readouts"][f"pctl@{r}"] = score(pctl_box(rows, cols, gh, gw), gt)
                    if r == KEEP_RATES[0]:
                        # raw coordinates so any other read-out can be scored offline
                        d["topk_rc"] = [[int(x), int(y)] for x, y in zip(rows, cols)]
                rec["layers"][f"L{L}"] = d
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
