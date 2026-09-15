"""
Phase 23 (E0): IS OUR `uniform` BASELINE A STRAWMAN? -- AnyRes tiling at matched budget.

THE HOLE THIS CLOSES
--------------------
Every allocation result so far (Phase 17/20/21) compares query-conditional placement against
`uniform`: the whole image DOWNSAMPLED to the budget. **No production VLM does that.**
LLaVA-NeXT (AnyRes), InternVL2 (dynamic tiling) and Qwen-VL all spend a large budget by
**tiling the image on a grid** -- uniformly over space, with the grid chosen from the image's
aspect ratio *before the question is read* -- plus a global thumbnail.

So `uniform` is the weak version of the real baseline, and a reviewer who builds these systems
will say so first. Until this arm exists we CANNOT write "modern VLMs allocate uniformly": that
sentence is currently an inference about how those systems work, not a measurement.

WHAT THE ARM IS
---------------
`anyres@B` reimplements the AnyRes/dynamic-tiling pathway at OUR matched budget:
  * grid (r,c) chosen by aspect-ratio match with r*c <= MAX_TILES and r*c >= 2 (1x1 is `uniform`,
    already an arm), ties broken toward MORE tiles -- InternVL's rule
  * image split into r*c equal cells, PLUS a global thumbnail (both LLaVA-NeXT and InternVL
    prepend one)
  * every tile AND the thumbnail resized to the SAME token count, B/(r*c+1) -- this is the
    defining property of AnyRes (all crops go to the same base_size), so the calibration is
    PER-IMAGE, not joint. Joint calibration would hand the thumbnail r*c times more tokens than
    a tile and would not be AnyRes.

Structurally this makes the contrast exact and small:
    anyres@B       = thumbnail + N tiles placed by the GRID          (query-independent)
    alloc_query@B  = thumbnail + 1 crop placed by the QUESTION       (query-conditional)
same budget, same architecture, same number of images only when N=1; the difference is the
placement POLICY.

THE CONFOUND ARM
----------------
`anyres_pick@B` keeps the AnyRes grid but forwards only the single tile with the largest overlap
with the target, + thumbnail (B/2 each). Its tile is GRID-ALIGNED and therefore loose -- the
target sits wherever it falls in the cell -- whereas `alloc_query`'s crop is tight around the
target. So:
    anyres_pick >> anyres      => the win is PLACEMENT (knowing which region to spend on)
    alloc_query >> anyres_pick => the win also needs CROP TIGHTNESS, not just region choice
Without this arm those two explanations are not separable.

PRE-REGISTERED OUTCOMES (written before the run)
------------------------------------------------
* `alloc_query` beats `anyres` at matched realized budget
      => the claim is about the pathway production systems actually ship, and the strawman
         objection dies in the same table.
* `anyres` closes most of the `uniform`->`alloc_query` gap
      => OUR UNIFORM BASELINE WAS A STRAWMAN. C1/C4's effect sizes shrink to whatever survives
         against tiling and the paper must be rewritten around that smaller number. This is the
         outcome that would hurt, and it gets reported either way.
* `anyres` beats `alloc_query`
      => query-conditional placement is not the operative variable at all; retract C1's framing.

GATE 1 (from the plan): Qwen3-VL uses dynamic resolution, NOT fixed tiling, so `anyres` here is a
genuine reimplementation rather than the model's default path -- there is no native tiling for it
to collapse onto. Realized tokens are logged per arm; the budget-match check is the analyzer's
first table and a mismatched comparison is void.
GATE 2: do not generalize the AnyRes result from Qwen alone -- InternVL2 / llava-onevision ship
tiling natively and belong in E1.
"""
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase17_budget_allocation import fit_to_budget
from phase7_vision_zoom import CONNECTOR_TEXT

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# The box is shared: a long training job holds ~35GB and other jobs come and go, so this run
# cannot assume the GPU is free. It waits for headroom before loading and retries an item on OOM
# rather than dying, which is what killed the first launch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
NEED_GB = 9.0          # weights ~4.3GB fp16 + activations for the 5-image anyres arm at B=1200
OOM_RETRIES = 40


def free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1024 ** 3


def wait_for_gpu(need=NEED_GB, every=120):
    while free_gb() < need:
        print(f"  waiting for GPU: {free_gb():.1f}GB free, need {need:.1f}GB", flush=True)
        time.sleep(every)
    print(f"  GPU has {free_gb():.1f}GB free, proceeding", flush=True)


OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase23_anyres_results.jsonl"
BUDGETS = [300, 600, 1200]
MAX_TILES = 4          # LLaVA-NeXT ships 4 (2x2 over the base grid); InternVL2 defaults to 6-12
PAD = 0.25             # identical to Phase 20 so `alloc_query` is the same arm as before
Image.MAX_IMAGE_PIXELS = None


def pick_grid(W, H, max_tiles=MAX_TILES):
    """Aspect-ratio-matched grid, InternVL's selection rule. r*c >= 2 (1x1 is the `uniform` arm)."""
    ar = W / H
    best = None
    for r in range(1, max_tiles + 1):
        for c in range(1, max_tiles + 1):
            n = r * c
            if n > max_tiles or n < 2:
                continue
            d = abs((c / r) - ar)
            # tie-break toward MORE tiles, as InternVL does when the image is large enough
            if best is None or (d, -n) < (best[0], -best[1] * best[2]):
                best = (d, r, c)
    return best[1], best[2]


def split_tiles(img, r, c):
    W, H = img.size
    out = []
    for i in range(r):
        for j in range(c):
            box = (int(j * W / c), int(i * H / r), int((j + 1) * W / c), int((i + 1) * H / r))
            out.append((box, img.crop(box)))
    return out


def best_tile(tiles, bbox):
    """Index of the tile with the largest intersection with the target bbox (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = bbox
    areas = []
    for (a, b, cc, d), _ in tiles:
        iw = max(0, min(cc, x1) - max(a, x0))
        ih = max(0, min(d, y1) - max(b, y0))
        areas.append(iw * ih)
    return max(range(len(areas)), key=lambda k: areas[k])


def main():
    rng = random.Random(107)   # same seed as Phase 20 => identical random regions
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    print("Fetching V*Bench...")
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    print(f"  {len(ds)} items, root={root}")

    wait_for_gpu()
    print("Loading Qwen3-VL-2B (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]

    def _chat(imgs, text, conn):
        if len(imgs) == 1:
            content = [{"type": "image"}]
        elif len(imgs) == 2:
            content = ([{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                        {"type": "image"}] if conn else [{"type": "image"}, {"type": "image"}])
        else:
            # >2 images only happens for the tiling arms; the connector text is the same one
            # Phase 7 fixed, inserted once, so the tiling arm is not handed a different prompt.
            content = [{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT}]
            content += [{"type": "image"} for _ in imgs[1:]]
        content.append({"type": "text", "text": text})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    def measure(imgs, text="x", conn=True):
        i = processor(images=imgs, text=_chat(imgs, text, conn), return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    def run(imgs, text, conn):
        """OOM-tolerant: a co-tenant job can take the memory mid-run, so back off and retry
        instead of dying. Never silently returns a wrong answer -- it either succeeds or raises."""
        for attempt in range(OOM_RETRIES):
            try:
                return _run_once(imgs, text, conn)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                wait = min(300, 15 * (attempt + 1))
                print(f"    OOM (free {free_gb():.1f}GB), retry {attempt+1} in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("persistent CUDA OOM")

    def _run_once(imgs, text, conn):
        i = processor(images=imgs, text=_chat(imgs, text, conn), return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        pooled = torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids])
        probs = torch.softmax(pooled, 0)
        realized = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        return probs.tolist(), int(probs.argmax().item()), realized

    def fit_each(imgs, per, total, text, conn):
        """AnyRes calibration, TWO STAGES.

        Stage 1 sizes every image (thumbnail included) to the SAME token count -- the defining
        property of AnyRes, where every tile and the thumbnail go to one base_size.
        Stage 2 applies a single common rescale to the whole list to hit the TOTAL budget.

        Stage 2 is not cosmetic. With per-image sizing alone, `smart_resize`'s granularity at
        ~60 tokens/image compounds across 5 images: the measured B=300 spread was 19%, handing
        `anyres` 350 tokens against everyone else's 300 -- a 17% advantage to the arm under test.
        That is bug #18 all over again. A common scale factor preserves the equal-share property
        (all images are already near-equal) while fixing the total, so the arm is judged at the
        budget it is supposed to have.
        """
        staged = []
        for im in imgs:
            f, _ = fit_to_budget(lambda ims: measure(ims, "x", False), [im], max(4, per))
            staged.append(f[0])
        fitted, _ = fit_to_budget(lambda ims: measure(ims, text, conn), staged, total)
        return fitted

    done = set()
    try:
        for l in open(OUT_PATH):
            done.add(json.loads(l)["question_id_full"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for ex in ds:
            rel = ex["image"]
            qid_full = f"{rel}::{ex['question_id']}"
            img_path = os.path.join(root, rel)
            ann_path = os.path.splitext(img_path)[0] + ".json"
            if not (os.path.exists(img_path) and os.path.exists(ann_path)):
                continue
            ann = json.load(open(ann_path))
            if not ann.get("bbox"):
                continue
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            x0 = min(b[0] for b in ann["bbox"]); y0 = min(b[1] for b in ann["bbox"])
            x1 = max(b[0] + b[2] for b in ann["bbox"]); y1 = max(b[1] + b[3] for b in ann["bbox"])
            pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
            cx0, cy0 = max(0, x0 - pw), max(0, y0 - ph)
            cx1, cy1 = min(W, x1 + pw), min(H, y1 + ph)
            if cx1 - cx0 < 8 or cy1 - cy0 < 8:
                continue
            gt_crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
            gw, gh = cx1 - cx0, cy1 - cy0
            # drawn from the SAME rng sequence as Phase 20 so the random region matches
            rx = rng.uniform(0, max(1, W - gw)); ry = rng.uniform(0, max(1, H - gh))
            rnd_crop = img.crop((int(rx), int(ry), int(rx + gw), int(ry + gh)))
            if qid_full in done:
                continue

            r, c = pick_grid(W, H)
            tiles = split_tiles(img, r, c)
            ti = best_tile(tiles, (x0, y0, x1, y1))
            tile_imgs = [t[1] for t in tiles]
            picked = tiles[ti][1]

            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            rec = {"question_id_full": qid_full, "image": rel, "category": ex["category"],
                   "label": label, "target_object": ann.get("target_object"),
                   "img_wh": [W, H], "bbox_area_frac": ((x1 - x0) * (y1 - y0)) / (W * H),
                   "grid": [r, c], "picked_tile": ti, "n_tiles": r * c,
                   "arms": {}, "pred": {}, "realized_tokens": {}}
            text = ex["text"]
            for B in BUDGETS:
                jobs = {
                    # old baseline, kept so this file is self-paired and replicates Phase 20
                    "uniform":     ([img], False, None),
                    "alloc_query": ([img, gt_crop], True, B // 2),
                    "alloc_random": ([img, rnd_crop], True, B // 2),
                    # THE NEW STRONG BASELINE: thumbnail + grid tiles, equal token share each
                    "anyres":      ([img] + tile_imgs, True, B // (r * c + 1)),
                    # grid tile chosen by the query -- separates placement from crop tightness
                    "anyres_pick": ([img, picked], True, B // 2),
                }
                for name, (imgs, conn, per) in jobs.items():
                    if per is None:
                        fitted, _ = fit_to_budget(lambda ims: measure(ims, text, conn), imgs, B)
                    else:
                        fitted = fit_each(imgs, per, B, text, conn)
                    probs, pred, realized = run(fitted, text, conn)
                    rec["arms"][f"{name}@{B}"] = probs
                    rec["pred"][f"{name}@{B}"] = pred
                    rec["realized_tokens"][f"{name}@{B}"] = realized
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
