"""
Phase 25: the Phase 23 confound fix -- AnyRes on the model AnyRes was INVENTED for.

WHY THIS EXISTS
---------------
Phase 23 found tiling losing to plain downsampling at matched budget (-9.4pp at B=600, -11.0pp at
B=1200) on Qwen3-VL. **That contrast is confounded and I refused to publish it.** Qwen3-VL was never
trained on tiled input, and the arms differed in IMAGE COUNT (5 tiles-as-separate-images vs 1), so
the comparison partly measures format mismatch rather than allocation policy. Gate 1 of the Phase 23
docstring predicted exactly this failure.

LLaVA-NeXT is the model that introduced AnyRes. Tiling is its NATIVE path, and -- the key design
point -- its native format is already **a base thumbnail plus grid tiles inside ONE image input**.
So the whole experiment can be run single-image, and the image-count confound disappears entirely.

THE DESIGN (every arm is ONE image through the model's own AnyRes pipeline)
--------------------------------------------------------------------------
    uniform@B       whole image at the model's CHEAPEST realization -- an UNMATCHED reference
                    row, because on this architecture there is no non-tiled way to spend more
                    (tiling IS the high-resolution path); coincides with `anyres` at the lowest B
    anyres@B        whole image, tiled by the native AnyRes path at budget B      <-- native path
    alloc_query@B   the GT-crop region, AnyRes'd to the same realized token count
    alloc_random@B  a size-matched WRONG region, same treatment                    <-- THE DECIDER

All four are single-image, all four go through `LlavaNextProcessor` unmodified, and all four are
matched on **realized** tokens -- counted as occurrences of the image token in `input_ids` after
processing, never computed analytically (that assumption is what faked Phase 17 v1, bug #18).

This asks the production-relevant question directly: **at equal token cost, is it better to tile the
whole image the way the system ships, or to spend those same tiles on the region the question is
about?** Neither Q-CueGraph (matches image AREA) nor RUTA (explicitly not rate-matched) asks it.

MEASURED FLOOR, recorded before running (cf. Qwen's 64 tokens/image, bug #21)
----------------------------------------------------------------------------
`image_grid_pinpoints` for this checkpoint is [[336,672],[672,336],[672,672],[1008,336],[336,1008]]
with 336x336 tiles at 576 tokens each, plus the base view. So LLaVA-NeXT cannot be run below roughly
2x576 = 1152 tokens: **the tiling pathway has a minimum budget on this architecture too.** Budgets
are therefore chosen from what the pinpoint set can actually realize, not from Phase 23's numbers.

PRE-REGISTERED OUTCOMES
-----------------------
* `alloc_query` > `anyres` at matched realized tokens, and > `alloc_random`
      => query-conditional placement beats the shipped high-resolution pathway, on that pathway's
         own model and own format. Phase 23's confounded contrast becomes a real claim.
* `anyres` >= `alloc_query`
      => Phase 23's tiling result WAS a Qwen format artifact. Section 4K's tiling rows get retracted
         and the paper keeps only the grid-alignment finding.
* `alloc_query` ~ `alloc_random`
      => placement is not the operative variable on this architecture; C1 does not transfer.
"""
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase25_llavanext_results.jsonl"
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def expand_to_aspect(box, W, H, ar):
    """Grow a region box to aspect ratio `ar`, centred, clipped to the image.

    WHY THIS IS REQUIRED, not cosmetic. LLaVA-NeXT picks a grid from `image_grid_pinpoints` by
    best fit AND then **unpads** the tile features using the original aspect ratio. So the realized
    token count is a function of aspect, and a crop whose aspect differs from the source image can
    NEVER realize the same count as the full image: measured, crops topped out at 1512 tokens
    against the full image's 2144 -- a 41.8% spread, VOID under the budget gate, even after a full
    ladder search over input scales.

    Matching the aspect makes every arm select the same pinpoint and unpad identically, which is
    the only way to put them on the same budget. The cost is that `alloc_query` becomes "the query
    region expanded to the source aspect ratio" rather than a tight box -- a slightly larger, still
    query-placed region. That is stated as what the arm is; it is not a tight-crop upper bound.
    """
    x0, y0, x1, y1 = box
    w, h = max(1., x1 - x0), max(1., y1 - y0)
    if w / h < ar:
        nw, nh = h * ar, h
    else:
        nw, nh = w, w / ar
    if nw > W or nh > H:                     # cannot honour the aspect inside the image
        sc = min(W / nw, H / nh)
        nw, nh = nw * sc, nh * sc
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0 = max(0., min(cx - nw / 2, W - nw))
    ny0 = max(0., min(cy - nh / 2, H - nh))
    return (nx0, ny0, nx0 + nw, ny0 + nh)


def fit_tokens(measure, img, target, scales=(0.12, 0.2, 0.3, 0.45, 0.65, 0.9, 1.3, 1.9, 2.8, 4.0)):
    """Pick the input scale whose REALIZED image-token count is closest to `target`.

    A multiplicative solver does not work here. LLaVA-NeXT snaps the input to one of five
    `image_grid_pinpoints`, so the token count is a coarse STEP function: measured realizations are
    1176 / 1224 / 2144 / 2242 depending on aspect. The step from one plateau to the next needs a
    scale jump far larger than the ~1.10 factor a proportional update proposes, so the solver sat on
    the first plateau, saw the same value twice and stopped -- leaving crops at 1176 tokens against
    the full image's 2144 (an 82% spread, VOID under the budget gate).

    A fixed geometric ladder crosses every plateau by construction. It costs ~10 processor calls per
    arm instead of ~3, which is CPU-only and worth it: without it the experiment has no matched
    budget and therefore no result.
    """
    W, H = img.size
    best = None
    for sc in scales:
        cur = img.resize((max(32, int(W * sc)), max(32, int(H * sc))), Image.BICUBIC)
        r = measure(cur)
        if best is None or abs(r - target) < abs(best[1] - target):
            best = (cur, r)
        if r == target:
            break
    return best


def main():
    rng = random.Random(107)
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    print(f"V*Bench: {len(ds)} items")

    print("Loading LLaVA-NeXT-vicuna-7b (fp16)...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    img_tok_id = model.config.image_token_index
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]

    def build(img, text):
        prompt = (f"USER: <image>\n{text}\n"
                  "Answer with the option's letter from the given choices directly. ASSISTANT:")
        return processor(images=img, text=prompt, return_tensors="pt")

    def measure(img, text="x"):
        return int((build(img, text)["input_ids"][0] == img_tok_id).sum())

    def run(img, text):
        i = build(img, text).to(model.device)
        realized = int((i["input_ids"][0] == img_tok_id).sum())
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        probs = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return probs.tolist(), int(probs.argmax().item()), realized

    # NO global budget axis. LLaVA-NeXT realizes only a handful of token counts, and which ones
    # depend on the image's aspect ratio, so a fixed ladder of budgets is not something this
    # architecture can honour. Instead each item is matched to ITS OWN native cost: whatever the
    # shipped AnyRes path spends tiling that image, every other arm is fitted to the same number.
    # That is also the production question stated exactly -- "the system spends R tokens tiling
    # this image; what if it spent R on the region the question is about?"
    print("  budget per item = that item's NATIVE AnyRes realization (no global budget axis)")

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
            # every region is expanded to the SOURCE aspect ratio so all arms realize the same
            # token count -- see expand_to_aspect(); without it the budget gate is unsatisfiable
            ar = W / H
            qb = expand_to_aspect((cx0, cy0, cx1, cy1), W, H, ar)
            gt_crop = img.crop(tuple(int(v) for v in qb))
            gw, gh = qb[2] - qb[0], qb[3] - qb[1]
            rx = rng.uniform(0, max(1, W - gw)); ry = rng.uniform(0, max(1, H - gh))
            rnd_crop = img.crop((int(rx), int(ry), int(rx + gw), int(ry + gh)))
            rec_qbox = [round(v, 1) for v in qb]
            if qid_full in done:
                continue

            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]
            rec = {"question_id_full": qid_full, "image": rel, "category": ex["category"],
                   "label": label, "img_wh": [W, H],
                   "bbox_area_frac": ((x1 - x0) * (y1 - y0)) / (W * H),
                   "query_box": None, "native_tokens": None, "arms": {}, "pred": {}, "realized_tokens": {}}
            native = measure(img, text)          # what the shipped path spends on THIS image
            rec["query_box"] = rec_qbox
            rec["native_tokens"] = native
            for B in [native]:
                for name, src in [("uniform", img), ("anyres", img),
                                  ("alloc_query", gt_crop), ("alloc_random", rnd_crop)]:
                    # `uniform` is the model's CHEAPEST whole-image realization -- an unmatched
                    # reference row, since on this architecture tiling IS how a larger budget gets
                    # spent on a whole image. The budget-matched trio is anyres/alloc_query/
                    # alloc_random, all pinned to this item's native cost.
                    tgt = 1 if name == "uniform" else B
                    fitted, _ = fit_tokens(lambda im: measure(im, text), src, tgt)
                    probs, pred, realized = run(fitted, text)
                    rec["arms"][f"{name}@native"] = probs
                    rec["pred"][f"{name}@native"] = pred
                    rec["realized_tokens"][f"{name}@native"] = realized
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
