"""
Phase 17 (E1) -- THE CENTERPIECE: budget-matched query-conditional token allocation.

THE QUESTION
------------
"More visual tokens helps" is NOT a finding -- AnyRes and the whole dynamic-resolution line already
assert it. The claim this project can actually make is different:

    At a FIXED TOTAL visual-token budget, does allocating tokens BY QUERY beat allocating them
    UNIFORMLY over the image?

If yes, the flaw is the ALLOCATOR -- token layout is fixed by image geometry before the model sees
the question -- and that is an architectural claim, not a preprocessing hack. It is also the claim
the four nulls support (pointing 0.0%, redbox 5.6%, attention patching 6e, steering 16): you cannot
redirect attention to capacity that was never allocated.

ARMS (all at the same realized budget B)
----------------------------------------
    uniform@B       whole image, downsampled to hit B                      -- the baseline allocator
    alloc_query@B   low-res full image (B/2) + high-res crop of the GT region (B/2)
    alloc_random@B  IDENTICAL layout, WRONG region                         <-- THE DECIDING CONTROL
    crop_only@B     the region crop alone, sized to hit B                  -- upper bound; discards
                                                                              the scene, so it is
                                                                              not a viable allocator
B in {150, 300 (~native), 600}.

WHY alloc_random DECIDES IT
---------------------------
Same non-uniform layout, same realized budget, region chosen wrong. If alloc_query beats it, the win
is ALLOCATION. If not, the win is merely "non-uniform layouts happen to help", which is a much
weaker paper. This is Phase 16's `rand_obj` lesson (where a norm-matched random direction matched
the real one exactly) moved to the input level.

NON-NEGOTIABLES
---------------
 1. MATCH ON MEASURED TOKENS, NOT INTENDED. `image_grid_thw` is logged per arm per item and the
    REALIZED total is what gets reported. An arm landing at 340 vs 300 is not budget-matched.
 2. The uniform arm sits at the SAME B -- it is downsampled to the target like everything else, and
    never allowed to keep its native-resolution advantage.
 3. Negative arm (genuine absences) for every cell. For negatives there is no object, so BOTH
    alloc arms use a random region; their FP rates are therefore matched by construction, which
    makes the positives-side comparison a clean paired contrast.
 4. ORACLE FRAMING, stated: the region comes from COCO ground truth, exactly as Phase 7. Legitimate
    for a mechanism claim, and it is what keeps this out of re-inventing V*/SEAL's localizer.

Cohort: RePOPE-clean (Phase 11), P(yes) re-identified in fp16 by Phase 12 (bug #6).

TOKEN GEOMETRY: Qwen3-VL merges 2x2 patches of 14px, so one merged token covers 28x28px of the
RESIZED image. To hit B merged tokens, target a resized area of B*28*28 preserving aspect ratio,
then round each side to a multiple of 28. Verified against the processor's own `image_grid_thw`.
"""
import json
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase7_vision_zoom import union_bbox, padded_crop_box, PAD_FRAC, CONNECTOR_TEXT
from phase15_category_specificity import build_full_lookup

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase17_budget_results.jsonl"
BUDGETS = [150, 300, 600]
TOK_PX = 28            # one merged token covers 28x28 px of the resized image
ANSWER_SUFFIX = " Please answer this question with yes or no."
N_NEG = 160


def _resize(img, scale):
    W, H = img.size
    w = max(TOK_PX, int(round(W * scale / TOK_PX)) * TOK_PX)
    h = max(TOK_PX, int(round(H * scale / TOK_PX)) * TOK_PX)
    return img.resize((w, h), Image.BICUBIC)


def fit_to_budget(measure, imgs, target, iters=6, tol=0.03):
    """Calibrate a scale factor so the REALIZED merged-token count (as reported by the processor's
    own image_grid_thw) matches `target`.

    An analytic solve is NOT sufficient: the processor applies its own smart_resize with min/max
    pixel constraints and overrode our dimensions badly in v1 (asked 150 tokens, got 120 for the
    uniform arm but 143 for the two-image arms -- a 19% budget advantage to the arm we are trying to
    show wins, which would have invalidated the whole experiment). So we measure and iterate.
    Returns (resized_images, realized_tokens)."""
    base_area = sum(im.size[0] * im.size[1] for im in imgs)
    scale = ((target * TOK_PX * TOK_PX) / base_area) ** 0.5
    best = None
    for _ in range(iters):
        cur = [_resize(im, scale) for im in imgs]
        realized = measure(cur)
        if best is None or abs(realized - target) < abs(best[1] - target):
            best = (cur, realized)
        if realized == 0:
            break
        if abs(realized - target) / target <= tol:
            return cur, realized
        scale *= (target / realized) ** 0.5
    return best


def main():
    rng = random.Random(83)
    clean = [json.loads(l) for l in open(f"{DATA}/phase12_clean_pyes.jsonl")]
    denials = [r for r in clean if r["label"] == "yes" and r["p_yes_fp16"] < 0.01]
    negs = rng.sample([r for r in clean if r["label"] == "no"], N_NEG)
    print(f"RePOPE-clean fp16 confident denials: {len(denials)}   clean negatives: {len(negs)}")

    lut = build_full_lookup()
    want = {r["uid"] for r in denials} | {r["uid"] for r in negs}
    print("Loading items...")
    items = {it["uid"]: it for it in build_items() if it["uid"] in want}
    print(f"  resolved {len(items)}/{len(want)}")

    print("Loading Qwen3-VL-2B (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                      for s in ["yes", "Yes", " yes", " Yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                     for s in ["no", "No", " no", " No"]})

    def _chat(images, question, connector):
        if len(images) == 1:
            content = [{"type": "image"}]
        else:
            content = ([{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                        {"type": "image"}] if connector
                       else [{"type": "image"}, {"type": "image"}])
        content.append({"type": "text", "text": question + ANSWER_SUFFIX})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    def measure(images, question="x", connector=True):
        """Realized merged-token total, from the processor only -- no GPU forward."""
        inputs = processor(images=images, text=_chat(images, question, connector),
                           return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in inputs["image_grid_thw"].tolist()))

    def run(images, question, connector):
        inputs = processor(images=images, text=_chat(images, question, connector),
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**inputs).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                       torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()
        realized = int(sum(g[1] * g[2] // 4 for g in inputs["image_grid_thw"].tolist()))
        return p, realized

    targets = [(r, "denial") for r in denials] + [(r, "negative") for r in negs]
    done = set()
    try:
        for line in open(OUT_PATH):
            done.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass
    targets = [t for t in targets if t[0]["uid"] not in done]

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for r, group in targets:
            it = items.get(r["uid"])
            if it is None:
                continue
            img = it["image"].convert("RGB")
            # NOTE: build_full_lookup() is keyed off pope_coco_area_joined.json, which contains
            # POSITIVES ONLY. v1 required it for every item and silently dropped all 160 negatives
            # -- i.e. the entire false-positive arm. Negatives need no lookup: they have no object.
            ent = lut.get((r["split"], str(r["question_id"])))
            if group == "denial":
                if ent is None:
                    continue
                (W, H), by_cat, qcat = ent
                bxs = by_cat.get(qcat) if qcat is not None else None
                if not bxs:
                    continue
                x0, y0, x1, y1 = union_bbox(bxs)
                gt_box = padded_crop_box(x0, y0, x1, y1, W, H, PAD_FRAC)
            else:
                # no object exists -> a random region, same as the control arm. Both alloc arms
                # therefore share an FP rate by construction, which makes the positives-side
                # alloc_query vs alloc_random contrast a clean paired comparison.
                W, H = img.size
                bw, bh = W * rng.uniform(.08, .35), H * rng.uniform(.08, .35)
                gx = rng.uniform(0, max(1, W - bw)); gy = rng.uniform(0, max(1, H - bh))
                gt_box = (gx, gy, gx + bw, gy + bh)
            if gt_box[2] - gt_box[0] < 4 or gt_box[3] - gt_box[1] < 4:
                continue
            gw, gh = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
            # size-matched WRONG region for the deciding control
            rx = rng.uniform(0, max(1, W - gw)); ry = rng.uniform(0, max(1, H - gh))
            rnd_box = (rx, ry, rx + gw, ry + gh)

            gt_crop = img.crop(tuple(int(v) for v in gt_box))
            rnd_crop = img.crop(tuple(int(v) for v in rnd_box))

            rec = {"uid": r["uid"], "group": group, "split": r["split"],
                   "question_id": r["question_id"], "category": r.get("category"),
                   "pixel_area_frac": r.get("pixel_area_frac"),
                   "baseline_p_yes": r["p_yes_fp16"],
                   "gt_box": list(gt_box), "rnd_box": list(rnd_box),
                   "arms": {}, "realized_tokens": {}}
            q = it["question"]
            for B in BUDGETS:
                cells = {
                    "uniform":      ([img], False),
                    "crop_only":    ([gt_crop], False),
                    "alloc_query":  ([img, gt_crop], True),
                    "alloc_random": ([img, rnd_crop], True),
                }
                for name, (imgs, conn) in cells.items():
                    fitted, _ = fit_to_budget(lambda ims: measure(ims, q, conn), imgs, B)
                    p, realized = run(fitted, q, conn)
                    rec["arms"][f"{name}@{B}"] = p
                    rec["realized_tokens"][f"{name}@{B}"] = realized
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(targets)}] {n/el:.2f} it/s "
                      f"eta={(len(targets)-n)/(n/el)/60:.1f}min", flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
