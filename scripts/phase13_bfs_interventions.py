"""
Phase 13: BREADTH-FIRST sweep of interventions that are NOT cropping.

PURPOSE
-------
Crop-alone recovers 40.4% of confident denials (72.2% on RePOPE-clean items) but confounds two
mechanisms: it REMOVES THE SCENE and it RAISES THE OBJECT'S EFFECTIVE RESOLUTION. Rather than go
deep on one alternative, this sweeps a wide set of cheap interventions in one pass so we can see
which family moves the needle at all, then go depth on the winner.

The arms are chosen so that the two confounded factors are separated by construction:

  CONSTANT RESOLUTION, SCENE REMOVED  (same canvas, same object pixels, only the surround changes)
      black_outside, blur_outside, dim_outside, gray_outside
  RESOLUTION RAISED, SCENE RETAINED   (whole image upscaled; nothing removed)
      upscale2x, upscale3x
  NEITHER (pointing without removing or rescaling)
      redbox            -- draw a rectangle around the object (visual-prompting literature)
      prompt_focus      -- text-only nudge
      prompt_coords     -- text-only, gives the object's normalized coordinates
  REFERENCE
      baseline (full image, 0% by cohort construction) and crop_alone (the known upper bound)

If the constant-resolution arms approach crop_alone => the mechanism is CONTEXT INTERFERENCE, and
the claim is about context, not pixels (which is what makes it not zoom prior art).
If instead the upscale arms approach crop_alone => it is a RESOLUTION story and should be reported
as one.

IMPLEMENTATION GATES (each would silently invalidate an arm)
------------------------------------------------------------
 1. The upscale arms are the ones that fail silently: FINDINGS bug #5 records that `max_pixels` was
    ignored because the fast processor reads `image_processor.size`. If the processor caps the
    upscaled image back down, the arm becomes a no-op and we would read "resolution doesn't matter"
    off a bug. `image_grid_thw` is therefore LOGGED PER ARM PER ITEM, so token counts are
    checkable after the fact rather than assumed.
 2. Black is NOT neutral for this model (Phase 8: it asserts presence on black images at 0.74-0.99).
    A large black surround is its own distribution shift. blur/dim/gray are the companion arms that
    remove scene CONTENT while keeping image statistics closer to natural. If black disagrees with
    blur, trust blur and report both.
 3. Object pixel geometry is byte-identical across baseline and all constant-resolution arms: same
    canvas, same bbox region untouched, only the surround is modified.
 4. Negative arm (object genuinely absent) for every arm, using a size-matched random region whose
    coordinates are generated ONCE per uid and reused across all arms. Every recovery number needs
    its false-positive rate -- Phase 10's lesson is that a bias-shifted arm looks like a win on raw
    recovery (the other-scene arm: +17.0pp raw, CI spanning zero on discrimination).

Results are reported both on ALL items and on the RePOPE-clean subset (Phase 11), which is free.
"""
import json
import re
import sys
import time
import random

import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import load_bbox_lookup
from phase7_vision_zoom import union_bbox, padded_crop_box, PAD_FRAC

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase13_bfs_results.jsonl"
ANSWER_SUFFIX = " Please answer this question with yes or no."


def extract_category(q):
    m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", q, re.I)
    return m.group(1).strip().lower() if m else None


def surround_op(img, box, mode):
    """Return a copy of `img` with everything OUTSIDE `box` modified. The region inside `box` is
    pasted back untouched, so object pixels are byte-identical to baseline."""
    x0, y0, x1, y1 = [int(v) for v in box]
    region = img.crop((x0, y0, x1, y1))
    if mode == "black":
        out = Image.new("RGB", img.size, (0, 0, 0))
    elif mode == "blur":
        r = max(4, int(0.03 * max(img.size)))
        out = img.filter(ImageFilter.GaussianBlur(radius=r))
    elif mode == "dim":
        out = ImageEnhance.Brightness(img).enhance(0.25)
    elif mode == "gray":
        out = ImageEnhance.Color(img).enhance(0.0)
    else:
        raise ValueError(mode)
    out.paste(region, (x0, y0))
    return out


def redbox(img, box):
    out = img.copy()
    d = ImageDraw.Draw(out)
    w = max(3, int(0.005 * max(img.size)))
    d.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])], outline=(255, 0, 0), width=w)
    return out


def main():
    rng = random.Random(41)
    p7 = [json.loads(l) for l in open(f"{DATA}/phase7_vision_zoom_results.jsonl")]
    pos_uids = [r["uid"] for r in p7]
    crop_pool = [(r["crop_w"], r["crop_h"]) for r in p7]
    p4 = [json.loads(l) for l in open(f"{DATA}/phase4_localize_results.jsonl")]
    neg_uids = [r["uid"] for r in p4 if r["group"] == "negative_control"]
    neg_uids = rng.sample(neg_uids, min(len(neg_uids), len(pos_uids)))

    recs = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{DATA}/phase1_results_qwen_dedup.jsonl")}
    for r in recs.values():
        if r["label"] == "no":
            r["category"] = extract_category(r["question"])

    done = set()
    try:
        for line in open(OUT_PATH):
            done.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done)} done")
    except FileNotFoundError:
        pass
    targets = [(u, "positive") for u in pos_uids] + [(u, "negative") for u in neg_uids]
    targets = [t for t in targets if t[0] not in done]
    if not targets:
        print("Nothing to do.")
        return
    print(f"targets: {len(targets)}")

    bb, isz = load_bbox_lookup()
    want = {u for u, _ in targets}
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

    def run(image, question):
        """One forward pass -> (P(yes), image_grid_thw). Grid is logged so the upscale arms can be
        verified to have actually produced more visual tokens (gate 1)."""
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question + ANSWER_SUFFIX}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        yl = torch.logsumexp(logits[yes_ids], dim=0)
        nl = torch.logsumexp(logits[no_ids], dim=0)
        p = torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()
        return p, inputs["image_grid_thw"][0].tolist()

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for uid, group in targets:
            item, rf = items.get(uid), recs.get(uid)
            if item is None or rf is None or not rf.get("category"):
                continue
            img = item["image"].convert("RGB")
            key = (rf["split"], str(rf["question_id"]))
            W, H = isz.get(key, img.size)
            q = item["question"]

            if group == "positive":
                boxes = bb.get(key)
                if not boxes:
                    continue
                x0, y0, x1, y1 = union_bbox(boxes)
                cbox = padded_crop_box(x0, y0, x1, y1, W, H, PAD_FRAC)
            else:
                cw, ch = rng.choice(crop_pool)
                cw, ch = min(cw, W), min(ch, H)
                bx = rng.uniform(0, max(1.0, W - cw)); by = rng.uniform(0, max(1.0, H - ch))
                cbox = (bx, by, bx + cw, by + ch)
            if cbox[2] - cbox[0] < 2 or cbox[3] - cbox[1] < 2:
                continue

            arms, grids = {}, {}
            def add(name, image, question=q):
                p, g = run(image, question)
                arms[name] = p; grids[name] = g

            add("baseline", img)
            add("crop_alone", img.crop(tuple(int(v) for v in cbox)))
            for m in ("black", "blur", "dim", "gray"):
                add(f"{m}_outside", surround_op(img, cbox, m))
            for s in (2, 3):
                add(f"upscale{s}x", img.resize((img.width * s, img.height * s), Image.BICUBIC))
            add("redbox", redbox(img, cbox))
            add("prompt_focus", img,
                q + " Look very carefully, including at small objects in the periphery.")
            nx0, ny0 = cbox[0] / W * 1000, cbox[1] / H * 1000
            nx1, ny1 = cbox[2] / W * 1000, cbox[3] / H * 1000
            add("prompt_coords", img,
                f"{q} Pay particular attention to the region with bounding box "
                f"[{nx0:.0f}, {ny0:.0f}, {nx1:.0f}, {ny1:.0f}] in 0-1000 normalized coordinates.")

            fout.write(json.dumps({
                "uid": uid, "group": group, "category": rf["category"], "split": rf["split"],
                "question_id": rf["question_id"], "pixel_area_frac": rf.get("pixel_area_frac"),
                "box": list(cbox), "img_wh": [W, H], "arms": arms, "grids": grids}) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(targets)}] {n/el:.2f} it/s eta={(len(targets)-n)/(n/el)/60:.1f}min",
                      flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
