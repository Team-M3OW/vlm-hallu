"""
Phase 7: vision-representation zoom intervention -- the vision-encoder-side causal test that
follows from three ruled-out decoder-side hypotheses (Phase 5 logit-lens layer readout, Phase 6d
decoding depth, Phase 6e attention patching, all falsified/inconclusive as fixes). If the true
bottleneck is that a small/peripheral object only occupies a handful of low-effective-resolution
patches once the full scene is downsampled to the model's input grid, the vision encoder may simply
never build a clear representation of it -- no decoder-side intervention (which patch, which layer,
how many decode steps) can fix a representation that was never formed well upstream.

Test: give the model a SECOND image alongside the original -- a crop of the SAME image, tightly
around the object's own ground-truth region, upscaled by Qwen's own image processor to occupy far
more patches than it did embedded in the full scene. This adds ZERO new information (it's the same
pixels, same photograph, no external data, no dataset construction) -- it only changes how many
patches / how much effective resolution the vision encoder spends on that region. If P(yes)
recovers, the bottleneck is representation/resolution, not attention or decoding -- and this is
itself a working, no-finetuning mitigation (an actual fix, not just a diagnosis).

Three conditions per item (same confident_denial cohort as Phase 6/6d/6e, n=141):
  1. baseline     -- single full image, standard forced yes/no prompt (matches Phase 6's baseline).
  2. oracle_zoom  -- full image + a crop tightly around the TRUE object's own bbox (padded 25%),
                     as a second image, with a generic (non-leaking) connecting sentence.
  3. random_zoom  -- full image + a crop of the SAME SIZE from a RANDOM other region of the image
                     (not overlapping the object), otherwise identical -- controls for "any second
                     zoomed-in image helps, regardless of content" (the same confound structure
                     that sank Phase 6d's filler-vs-cot comparison).

If oracle_zoom recovers P(yes) substantially more than random_zoom, that is well-controlled
evidence for the vision-representation-quality account. If both recover similarly (or neither
does), the bottleneck isn't resolution either, and the null gets logged honestly, same discipline
as every other phase.
"""
import json
import sys
import time
import random
import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import Ctx, load_bbox_lookup

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase7_vision_zoom_results.jsonl"
PAD_FRAC = 0.25
CONNECTOR_TEXT = "Here is a zoomed-in crop of a region from the same image."


def union_bbox(bboxes):
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[0] + b[2] for b in bboxes)
    y1 = max(b[1] + b[3] for b in bboxes)
    return x0, y0, x1, y1


def padded_crop_box(x0, y0, x1, y1, img_w, img_h, pad_frac):
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = w * pad_frac, h * pad_frac
    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(img_w, x1 + pad_x)
    cy1 = min(img_h, y1 + pad_y)
    return cx0, cy0, cx1, cy1


def random_other_box(crop_w, crop_h, obj_box, img_w, img_h, rng, max_tries=50):
    ox0, oy0, ox1, oy1 = obj_box
    for _ in range(max_tries):
        rx0 = rng.uniform(0, max(1.0, img_w - crop_w))
        ry0 = rng.uniform(0, max(1.0, img_h - crop_h))
        rx1, ry1 = rx0 + crop_w, ry0 + crop_h
        overlap = not (rx1 < ox0 or rx0 > ox1 or ry1 < oy0 or ry0 > oy1)
        if not overlap:
            return rx0, ry0, rx1, ry1
    return rx0, ry0, rx1, ry1  # fall back to last (rare, small images)


def p_yes(ctx, images, question, with_connector):
    if with_connector:
        content = [{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT}, {"type": "image"},
                   {"type": "text", "text": f"{question} Please answer this question with yes or no."}]
    else:
        content = [{"type": "image"},
                   {"type": "text", "text": f"{question} Please answer this question with yes or no."}]
    messages = [{"role": "user", "content": content}]
    text = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ctx.processor(images=images, text=text, return_tensors="pt").to(ctx.model.device)
    with torch.no_grad():
        out = ctx.model(**inputs)
    logits = out.logits[0, -1]
    yl = torch.logsumexp(logits[ctx.yes_ids], dim=0)
    nl = torch.logsumexp(logits[ctx.no_ids], dim=0)
    return torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()


def main():
    with open(f"{DATA}/phase6_attention_results.jsonl") as f:
        denials = [r for r in (json.loads(l) for l in f) if r["group"] == "confident_denial"]
    print(f"confident_denial cohort: {len(denials)}")

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs_by_uid = {(d := json.loads(l))["uid"]: d for l in f}

    bbox_by_key, imgsize_by_key = load_bbox_lookup()

    target_uids = {r["uid"] for r in denials}
    print("Loading items...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)}")

    done_uids = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done_uids)} already done")
    except FileNotFoundError:
        pass

    print("Loading Qwen3-VL-2B (fp16)...")
    ctx = Ctx()
    # Phase 7 doesn't need attention output -- eager attention (set up for Phase 6's Ctx) still
    # works fine here, just costs a bit of unused memory bookkeeping; not worth a separate Ctx.

    rng = random.Random(17)
    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r in denials:
            uid = r["uid"]
            if uid in done_uids:
                continue
            item = items_by_uid.get(uid)
            if item is None:
                continue
            rfull = recs_by_uid.get(uid)
            key = (rfull["split"], str(rfull["question_id"]))
            bboxes = bbox_by_key.get(key)
            imgsize = imgsize_by_key.get(key)
            if not bboxes or imgsize is None:
                continue
            img = item["image"].convert("RGB")
            img_w, img_h = imgsize

            x0, y0, x1, y1 = union_bbox(bboxes)
            cx0, cy0, cx1, cy1 = padded_crop_box(x0, y0, x1, y1, img_w, img_h, PAD_FRAC)
            crop_w, crop_h = cx1 - cx0, cy1 - cy0
            if crop_w < 2 or crop_h < 2:
                continue
            oracle_crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))

            rx0, ry0, rx1, ry1 = random_other_box(crop_w, crop_h, (x0, y0, x1, y1), img_w, img_h, rng)
            random_crop = img.crop((int(rx0), int(ry0), int(rx1), int(ry1)))

            base_py = p_yes(ctx, img, item["question"], with_connector=False)
            oracle_py = p_yes(ctx, [img, oracle_crop], item["question"], with_connector=True)
            random_py = p_yes(ctx, [img, random_crop], item["question"], with_connector=True)

            rec = {"uid": uid, "category": r["category"], "pixel_area_frac": r["pixel_area_frac"],
                   "crop_w": crop_w, "crop_h": crop_h,
                   "baseline_p_yes": base_py, "oracle_zoom_p_yes": oracle_py,
                   "random_zoom_p_yes": random_py}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
