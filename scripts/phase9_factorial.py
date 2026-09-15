"""
Phase 9 (E2): the 2x2 factorial that closes the metric mismatch in the readout-vs-resolution
head-to-head.

WHY THIS EXISTS
---------------
Comparing Phase 7 (oracle zoom recovers 13.3% of confident denials) against Phase 4 (grounding
recovers 28.9% at IoU>0.5 on the same items) is suggestive but NOT a clean comparison: 13.3% is an
ASSERTION event (P(yes)>0.5) and 28.9% is a LOCALIZATION event (IoU>0.5). Different events. This
project's §6 near-miss table is a list of exactly this kind of mismatch caught before it became a
headline, so it gets closed rather than argued around.

The fix is not to patch the comparison but to make it a proper factorial:

                        |  yes/no readout   |  grounding readout
    --------------------+-------------------+---------------------------------
    original pixels     |  0% by constr.    |  Phase 4  (assert + IoU)
    zoomed (oracle crop)|  Phase 7: 13.3%   |  <-- THIS SCRIPT
                        |  + here, crop-alone

Common metric in EVERY cell = "asserts the object is present". For the yes/no readout that is
P(yes)>0.5; for the grounding readout it is "emits a bbox rather than 'There are none.'". IoU>0.5 is
reported as the stricter secondary wherever the grounding channel is used. This separates the
resolution factor from the readout factor and gives their interaction, which no single ratio can.

CONDITIONS (positives = the Phase 7 confident-denial cohort, all with GT boxes)
-------------------------------------------------------------------------------
Two zoom formats are run, because they trade off against each other and neither alone is sufficient:

  * TWO-IMAGE  [full image + connector + oracle crop] -- format-identical to Phase 7, so its
    grounding cell is directly comparable to Phase 7's yes/no cell. But bbox_2d is ambiguous as to
    WHICH image it indexes, so only the assertion metric is read from this format.
  * CROP-ALONE [oracle crop only] -- unambiguous coordinates, so IoU is computable (GT box mapped
    into crop coordinates). This is also the purer resolution manipulation: single image in both
    rows of the factorial, only effective resolution differs.

  pos conditions: crop_alone_yesno, crop_alone_ground, twoimg_ground,
                  randcrop_alone_yesno, randcrop_alone_ground   (last two = content control)

NEGATIVE ARM (mandatory -- no cell is reported bare)
---------------------------------------------------
The grounding channel has a large presence-INDEPENDENT emission prior: Phase 4's negative_control
emitted boxes 45.3% of the time, and Phase 8's blank-image readout sat at 0.74-0.99 on BLACK images.
So an emission rate means nothing without its false-positive rate at the same resolution.

Negatives (object genuinely absent) have no GT box, so there is no "oracle" crop for them -- the
matched control is a RANDOM crop whose size is drawn from the positives' oracle-crop size
distribution. This measures: when you zoom a VLM into a region and ask it to localize an absent
object, how often does it invent one? That false-positive rate is what makes the positive-arm
numbers interpretable.

  neg conditions: crop_alone_yesno, crop_alone_ground  (on a size-matched random crop)

IoU CIRCULARITY (carried over from FINDINGS §2.3): absent objects have no GT box, so IoU==0 for the
negative arm by construction. The non-circular signal on the negative arm is EMISSION, not IoU.
"""
import json
import re
import sys
import time
import random
from collections import defaultdict

import torch

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase4_localize_vs_answer import extract_boxes, iou
from phase6_attention_entanglement import load_bbox_lookup
from phase7_vision_zoom import union_bbox, padded_crop_box, random_other_box, PAD_FRAC, CONNECTOR_TEXT

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase9_factorial_results.jsonl"
GROUND_PROMPT = "Locate the {obj} in the image and output its bounding box coordinates."
ANSWER_SUFFIX = " Please answer this question with yes or no."
BOX_TOK, ABSTAIN_TOK = 73594, 3862  # verified over all 486 Phase 4 generations; asserted below


def extract_category(question):
    m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", question, re.I)
    return m.group(1).strip().lower() if m else None


def main():
    rng = random.Random(23)

    # ---- positive arm: the Phase 7 cohort (confident denials with usable GT crops) ----
    with open(f"{DATA}/phase7_vision_zoom_results.jsonl") as f:
        p7 = [json.loads(l) for l in f]
    pos_uids = [r["uid"] for r in p7]
    crop_size_pool = [(r["crop_w"], r["crop_h"]) for r in p7]

    # ---- negative arm: Phase 4's negative_control (object absent, confidently denied) ----
    with open(f"{DATA}/phase4_localize_results.jsonl") as f:
        p4 = [json.loads(l) for l in f]
    neg_uids = [r["uid"] for r in p4 if r["group"] == "negative_control"]
    neg_uids = rng.sample(neg_uids, min(len(neg_uids), len(pos_uids)))

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs_by_uid = {(d := json.loads(l))["uid"]: d for l in f}
    for r in recs_by_uid.values():
        if r["label"] == "no":
            r["category"] = extract_category(r["question"])

    print(f"positive arm (confident denials, oracle crop available): {len(pos_uids)}")
    print(f"negative arm (object absent, size-matched random crop):  {len(neg_uids)}")

    done = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done)} already done")
    except FileNotFoundError:
        pass

    targets = [(u, "positive") for u in pos_uids] + [(u, "negative") for u in neg_uids]
    targets = [(u, g) for u, g in targets if u not in done]
    if not targets:
        print("Nothing to do.")
        return

    bbox_by_key, imgsize_by_key = load_bbox_lookup()

    print("Loading items...")
    want = {u for u, _ in targets}
    items_by_uid = {it["uid"]: it for it in build_items() if it["uid"] in want}
    print(f"  resolved {len(items_by_uid)}/{len(want)}")

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
    assert tok("```", add_special_tokens=False)["input_ids"][0] == BOX_TOK
    assert tok("There", add_special_tokens=False)["input_ids"][0] == ABSTAIN_TOK

    def prompt_for(images, text, connector):
        if connector:
            content = [{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                       {"type": "image"}, {"type": "text", "text": text}]
        else:
            content = [{"type": "image"}, {"type": "text", "text": text}]
        chat = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        return processor(images=images, text=chat, return_tensors="pt").to(model.device)

    def do_yesno(images, question, connector=False):
        inputs = prompt_for(images, question + ANSWER_SUFFIX, connector)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        yl = torch.logsumexp(logits[yes_ids], dim=0)
        nl = torch.logsumexp(logits[no_ids], dim=0)
        return torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()

    def do_ground(images, category, connector=False):
        """Full generation + the first-token presence readout, from ONE generate() call
        (scores[0] holds the logits at the first generated position)."""
        inputs = prompt_for(images, GROUND_PROMPT.format(obj=category), connector)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
        seq = out.sequences[0][inputs["input_ids"].shape[1]:]
        gen = processor.decode(seq, skip_special_tokens=True)
        first = out.scores[0][0].float()
        two = torch.softmax(torch.stack([first[BOX_TOK], first[ABSTAIN_TOK]]), dim=0)[0].item()
        return {"gen": gen[:300], "boxes": extract_boxes(gen), "p_ground": two,
                "mass": float(torch.softmax(first, dim=0)[[BOX_TOK, ABSTAIN_TOK]].sum())}

    def to_px(boxes, w, h):
        return [[b[0]*w/1000, b[1]*h/1000, b[2]*w/1000, b[3]*h/1000] for b in boxes]

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for uid, group in targets:
            item = items_by_uid.get(uid)
            rfull = recs_by_uid.get(uid)
            if item is None or rfull is None:
                continue
            cat = rfull.get("category")
            if not cat:
                continue
            img = item["image"].convert("RGB")
            key = (rfull["split"], str(rfull["question_id"]))
            imgsize = imgsize_by_key.get(key)
            img_w, img_h = imgsize if imgsize else img.size

            if group == "positive":
                bboxes = bbox_by_key.get(key)
                if not bboxes:
                    continue
                x0, y0, x1, y1 = union_bbox(bboxes)
                cx0, cy0, cx1, cy1 = padded_crop_box(x0, y0, x1, y1, img_w, img_h, PAD_FRAC)
                obj_box = (x0, y0, x1, y1)
            else:
                # no object => no oracle crop exists. Size-matched random crop instead.
                cw, ch = rng.choice(crop_size_pool)
                cw, ch = min(cw, img_w), min(ch, img_h)
                cx0 = rng.uniform(0, max(1.0, img_w - cw)); cy0 = rng.uniform(0, max(1.0, img_h - ch))
                cx1, cy1 = cx0 + cw, cy0 + ch
                obj_box = None
            crop_w, crop_h = cx1 - cx0, cy1 - cy0
            if crop_w < 2 or crop_h < 2:
                continue
            crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))

            rec = {"uid": uid, "group": group, "category": cat, "split": rfull["split"],
                   "question_id": rfull["question_id"],
                   "pixel_area_frac": rfull.get("pixel_area_frac"),
                   "crop_w": crop_w, "crop_h": crop_h}

            # ---- ZOOMED row, both readouts ----
            rec["crop_alone_yesno"] = do_yesno(crop, item["question"])
            g = do_ground(crop, cat)
            # GT box expressed in CROP pixel coordinates (positives only; absent objects have none)
            gt_crop = []
            if group == "positive":
                for b in bbox_by_key.get(key, []):
                    gx0, gy0 = b[0] - cx0, b[1] - cy0
                    gx1, gy1 = b[0] + b[2] - cx0, b[1] + b[3] - cy0
                    gt_crop.append([max(0, gx0), max(0, gy0), min(crop_w, gx1), min(crop_h, gy1)])
            pred = to_px(g["boxes"], crop_w, crop_h)
            rec["crop_alone_ground"] = {
                "n_boxes": len(g["boxes"]), "p_ground": g["p_ground"], "mass": g["mass"],
                "best_iou": max([iou(p, q) for p in pred for q in gt_crop], default=0.0),
                "gen": g["gen"]}

            g2 = do_ground([img, crop], cat, connector=True)  # format-matched to Phase 7
            rec["twoimg_ground"] = {"n_boxes": len(g2["boxes"]), "p_ground": g2["p_ground"],
                                    "mass": g2["mass"], "gen": g2["gen"]}

            # ---- content control: a random crop from the SAME image (positives only;
            #      the negative arm's crop is already random by construction) ----
            if group == "positive":
                rx0, ry0, rx1, ry1 = random_other_box(crop_w, crop_h, obj_box, img_w, img_h, rng)
                rcrop = img.crop((int(rx0), int(ry0), int(rx1), int(ry1)))
                rec["randcrop_alone_yesno"] = do_yesno(rcrop, item["question"])
                g3 = do_ground(rcrop, cat)
                rec["randcrop_alone_ground"] = {"n_boxes": len(g3["boxes"]),
                                                "p_ground": g3["p_ground"], "gen": g3["gen"]}

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(targets)}] {n/el:.2f} it/s eta={(len(targets)-n)/(n/el)/60:.1f}min",
                      flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
