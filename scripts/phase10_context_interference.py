"""
Phase 10 (E3'): disentangling the two-image vs crop-alone gap.

THE OBSERVATION TO EXPLAIN
--------------------------
On the SAME 135 confident denials, with the SAME oracle crop and the SAME model:
    crop alone                                  -> 37.8% recover (P(yes)>0.5)
    two-image [full + connector + crop]  (Ph.7) -> 13.3% recover
    paired diff CI [+17.0, +31.9] pp, 100% of resamples favor crop-alone.
Not a format artifact: random-crop-alone gives 5.7%, and the false-positive rate on genuinely
absent objects is 2.8%.

The information is present in BOTH conditions. Yet adding the full scene to a prompt that already
contains a sufficient high-resolution crop destroys ~2/3 of the recovery. But the two conditions
differ in THREE ways at once -- full-scene presence, image count, and the connector sentence -- so
the gap is not yet attributable. This script varies them one at a time.

ARMS (identical items, identical crop, yes/no readout, P(yes)>0.5 throughout)
----------------------------------------------------------------------------
  full_alone        full image only                        -- 0% by cohort construction (sanity)
  crop_alone        crop only, no connector                -- the strong condition
  crop_conn         crop only + connector sentence         -- isolates CONNECTOR TEXT
  crop_crop         [crop + conn + crop] (duplicated)      -- isolates IMAGE COUNT (see note)
  blank_crop        [black + conn + crop]                  -- image count w/ no scene (see note)
  full_crop_conn    [full + conn + crop]                   -- Phase 7 replication
  full_crop_noconn  [full + crop], no connector            -- connector WITHIN the two-image format
  crop_full_conn    [crop + conn + full] (order swapped)   -- isolates POSITION of the full scene
  other_full_crop   [DIFFERENT image's full scene + conn + crop] -- isolates SCENE IDENTITY

Note on the image-count control: `blank_crop` is NOT clean on its own -- Phase 8 showed this model
asserts presence on black images (grounding readout 0.74-0.99), so black is not inert input. The
load-bearing image-count control is `crop_crop`: image count held at two, zero added scene content,
zero novel pixels.

Note on `other_full_crop`: the connector sentence ("...from the same image") is literally false in
this arm. Held fixed anyway, because varying the text would reintroduce the connector confound.
This is CONSERVATIVE with respect to the hypothesis: if the model notices the mismatch and is
disrupted, that pushes this arm's recovery DOWN, i.e. against the prediction below.

PRE-REGISTERED PREDICTIONS (written before any number was seen)
--------------------------------------------------------------
  SUPPORTS "the full scene re-anchors the model on its own failed percept":
      crop_alone ~= crop_conn ~= crop_crop  >>  full_crop_conn ~= full_crop_noconn,
      with other_full_crop landing STRICTLY BETWEEN the two clusters.
  REFUTES it (generic multi-image dilution instead):
      other_full_crop ~= full_crop_conn. Then this is a finding about multi-image prompting, not
      about re-anchoring, and it gets reported that way rather than reframed.
  ALSO REFUTES the whole attribution:
      crop_crop ~= full_crop_conn  =>  the gap is image count, not scene content at all.

READING CAUTION: every arm uses the yes/no readout on a cohort selected for P(yes)<0.01, so the
floor is zero BY CONSTRUCTION. Rates are valid for comparing arms TO EACH OTHER; no single arm's
rate is a calibrated capability estimate. The negative arm's false-positive rate travels with every
recovery number quoted.

PAIRING: each item's crop box is generated ONCE and reused across all arms (and its coordinates are
stored, unlike Phase 9), so any between-arm difference cannot be crop-location noise.
"""
import json
import re
import sys
import time
import random

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import load_bbox_lookup
from phase7_vision_zoom import union_bbox, padded_crop_box, PAD_FRAC, CONNECTOR_TEXT

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase10_context_interference_results.jsonl"
ANSWER_SUFFIX = " Please answer this question with yes or no."


def extract_category(q):
    m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", q, re.I)
    return m.group(1).strip().lower() if m else None


def main():
    rng = random.Random(31)

    p7 = [json.loads(l) for l in open(f"{DATA}/phase7_vision_zoom_results.jsonl")]
    pos_uids = [r["uid"] for r in p7]
    crop_size_pool = [(r["crop_w"], r["crop_h"]) for r in p7]

    p4 = [json.loads(l) for l in open(f"{DATA}/phase4_localize_results.jsonl")]
    neg_uids = [r["uid"] for r in p4 if r["group"] == "negative_control"]
    neg_uids = rng.sample(neg_uids, min(len(neg_uids), len(pos_uids)))

    recs = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{DATA}/phase1_results_qwen_dedup.jsonl")}
    for r in recs.values():
        if r["label"] == "no":
            r["category"] = extract_category(r["question"])

    print(f"positives (confident denials): {len(pos_uids)}   negatives (absent): {len(neg_uids)}")

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

    bb, isz = load_bbox_lookup()
    print("Loading items...")
    want = {u for u, _ in targets}
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

    def p_yes(images, question, connector):
        """images: list. connector=True inserts CONNECTOR_TEXT between image 1 and image 2."""
        if len(images) == 1:
            content = [{"type": "image"}]
            if connector:
                content.append({"type": "text", "text": CONNECTOR_TEXT})
        else:
            content = [{"type": "image"}]
            if connector:
                content.append({"type": "text", "text": CONNECTOR_TEXT})
            content.append({"type": "image"})
        content.append({"type": "text", "text": question + ANSWER_SUFFIX})
        chat = processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = processor(images=images, text=chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        yl = torch.logsumexp(logits[yes_ids], dim=0)
        nl = torch.logsumexp(logits[no_ids], dim=0)
        return torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()

    # donor full images for the other_full_crop arm: a fixed derangement over the target list
    order = [u for u, _ in targets]
    donor = {u: order[(i + 1) % len(order)] for i, u in enumerate(order)}

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for uid, group in targets:
            item, rfull = items.get(uid), recs.get(uid)
            if item is None or rfull is None or not rfull.get("category"):
                continue
            img = item["image"].convert("RGB")
            key = (rfull["split"], str(rfull["question_id"]))
            W, H = isz.get(key, img.size)

            if group == "positive":
                boxes = bb.get(key)
                if not boxes:
                    continue
                x0, y0, x1, y1 = union_bbox(boxes)
                cx0, cy0, cx1, cy1 = padded_crop_box(x0, y0, x1, y1, W, H, PAD_FRAC)
            else:
                cw, ch = rng.choice(crop_size_pool)
                cw, ch = min(cw, W), min(ch, H)
                cx0 = rng.uniform(0, max(1.0, W - cw)); cy0 = rng.uniform(0, max(1.0, H - ch))
                cx1, cy1 = cx0 + cw, cy0 + ch
            if cx1 - cx0 < 2 or cy1 - cy0 < 2:
                continue
            crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
            blank = Image.new("RGB", crop.size, (0, 0, 0))

            d_uid = donor[uid]
            d_item = items.get(d_uid)
            other_full = (d_item["image"].convert("RGB") if d_item is not None and d_uid != uid
                          else None)

            q = item["question"]
            arms = {
                "full_alone":       p_yes([img], q, False),
                "crop_alone":       p_yes([crop], q, False),
                "crop_conn":        p_yes([crop], q, True),
                "crop_crop":        p_yes([crop, crop], q, True),
                "blank_crop":       p_yes([blank, crop], q, True),
                "full_crop_conn":   p_yes([img, crop], q, True),
                "full_crop_noconn": p_yes([img, crop], q, False),
                "crop_full_conn":   p_yes([crop, img], q, True),
            }
            if other_full is not None:
                arms["other_full_crop"] = p_yes([other_full, crop], q, True)

            rec = {"uid": uid, "group": group, "category": rfull["category"],
                   "split": rfull["split"], "question_id": rfull["question_id"],
                   "pixel_area_frac": rfull.get("pixel_area_frac"),
                   "crop_box": [cx0, cy0, cx1, cy1], "img_wh": [W, H],
                   "donor_uid": d_uid, "arms": arms}
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
