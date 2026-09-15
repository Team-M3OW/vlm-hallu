"""
The dissociation test (per advisor, 2026-09-03), operationalizing the ORIGINAL thesis of this
project ("VLMs localise well but don't answer/perceive/classify good enough") directly for the
first time: every prior experiment measured only the answer side (forced-choice P(yes) vs. ground
truth object area/position). This asks Qwen3-VL-2B-Instruct to LOCALIZE the same queried object,
for exactly the items where it confidently DENIED the object's existence in the forced-choice
setting (p_yes_real < 0.01, true label=yes -- a confident false "no").

If the model's predicted bounding box overlaps the true COCO box at a meaningfully high rate on
these confident-denial items, that is a real capability dissociation: the model locates evidence
its own answer module denies exists. This is NOT a scaling-curve restatement -- it is a claim about
two distinct capabilities (localization vs. verbalized judgment) being decoupled within the same
model on the same input.

Critical control (what would kill the finding): run the identical grounding prompt on POPE
NEGATIVES (where the object is genuinely absent). If the model still emits confident, plausible-
looking boxes for objects that aren't there, it has a generic "always guess a plausible location"
prior, not real localization -- and the confident-denial result would need to be reinterpreted
accordingly, not taken at face value.

Output format (determined empirically in phase4_ground_format_check.py): Qwen3-VL emits
```json [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]``` with coordinates normalized to a 0-1000
scale regardless of actual image size -- rescale by (img_w/1000, img_h/1000) before computing IoU
against the COCO ground-truth box (which is in raw pixel coordinates).
"""
import json
import re
import sys
import time
import random
import argparse
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase4_localize_results.jsonl"
PROMPT_TEMPLATE = "Locate the {obj} in the image and output its bounding box coordinates."


def extract_boxes(text):
    """Parse Qwen3-VL's ```json [{"bbox_2d": [...], ...}]``` output; tolerate truncated/malformed
    JSON (max_new_tokens can cut off mid-list) by regex-extracting bbox_2d arrays directly rather
    than requiring the whole blob to be valid JSON."""
    boxes = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text):
        boxes.append([float(m.group(i)) for i in range(1, 5)])
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs = [json.loads(l) for l in f]
    with open(f"{DATA}/pope_coco_area_joined.json") as f:
        joined = json.load(f)
    bbox_by_key = {}
    with open(f"{DATA}/coco_ann/instances_val2014.json") as f:
        coco = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    img_id_to_info = {im["id"]: im for im in coco["images"]}
    from collections import defaultdict
    anns_by_image_cat = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_cat[(a["image_id"], a["category_id"])].append(a)

    def coco_image_id_from_filename(fn):
        m = re.search(r"(\d+)\.jpg$", fn)
        return int(m.group(1))

    for j in joined:
        key = (j["split"], str(j["question_id"]))
        img_id = coco_image_id_from_filename(j["image"])
        cat_id = cat_name_to_id.get(j["category"])
        if cat_id is None:
            continue
        instances = anns_by_image_cat.get((img_id, cat_id), [])
        boxes = []
        for a in instances:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
        bbox_by_key[key] = boxes

    def extract_category(question):
        m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", question, re.I)
        return m.group(1).strip().lower() if m else None

    positives = [r for r in recs if r["label"] == "yes"]
    negatives = [r for r in recs if r["label"] == "no"]
    # negatives have category=None in the schema (build_items() never populates it) -- recover the
    # queried object name from the question text itself, same regex phase0_area_join.py uses.
    for r in negatives:
        r["category"] = extract_category(r["question"])
    negatives = [r for r in negatives if r["category"] is not None]
    confident_denials = [r for r in positives if r["p_yes_real"] < 0.01]
    confident_no_negs = random.Random(2).sample(
        [r for r in negatives if r["p_yes_real"] < 0.01], min(len(confident_denials), 1348))

    print(f"confident denials (true positive, model says no confidently): {len(confident_denials)}")
    print(f"confident-no negatives (control, object genuinely absent): {len(confident_no_negs)}")

    targets = [(r, "confident_denial") for r in confident_denials] + \
              [(r, "negative_control") for r in confident_no_negs]
    if args.limit:
        targets = targets[:args.limit]

    done_uids = set()
    if args.resume:
        try:
            with open(OUT_PATH) as f:
                for line in f:
                    done_uids.add(json.loads(line)["uid"])
            print(f"Resuming: {len(done_uids)} already done")
        except FileNotFoundError:
            pass

    target_uids = {r["uid"] for r, _ in targets if r["uid"] not in done_uids}
    print(f"Loading item images for {len(target_uids)} targets...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)}")

    print("Loading Qwen3-VL-2B...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r, group in targets:
            if r["uid"] in done_uids:
                continue
            item = items_by_uid.get(r["uid"])
            if item is None:
                continue
            img = item["image"].convert("RGB")
            img_w, img_h = img.size
            prompt = PROMPT_TEMPLATE.format(obj=r["category"])
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            gen = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            raw_boxes = extract_boxes(gen)
            pred_boxes = [[b[0] * img_w / 1000, b[1] * img_h / 1000, b[2] * img_w / 1000, b[3] * img_h / 1000]
                          for b in raw_boxes]

            key = (r["split"], str(r["question_id"]))
            gt_boxes = bbox_by_key.get(key, [])
            best_iou = 0.0
            for pb in pred_boxes:
                for gb in gt_boxes:
                    best_iou = max(best_iou, iou(pb, gb))

            rec = {"uid": r["uid"], "group": group, "category": r["category"],
                   "split": r["split"], "question_id": r["question_id"],
                   "p_yes_real": r["p_yes_real"], "n_pred_boxes": len(pred_boxes),
                   "n_gt_boxes": len(gt_boxes), "best_iou": best_iou,
                   "raw_gen": gen[:300]}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                print(f"[{n_done}] rate={n_done/elapsed:.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
