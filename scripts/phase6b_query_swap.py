"""
Phase 6b: query-swap control (per advisor, 2026-09-03), discriminating between two readings of the
Phase 6 between-group result (confident_denial pooled yes/no enrichment 0.848 vs
confident_correct_small 3.173, 3.74x, bootstrap CI [3.06x, 4.57x]):

1. TAUTOLOGY reading: attention only concentrates on a region when the model is ABOUT TO SAY YES
   about it -- "yes" requires pointing at evidence, "no" doesn't. Under this reading the 3.74x gap
   is just a restatement of the answer token in attention-space, true for ANY object, present or
   not, and reduces to nothing more than the existing area-scaling finding once you condition on
   answer polarity.
2. ROUTING-FAILURE reading: attention is query-conditioned (the model looks where the question
   points), and confident_denial items specifically fail to route attention to the true object
   location REGARDLESS of what is asked, while confident_correct_small items DO route attention to
   the true object location when asked about it.

Discriminating test: hold the SAME image and the SAME object-X token set fixed, but swap the
question to ask about a DIFFERENT, genuinely absent category Y (verified absent via full COCO
instance annotations for that image, not just the POPE-joined subset). Measure enrichment on X's
token set under this swapped query.
  - If X's-region enrichment under the Y-query looks the same as under the X-query, for a GIVEN
    group -> attention on X's region is not query-conditioned for that group; it's fixed
    image-intrinsic saliency (or lack thereof).
  - If confident_correct_small shows a big drop (attention was on X only because X was asked about)
    while confident_denial shows no drop (attention on X was already low and query-independent) ->
    that pattern discriminates routing-failure from the tautology explanation.

Reuses phase6_attention_entanglement.py's Ctx, grid_from_image, object_token_indices,
yes_no_enrichment, load_bbox_lookup -- only the question and the absent-category selection differ.
"""
import json
import re
import sys
import random
import time
from collections import defaultdict

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase6_attention_entanglement import (
    Ctx, grid_from_image, object_token_indices, yes_no_enrichment, load_bbox_lookup, p_yes_final,
)
from phase1_eval import build_items

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase6b_query_swap_results.jsonl"

COMMON_CATEGORIES = [
    "person", "car", "chair", "bottle", "dog", "cat", "truck", "bicycle", "umbrella", "backpack",
    "handbag", "bowl", "cup", "tv", "laptop", "book", "clock", "sports ball", "snowboard", "bench",
    "horse", "boat", "train", "airplane", "motorcycle", "couch", "bed", "dining table", "oven",
    "refrigerator", "sink", "toilet", "kite", "skateboard", "surfboard", "tennis racket",
]


def build_image_category_index(coco):
    """category_ids present in EACH image, across the FULL COCO annotation set (not just the
    POPE-joined subset) -- needed so the swap category is verifiably absent from the actual image,
    not merely absent from the small set of categories POPE happened to ask about for it."""
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    present = defaultdict(set)
    for a in coco["annotations"]:
        present[a["image_id"]].add(a["category_id"])
    return present, cat_name_to_id


def pick_absent_category(rng, present_cat_ids, cat_name_to_id, exclude_name):
    candidates = [c for c in COMMON_CATEGORIES
                  if c != exclude_name and cat_name_to_id.get(c) not in present_cat_ids]
    if not candidates:
        return None
    return rng.choice(candidates)


def swap_question(category):
    article = "an" if category[0].lower() in "aeiou" else "a"
    return f"Is there {article} {category} in the image?"


def main():
    with open(f"{DATA}/coco_ann/instances_val2014.json") as f:
        coco = json.load(f)
    present_by_image, cat_name_to_id = build_image_category_index(coco)

    def coco_image_id_from_filename(fn):
        return int(re.search(r"(\d+)\.jpg$", fn).group(1))

    with open(f"{DATA}/pope_coco_area_joined.json") as f:
        joined = json.load(f)
    image_id_by_key = {}
    for j in joined:
        key = (j["split"], str(j["question_id"]))
        image_id_by_key[key] = coco_image_id_from_filename(j["image"])

    bbox_by_key, imgsize_by_key = load_bbox_lookup()

    with open(f"{DATA}/phase6_attention_results.jsonl") as f:
        phase6_recs = [json.loads(l) for l in f]
    print(f"phase6 items available: {len(phase6_recs)}")

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs_by_uid = {(d := json.loads(l))["uid"]: d for l in f}

    rng = random.Random(3)
    targets = []
    for p6 in phase6_recs:
        uid = p6["uid"]
        r = recs_by_uid.get(uid)
        if r is None:
            continue
        key = (r["split"], str(r["question_id"]))
        image_id = image_id_by_key.get(key)
        bboxes = bbox_by_key.get(key)
        imgsize = imgsize_by_key.get(key)
        if image_id is None or not bboxes or imgsize is None:
            continue
        absent_cat = pick_absent_category(rng, present_by_image.get(image_id, set()),
                                           cat_name_to_id, r["category"])
        if absent_cat is None:
            continue
        targets.append((r, p6["group"], bboxes, imgsize, absent_cat))

    print(f"targets with a valid swap category: {len(targets)}")

    target_uids = {r["uid"] for r, _, _, _, _ in targets}
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

    print("Loading Qwen3-VL-2B (fp16, eager attention)...")
    ctx = Ctx()

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r, group, bboxes, imgsize, absent_cat in targets:
            uid = r["uid"]
            if uid in done_uids:
                continue
            item = items_by_uid.get(uid)
            if item is None:
                continue
            img = item["image"].convert("RGB")
            img_w, img_h = imgsize
            grid_h, grid_w, resized_h, resized_w = grid_from_image(ctx, img)
            obj_indices = object_token_indices(bboxes, img_w, img_h, resized_w, resized_h, grid_h, grid_w)
            if not obj_indices:
                continue

            swap_q = swap_question(absent_cat)
            swap_py = p_yes_final(ctx, img, swap_q)
            swap_enrich = yes_no_enrichment(ctx, img, swap_q, obj_indices)

            rec = {"uid": uid, "group": group, "category": r["category"], "swap_category": absent_cat,
                   "n_obj_tokens": len(obj_indices), "n_image_tokens": grid_h * grid_w,
                   "swap_p_yes": swap_py, "swap_attn": swap_enrich}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
