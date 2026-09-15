"""
Phase 6c: present-but-not-asked control (per advisor, 2026-09-03) -- fixes an underspecified design
in Phase 6b. Phase 6b's "modulation index" compared query=X (answer=YES for confident_correct_small,
NO for confident_denial) against query=absent-Y (answer=NO for both groups). For
confident_correct_small this confounds query-TARGET (X vs Y) with answer-POLARITY (yes vs no) --
the observed 5.51x drop is equally consistent with pure "attend when about to say yes, don't when
about to say no" (the tautology the whole Phase 6 line is trying to rule out) as with real
query-target-conditioned attention. confident_denial's 1.21x cannot have this confound (both its
conditions are answered NO), so the two modulation indices were not actually comparable evidence.

Third condition: query = Z, a DIFFERENT object verified PRESENT in the same image (via the same
full-COCO `present_by_image` index phase6b built, membership test inverted and restricted to
categories with an actual annotated instance in that image), excluding the item's own category.
Answer to "is Z present" should be YES for both groups (Z genuinely is there) -- this decouples
target-conditioning from polarity-conditioning:
  - confident_correct_small: if enrichment on X's tokens under query=Z stays ~3 (same ballpark as
    query=X), the original 3.17 was polarity, not target -- the whole between-group finding
    collapses into the tautology. If it drops toward ~0.6 (like the query=Y condition), attention
    really did move because Z, not X, was asked about -- polarity is not doing the work, and
    confident_denial's flatness across ALL THREE conditions (X/no-present, Y/no-absent, Z/yes-other)
    would be the strong version of the routing-failure claim.
  - confident_denial: run the same query=Z condition and check whether X's-token enrichment stays
    in the same ~0.7-0.85 band it occupied under query=X and query=absent-Y, now under a query that
    is answered YES about something else -- i.e. X's own attention is invariant to both the target
    AND the polarity of what's asked, not just to swapping in another "no".

Reuses phase6_attention_entanglement.py's Ctx/grid_from_image/object_token_indices/
yes_no_enrichment/load_bbox_lookup/p_yes_final and phase6b_query_swap.py's image-category index and
question-building code -- only the category-selection predicate differs (present, not absent).
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
from phase6b_query_swap import build_image_category_index, swap_question, COMMON_CATEGORIES
from phase1_eval import build_items

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase6c_present_other_results.jsonl"


def pick_present_other_category(rng, present_cat_ids, cat_name_to_id, exclude_name):
    candidates = [c for c in COMMON_CATEGORIES
                  if c != exclude_name and cat_name_to_id.get(c) in present_cat_ids]
    if not candidates:
        return None
    return rng.choice(candidates)


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

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs_by_uid = {(d := json.loads(l))["uid"]: d for l in f}

    rng = random.Random(11)
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
        present_ids = present_by_image.get(image_id, set())
        # need at least the item's own category present (sanity) plus one OTHER present category
        other_cat = pick_present_other_category(rng, present_ids, cat_name_to_id, r["category"])
        if other_cat is None:
            continue
        targets.append((r, p6["group"], bboxes, imgsize, other_cat))

    print(f"targets with a valid present-other category: {len(targets)}")

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
        for r, group, bboxes, imgsize, other_cat in targets:
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

            other_q = swap_question(other_cat)
            other_py = p_yes_final(ctx, img, other_q)
            other_enrich = yes_no_enrichment(ctx, img, other_q, obj_indices)

            rec = {"uid": uid, "group": group, "category": r["category"], "other_category": other_cat,
                   "n_obj_tokens": len(obj_indices), "n_image_tokens": grid_h * grid_w,
                   "other_p_yes": other_py, "other_attn": other_enrich}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
