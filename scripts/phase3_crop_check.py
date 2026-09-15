"""
Crop-truncation check (per advisor review, 2026-09-03): LLaVA-1.5's CLIP preprocessing does
resize-shortest-edge-to-336 THEN CENTER-CROP to 336x336, discarding whatever falls outside the
crop window on the longer axis. Qwen's smart-resize keeps the whole frame (no crop). This predicts
exactly the observed LLaVA-vs-Qwen interaction divergence: peripheral objects can be PHYSICALLY
ABSENT from LLaVA's input entirely, while Qwen always sees the full frame. If true, this reframes
"two models differ mysteriously" into "a standard preprocessing default silently deletes
peripheral evidence" -- a mechanistic, actionable finding.

Reuses resize_crop_bbox from phase0_area_join.py. For each positive, computes:
  crop_survival_frac = (area of the object's bbox that survives the resize+crop, in the cropped
                         336x336 frame) / (area the bbox WOULD have after resize alone, no crop)
0 = bbox entirely outside the crop window (LLaVA cannot see it at all), 1 = fully inside (no loss).
"""
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase0_area_join import CLIP_SIZE, resize_crop_bbox
from phase3_centrality import (build_centrality_by_key, standardize, pearson_r,
                                logistic_regression, tertile_edges, tertile_of, auroc)

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
POPE_FILES = [
    f"{DATA}/pope/coco_pope_random.json",
    f"{DATA}/pope/coco_pope_popular.json",
    f"{DATA}/pope/coco_pope_adversarial.json",
]
COCO_ANN = f"{DATA}/coco_ann/instances_val2014.json"


def resized_area_no_crop(bbox, img_w, img_h):
    x, y, w, h = bbox
    scale = CLIP_SIZE / min(img_w, img_h)
    return (w * scale) * (h * scale)


def cropped_area(bbox, img_w, img_h):
    r = resize_crop_bbox(bbox, img_w, img_h)
    if r is None:
        return 0.0
    x0, y0, x1, y1 = r
    return (x1 - x0) * (y1 - y0)


def build_survival_by_key():
    with open(COCO_ANN) as f:
        coco = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    img_id_to_info = {im["id"]: im for im in coco["images"]}
    anns_by_image_cat = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_cat[(a["image_id"], a["category_id"])].append(a)

    def coco_image_id_from_filename(fn):
        m = re.search(r"(\d+)\.jpg$", fn)
        return int(m.group(1))

    survival_by_key = {}
    for pf in POPE_FILES:
        split = pf.split("coco_pope_")[1].replace(".json", "")
        with open(pf) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for item in lines:
            if item["label"] == "no":
                continue
            m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", item["text"], re.I)
            if not m:
                continue
            obj = m.group(1).strip().lower()
            if obj not in cat_name_to_id:
                continue
            cat_id = cat_name_to_id[obj]
            img_id = coco_image_id_from_filename(item["image"])
            info = img_id_to_info.get(img_id)
            if info is None:
                continue
            instances = anns_by_image_cat.get((img_id, cat_id), [])
            if not instances:
                continue
            img_w, img_h = info["width"], info["height"]
            total_resized = sum(resized_area_no_crop(a["bbox"], img_w, img_h) for a in instances)
            total_cropped = sum(cropped_area(a["bbox"], img_w, img_h) for a in instances)
            survival = total_cropped / total_resized if total_resized > 0 else 0.0
            survival_by_key[(split, str(item["question_id"]))] = survival
    return survival_by_key


def main():
    print("Building crop-survival lookup ...")
    survival_by_key = build_survival_by_key()
    print(f"Built survival for {len(survival_by_key)} keys\n")

    with open(f"{DATA}/phase1_results.jsonl") as f:
        recs = [json.loads(l) for l in f]
    positives = [r for r in recs if r["label"] == "yes" and r.get("patch_token_frac") is not None]

    survivals = [survival_by_key.get((r["split"], str(r["question_id"])), None) for r in positives]
    matched = [(r, s) for r, s in zip(positives, survivals) if s is not None]
    print(f"matched {len(matched)}/{len(positives)} positives to a survival value\n")

    zero_survival = sum(1 for r, s in matched if s == 0.0)
    print(f"--- Check 1: how many positives are FULLY cropped out (survival==0)? ---")
    print(f"  {zero_survival}/{len(matched)} ({zero_survival/len(matched):.4f}) -- LLaVA cannot "
          f"see these objects AT ALL, regardless of how the model 'should' answer")

    print(f"\n--- Check 1b: distribution of survival fraction ---")
    svals = sorted(s for _, s in matched)
    n = len(svals)
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        idx = min(n - 1, int(p / 100 * n))
        print(f"  {p}th pct: {svals[idx]:.4f}")

    print(f"\n--- Check 2: is survival predicted by centrality? (should be strongly negative: "
          f"more peripheral -> more likely cropped away) ---")
    centrality_by_key = build_centrality_by_key()
    triples = []
    for r, s in matched:
        c = centrality_by_key.get((r["split"], str(r["question_id"])))
        if c is not None:
            triples.append((s, c, r["patch_token_frac"], 1.0 if r["p_yes_real"] > 0.5 else 0.0))
    svs = [t[0] for t in triples]
    cents = [t[1] for t in triples]
    print(f"  r(survival, centrality) = {pearson_r(svs, cents):+.4f}")

    print(f"\n--- Check 2b: is survival concentrated in the small+peripheral cell specifically? ---")
    areas = [t[2] for t in triples]
    a_edges = tertile_edges(areas)
    c_edges = tertile_edges(cents)
    for ai, aname in enumerate(["small", "mid", "large"]):
        for ci, cname in enumerate(["central", "mid", "peripheral"]):
            cell = [t for t in triples if tertile_of(t[2], a_edges) == ai and tertile_of(t[1], c_edges) == ci]
            if len(cell) < 10:
                continue
            mean_surv = sum(t[0] for t in cell) / len(cell)
            frac_zero = sum(1 for t in cell if t[0] == 0.0) / len(cell)
            print(f"  {aname:>10s} x {cname:>10s}: n={len(cell):4d} mean_survival={mean_surv:.3f} "
                  f"frac_fully_cropped={frac_zero:.3f}")

    print(f"\n--- Check 3: does the interaction coefficient collapse when restricted to "
          f"positives whose bbox survives the crop ~fully (survival >= 0.95)? ---")
    full_survival = [t for t in triples if t[0] >= 0.95]
    print(f"  n={len(full_survival)}/{len(triples)} retained")
    areas_f = [t[2] for t in full_survival]
    cents_f = [t[1] for t in full_survival]
    correct_f = [t[3] for t in full_survival]
    areas_fz, _, _ = standardize(areas_f)
    cents_fz, _, _ = standardize(cents_f)
    inter_f = [a * c for a, c in zip(areas_fz, cents_fz)]
    w, b = logistic_regression(list(zip(areas_fz, cents_fz, inter_f)), correct_f)
    print(f"  RESTRICTED (survival>=0.95): coef(area_z)={w[0]:+.4f}  coef(cent_z)={w[1]:+.4f}  "
          f"coef(area_z*cent_z)={w[2]:+.4f}")
    print(f"  (compare to FULL-sample interaction coef = +0.3699 -- if this collapses toward 0 or "
          f"toward Qwen's +0.061, the crop is the mechanism)")

    # Also recompute the joint table restricted to full-survival items for a direct visual check
    print(f"\n--- Restricted joint table (survival>=0.95 only) ---")
    a_edges_f = tertile_edges(areas_f)
    c_edges_f = tertile_edges(cents_f)
    neg_scores = [r["p_yes_real"] for r in recs if r["label"] == "no"]
    for ai, aname in enumerate(["small", "mid", "large"]):
        row = []
        for ci, cname in enumerate(["central", "mid", "peripheral"]):
            cell = [t for t in full_survival if tertile_of(t[2], a_edges_f) == ai
                    and tertile_of(t[1], c_edges_f) == ci]
            if len(cell) < 10:
                row.append(f"n={len(cell)}(few)")
                continue
            acc = sum(t[3] for t in cell) / len(cell)
            row.append(f"n={len(cell)} acc={acc:.2f}")
        print(f"  {aname:>6s}: " + " | ".join(row))


if __name__ == "__main__":
    main()
