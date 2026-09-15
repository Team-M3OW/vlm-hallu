"""
Phase 0: join POPE (COCO split) positive questions to COCO instance annotations,
compute pixel-area-fraction and patch-token-fraction per positive, histogram them,
and report N-per-candidate-bin so we can lock bin edges before touching the model.

No GPU, no model load, no images downloaded (uses stored width/height in COCO json).
"""
import json
import re
from collections import defaultdict

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
POPE_FILES = [
    f"{DATA}/pope/coco_pope_random.json",
    f"{DATA}/pope/coco_pope_popular.json",
    f"{DATA}/pope/coco_pope_adversarial.json",
]
COCO_ANN = f"{DATA}/coco_ann/instances_val2014.json"

# LLaVA-1.5 CLIP preprocessing: resize shorter side + center-crop to 336x336, patch=14 -> 24x24=576 tokens.
# We approximate the token-overlap fraction analytically from bbox in original image coords by
# replicating this exact resize+crop geometry (verified against actual CLIPImageProcessor config
# in phase0b before trusting these numbers for the paper).
CLIP_SIZE = 336
PATCH = 14
GRID = CLIP_SIZE // PATCH  # 24


def resize_crop_bbox(bbox, img_w, img_h):
    """Map a COCO bbox [x,y,w,h] through CLIP's resize-shorter-side-then-center-crop to 336x336,
    return bbox in the cropped 336x336 frame (clipped to bounds), or None if fully cropped out."""
    x, y, w, h = bbox
    scale = CLIP_SIZE / min(img_w, img_h)
    new_w, new_h = img_w * scale, img_h * scale
    # center crop offsets in resized frame
    off_x = (new_w - CLIP_SIZE) / 2.0
    off_y = (new_h - CLIP_SIZE) / 2.0
    rx0, ry0 = x * scale - off_x, y * scale - off_y
    rx1, ry1 = (x + w) * scale - off_x, (y + h) * scale - off_y
    cx0, cy0 = max(0, rx0), max(0, ry0)
    cx1, cy1 = min(CLIP_SIZE, rx1), min(CLIP_SIZE, ry1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return cx0, cy0, cx1, cy1


def patch_token_frac(bbox, img_w, img_h):
    r = resize_crop_bbox(bbox, img_w, img_h)
    if r is None:
        return 0.0
    x0, y0, x1, y1 = r
    covered = 0
    for gy in range(GRID):
        py0, py1 = gy * PATCH, (gy + 1) * PATCH
        if py1 <= y0 or py0 >= y1:
            continue
        for gx in range(GRID):
            px0, px1 = gx * PATCH, (gx + 1) * PATCH
            if px1 <= x0 or px0 >= x1:
                continue
            covered += 1
    return covered / (GRID * GRID)


def main():
    print("Loading COCO instances_val2014.json ...")
    with open(COCO_ANN) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    img_id_to_info = {im["id"]: im for im in coco["images"]}

    anns_by_image_cat = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_cat[(a["image_id"], a["category_id"])].append(a)

    def coco_image_id_from_filename(fn):
        m = re.search(r"(\d+)\.jpg$", fn)
        return int(m.group(1))

    all_pos = []
    all_neg_count = 0
    unmatched_categories = set()

    for pf in POPE_FILES:
        split = pf.split("coco_pope_")[1].replace(".json", "")
        with open(pf) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for item in lines:
            if item["label"] == "no":
                all_neg_count += 1
                continue
            m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", item["text"], re.I)
            if not m:
                continue
            obj = m.group(1).strip().lower()
            if obj not in cat_name_to_id:
                unmatched_categories.add(obj)
                continue
            cat_id = cat_name_to_id[obj]
            img_id = coco_image_id_from_filename(item["image"])
            info = img_id_to_info.get(img_id)
            if info is None:
                continue
            instances = anns_by_image_cat.get((img_id, cat_id), [])
            if not instances:
                unmatched_categories.add(f"{obj} (no instance found in {item['image']})")
                continue
            img_w, img_h = info["width"], info["height"]
            pixel_area_frac = sum(a["area"] for a in instances) / (img_w * img_h)
            # union of per-instance patch coverage (approx via max envelope of individual boxes'
            # covered-patch sets, computed properly as a set union below)
            covered_patches = set()
            for a in instances:
                r = resize_crop_bbox(a["bbox"], img_w, img_h)
                if r is None:
                    continue
                x0, y0, x1, y1 = r
                for gy in range(GRID):
                    py0, py1 = gy * PATCH, (gy + 1) * PATCH
                    if py1 <= y0 or py0 >= y1:
                        continue
                    for gx in range(GRID):
                        px0, px1 = gx * PATCH, (gx + 1) * PATCH
                        if px1 <= x0 or px0 >= x1:
                            continue
                        covered_patches.add((gx, gy))
            patch_frac = len(covered_patches) / (GRID * GRID)
            all_pos.append({
                "split": split,
                "question_id": item["question_id"],
                "image": item["image"],
                "category": obj,
                "n_instances": len(instances),
                "pixel_area_frac": pixel_area_frac,
                "patch_token_frac": patch_frac,
            })

    print(f"Total positives matched: {len(all_pos)}")
    print(f"Total negatives (label=no, not area-joined): {all_neg_count}")
    print(f"Unmatched/missing categories (sample): {list(unmatched_categories)[:20]}")
    print(f"  ({len(unmatched_categories)} unique unmatched entries)")

    out_path = f"{DATA}/pope_coco_area_joined.json"
    with open(out_path, "w") as f:
        json.dump(all_pos, f)
    print(f"Wrote {out_path}")

    # histogram
    import statistics
    pf_vals = sorted(p["patch_token_frac"] for p in all_pos)
    px_vals = sorted(p["pixel_area_frac"] for p in all_pos)

    def hist(vals, edges):
        counts = [0] * (len(edges) - 1)
        for v in vals:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                    counts[i] += 1
                    break
        return counts

    edges = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
    print("\npatch_token_frac histogram, edges=", edges)
    print(hist(pf_vals, edges))
    print("pixel_area_frac histogram, same edges")
    print(hist(px_vals, edges))
    print(f"\npatch_token_frac: min={pf_vals[0]:.4f} p10={pf_vals[len(pf_vals)//10]:.4f} "
          f"median={statistics.median(pf_vals):.4f} p90={pf_vals[int(len(pf_vals)*0.9)]:.4f} max={pf_vals[-1]:.4f}")
    print(f"pixel_area_frac: min={px_vals[0]:.4f} p10={px_vals[len(px_vals)//10]:.4f} "
          f"median={statistics.median(px_vals):.4f} p90={px_vals[int(len(px_vals)*0.9)]:.4f} max={px_vals[-1]:.4f}")


if __name__ == "__main__":
    main()
