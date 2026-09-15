"""
Centrality confound check (per advisor review, 2026-09-03): COCO bbox carries x,y position, which
phase0_area_join.py discarded (only area survived). If object CENTRALITY (how close to the image
center the evidence sits) explains the accuracy-vs-area effect better than raw area does, the
phenomenon is actually about salience/framing, not evidence size -- a genuine reversal of the
paper's working thesis, not just a decoration of it. If area survives controlling for centrality,
the last cheap confound is closed.

Re-derives centrality independently from instances_val2014.json (bbox x,y), joined the same way
phase0_area_join.py joins POPE positives to COCO instances, then merges onto phase1_results.jsonl
(real p_yes, correct) by (split, question_id) for both LLaVA and Qwen.

Centrality definition: area-weighted centroid across all instances of the queried category in the
image (mirrors phase0_area_join.py's area = SUM of instance areas, i.e. "union of evidence"),
normalized Euclidean distance from image center:
    centrality = sqrt(((cx - img_w/2)/img_w)^2 + ((cy - img_h/2)/img_h)^2)
0 = dead center, ~0.5-0.7 = near a corner. Higher = LESS central (more peripheral).
"""
import json
import re
import math
from collections import defaultdict

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
POPE_FILES = [
    f"{DATA}/pope/coco_pope_random.json",
    f"{DATA}/pope/coco_pope_popular.json",
    f"{DATA}/pope/coco_pope_adversarial.json",
]
COCO_ANN = f"{DATA}/coco_ann/instances_val2014.json"


def standardize(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var) if var > 0 else 1.0
    return [(x - m) / sd for x in xs], m, sd


def pearson_r(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def logistic_regression(X, y, iters=500, lr=0.1):
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        grad_w = [0.0] * d
        grad_b = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[k] * xi[k] for k in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            err = p - yi
            for k in range(d):
                grad_w[k] += err * xi[k]
            grad_b += err
        for k in range(d):
            w[k] -= lr * grad_w[k] / n
        b -= lr * grad_b / n
    return w, b


def build_centrality_by_key():
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

    centrality_by_key = {}
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
            total_area = sum(a["area"] for a in instances)
            if total_area <= 0:
                continue
            cx = sum(a["area"] * (a["bbox"][0] + a["bbox"][2] / 2.0) for a in instances) / total_area
            cy = sum(a["area"] * (a["bbox"][1] + a["bbox"][3] / 2.0) for a in instances) / total_area
            centrality = math.sqrt(((cx - img_w / 2.0) / img_w) ** 2 + ((cy - img_h / 2.0) / img_h) ** 2)
            centrality_by_key[(split, str(item["question_id"]))] = centrality
    return centrality_by_key


def auroc(pos_scores, neg_scores):
    labeled = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    labeled.sort(key=lambda x: x[0])
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    n = len(labeled)
    while i < n:
        j = i
        while j < n and labeled[j][0] == labeled[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if labeled[k][1] == 1:
                rank_sum += avg_rank
        i = j
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def tertile_edges(vals):
    s = sorted(vals)
    n = len(s)
    return [s[0], s[n // 3], s[2 * n // 3], s[-1] + 1e-9]


def tertile_of(v, edges):
    for i in range(3):
        if edges[i] <= v < edges[i + 1] or (i == 2 and v == edges[-1] - 1e-9):
            return i
    return 2


def joint_analysis(results_path, area_key, label, centrality_by_key):
    print(f"\n=== JOINT area x centrality interaction: {label} ===")
    recs = []
    with open(results_path) as f:
        for line in f:
            recs.append(json.loads(line))
    positives = [r for r in recs if r["label"] == "yes" and r.get(area_key) is not None]
    negatives = [r for r in recs if r["label"] == "no"]
    neg_scores = [r["p_yes_real"] for r in negatives]

    merged = []
    for r in positives:
        key = (r["split"], str(r["question_id"]))
        c = centrality_by_key.get(key)
        if c is None:
            continue
        merged.append({"area": r[area_key], "cent": c, "p_yes": r["p_yes_real"],
                        "correct": 1.0 if r["p_yes_real"] > 0.5 else 0.0})

    areas = [m["area"] for m in merged]
    cents = [m["cent"] for m in merged]
    a_edges = tertile_edges(areas)
    c_edges = tertile_edges(cents)
    a_names = ["small", "mid", "large"]
    c_names = ["central", "mid", "peripheral"]

    print(f"  area tertile edges: {[f'{x:.4f}' for x in a_edges]}")
    print(f"  centrality tertile edges: {[f'{x:.4f}' for x in c_edges]}")
    print(f"\n  {'':>10s}", end="")
    for cn in c_names:
        print(f"{cn:>28s}", end="")
    print()
    for ai in range(3):
        print(f"  {a_names[ai]:>10s}", end="")
        for ci in range(3):
            cell = [m for m in merged if tertile_of(m["area"], a_edges) == ai
                    and tertile_of(m["cent"], c_edges) == ci]
            if len(cell) < 10:
                print(f"{'n=' + str(len(cell)) + ' (too few)':>28s}", end="")
                continue
            acc = sum(m["correct"] for m in cell) / len(cell)
            a = auroc([m["p_yes"] for m in cell], neg_scores)
            print(f"{'n=' + str(len(cell)) + f' acc={acc:.2f} AUROC={a:.2f}':>28s}", end="")
        print()

    # Interaction regression: correct ~ area_z + cent_z + area_z*cent_z (+ blank_z if available)
    areas_z, _, _ = standardize(areas)
    cents_z, _, _ = standardize(cents)
    inter = [a * c for a, c in zip(areas_z, cents_z)]
    correct = [m["correct"] for m in merged]
    w, b = logistic_regression(list(zip(areas_z, cents_z, inter)), correct)
    print(f"\n  Interaction regression: correct ~ area_z + cent_z + (area_z*cent_z)")
    print(f"  coef(area_z)={w[0]:+.4f}  coef(cent_z)={w[1]:+.4f}  coef(area_z*cent_z)={w[2]:+.4f}")
    print(f"  (a negative interaction coef would mean area matters LESS when already central --")
    print(f"  i.e. centrality partially COMPENSATES for small area; near-zero means the two")
    print(f"  factors act independently/additively)")


def analyze(results_path, area_key, label, centrality_by_key):
    print(f"\n=== {label} ({results_path}, area_key={area_key}) ===")
    recs = []
    with open(results_path) as f:
        for line in f:
            recs.append(json.loads(line))
    positives = [r for r in recs if r["label"] == "yes" and r.get(area_key) is not None]

    merged = []
    for r in positives:
        key = (r["split"], str(r["question_id"]))
        c = centrality_by_key.get(key)
        if c is None:
            continue
        merged.append((r[area_key], c, r["p_yes_blank"], 1.0 if r["p_yes_real"] > 0.5 else 0.0))
    print(f"  matched {len(merged)}/{len(positives)} positives to a centrality value")

    areas = [m[0] for m in merged]
    centralities = [m[1] for m in merged]
    blanks = [m[2] for m in merged]
    correct = [m[3] for m in merged]

    print(f"  raw pearson r(centrality, correct) = {pearson_r(centralities, correct):+.4f}")
    print(f"  raw pearson r(area, correct)       = {pearson_r(areas, correct):+.4f}")
    print(f"  raw pearson r(area, centrality)    = {pearson_r(areas, centralities):+.4f}  "
          f"(are larger objects also more central? if strongly negative, area and centrality are "
          f"themselves confounded with each other, e.g. bigger objects tend to be more centered)")

    areas_z, _, _ = standardize(areas)
    cent_z, _, _ = standardize(centralities)
    blanks_z, _, _ = standardize(blanks)

    # Model A: area only (replicates the original primary test)
    w_a, b_a = logistic_regression(list(zip(areas_z, blanks_z)), correct)
    # Model B: centrality only
    w_b, b_b = logistic_regression(list(zip(cent_z, blanks_z)), correct)
    # Model C: both together -- the decisive comparison
    w_c, b_c = logistic_regression(list(zip(areas_z, cent_z, blanks_z)), correct)

    print(f"\n  Model A (area only):         coef(area_z)={w_a[0]:+.4f}  coef(blank_z)={w_a[1]:+.4f}")
    print(f"  Model B (centrality only):   coef(cent_z)={w_b[0]:+.4f}  coef(blank_z)={w_b[1]:+.4f}")
    print(f"  Model C (both together):     coef(area_z)={w_c[0]:+.4f}  coef(cent_z)={w_c[1]:+.4f}  "
          f"coef(blank_z)={w_c[2]:+.4f}")
    print(f"\n  Interpretation: compare coef(area_z) in A ({w_a[0]:+.4f}) vs in C ({w_c[0]:+.4f}).")
    print(f"  If area's coefficient collapses toward 0 in C while centrality's stays strong, the")
    print(f"  effect is really about salience/framing, not evidence size -- a reversal of the")
    print(f"  paper's thesis. If area survives in C with similar sign/magnitude, area is the real")
    print(f"  effect and centrality is not explaining it away.")
    return w_a[0], w_c[0], w_c[1]


def main():
    print("Building centrality lookup from instances_val2014.json ...")
    centrality_by_key = build_centrality_by_key()
    print(f"Built centrality for {len(centrality_by_key)} (split, question_id) keys")

    analyze(f"{DATA}/phase1_results.jsonl", "patch_token_frac", "LLaVA-1.5-7B (patch_token_frac)",
            centrality_by_key)
    analyze(f"{DATA}/phase1_results.jsonl", "pixel_area_frac", "LLaVA-1.5-7B (pixel_area_frac)",
            centrality_by_key)
    analyze(f"{DATA}/phase1_results_qwen_dedup.jsonl", "pixel_area_frac", "Qwen3-VL-2B (pixel_area_frac)",
            centrality_by_key)

    joint_analysis(f"{DATA}/phase1_results.jsonl", "pixel_area_frac", "LLaVA-1.5-7B", centrality_by_key)
    joint_analysis(f"{DATA}/phase1_results_qwen_dedup.jsonl", "pixel_area_frac", "Qwen3-VL-2B", centrality_by_key)


if __name__ == "__main__":
    main()
