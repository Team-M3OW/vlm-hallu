"""
Analysis of phase1_results.jsonl:
1. POPE baseline accuracy check (sanity gate: should land near published ~87/86/84% for
   random/popular/adversarial on LLaVA-1.5-7B; if it's at ~60%, the harness is broken and no
   area effect means anything).
2. Per-bin AUROC of P(yes) vs. true label (positives in that patch_token_frac bin, pooled against
   ALL negatives) -- the headline, threshold-free plot.
3. Secondary: raw accuracy per bin, mean P(yes) per bin for positives.
4. Primary statistical test (locked pre-registered choice): does patch_token_frac predict
   p_yes_real after controlling for p_yes_blank (logistic regression on positives, DV = correct,
   covariate = p_yes_blank, predictor = patch_token_frac; also report the raw correlation of
   patch_token_frac with p_yes_real and with (p_yes_real - p_yes_blank)).
"""
import json
import math

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BIN_EDGES = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def auroc(pos_scores, neg_scores):
    # Mann-Whitney U based AUROC, O(n log n)
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


def bin_of(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
            return i
    return None


def logistic_regression(X, y, iters=500, lr=0.1):
    # tiny hand-rolled 2-feature (+intercept) logistic regression via gradient descent, to avoid
    # a sklearn dependency for a quick pilot-stage check.
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


def main(path=f"{DATA}/phase1_results.jsonl", label="LLaVA-1.5-7B", area_key="patch_token_frac"):
    recs = load(path)
    print(f"=== {label}: {len(recs)} records loaded from {path} (area_key={area_key}) ===\n")

    # --- 1. POPE baseline accuracy per split (positives+negatives, real image only) ---
    print("--- Baseline accuracy vs published POPE (sanity gate) ---")
    for split in ["random", "popular", "adversarial"]:
        sub = [r for r in recs if r["split"] == split]
        if not sub:
            continue
        correct = 0
        for r in sub:
            pred_yes = r["p_yes_real"] > 0.5
            true_yes = r["label"] == "yes"
            correct += int(pred_yes == true_yes)
        acc = correct / len(sub)
        n_pos = sum(1 for r in sub if r["label"] == "yes")
        n_neg = len(sub) - n_pos
        print(f"  {split}: n={len(sub)} (pos={n_pos},neg={n_neg}) accuracy={acc:.3f} "
              f"(published LLaVA-1.5-7B ballpark: ~0.87/0.86/0.84 for random/popular/adversarial)")
    overall_correct = sum(1 for r in recs if (r["p_yes_real"] > 0.5) == (r["label"] == "yes"))
    print(f"  overall: n={len(recs)} accuracy={overall_correct/len(recs):.3f}\n")

    positives = [r for r in recs if r["label"] == "yes" and r[area_key] is not None]
    negatives = [r for r in recs if r["label"] == "no"]
    print(f"positives with area={len(positives)}, negatives={len(negatives)}\n")

    # --- 2. Per-bin AUROC (headline) ---
    print("--- Per-bin AUROC of P(yes), positives-in-bin vs ALL negatives (headline plot) ---")
    neg_scores_all = [r["p_yes_real"] for r in negatives]
    bin_labels = [f"[{BIN_EDGES[i]},{BIN_EDGES[i+1]})" for i in range(len(BIN_EDGES) - 1)]
    for i, lab in enumerate(bin_labels):
        pos_in_bin = [r for r in positives if bin_of(r[area_key], BIN_EDGES) == i]
        if len(pos_in_bin) < 10:
            print(f"  bin {lab}: n={len(pos_in_bin)} (too few, skipping)")
            continue
        pos_scores = [r["p_yes_real"] for r in pos_in_bin]
        a = auroc(pos_scores, neg_scores_all)
        acc = sum(1 for r in pos_in_bin if r["p_yes_real"] > 0.5) / len(pos_in_bin)
        mean_py = sum(pos_scores) / len(pos_scores)
        mean_py_blank = sum(r["p_yes_blank"] for r in pos_in_bin) / len(pos_in_bin)
        print(f"  bin {lab}: n={len(pos_in_bin)} AUROC={a:.3f} recall(acc)={acc:.3f} "
              f"mean_p_yes_real={mean_py:.3f} mean_p_yes_blank={mean_py_blank:.3f}")
    print()

    # --- 3. Primary locked statistical test: does area predict p_yes_real controlling for blank ---
    print("--- Primary test: patch_token_frac -> correct, controlling for p_yes_blank ---")
    areas = [r[area_key] for r in positives]
    correct = [1.0 if r["p_yes_real"] > 0.5 else 0.0 for r in positives]
    blanks = [r["p_yes_blank"] for r in positives]
    areas_z, _, _ = standardize(areas)
    blanks_z, _, _ = standardize(blanks)
    X = list(zip(areas_z, blanks_z))
    w, b = logistic_regression(X, correct)
    print(f"  logistic regression: correct ~ patch_token_frac_z + p_yes_blank_z")
    print(f"  coef(patch_token_frac_z) = {w[0]:+.4f}   coef(p_yes_blank_z) = {w[1]:+.4f}   intercept={b:+.4f}")
    print(f"  (positive coef on area = LARGER evidence area -> MORE likely correct, i.e. small "
          f"evidence hurts accuracy, after controlling for blank-image prior)")

    r_area_correct = pearson_r(areas, correct)
    r_area_pyes = pearson_r(areas, [r["p_yes_real"] for r in positives])
    delta = [r["p_yes_real"] - r["p_yes_blank"] for r in positives]
    r_area_delta = pearson_r(areas, delta)
    print(f"  raw pearson r(area, correct)      = {r_area_correct:+.4f}")
    print(f"  raw pearson r(area, p_yes_real)    = {r_area_pyes:+.4f}")
    print(f"  raw pearson r(area, p_yes_real-blank) = {r_area_delta:+.4f}  (isolates area's effect "
          f"net of the blank-image/prior baseline)")

    mean_blank = sum(blanks) / len(blanks)
    print(f"\n  mean p_yes_blank across positives = {mean_blank:.3f} "
          f"(near 0 = strong no-bias baseline, near 1 = strong yes-bias baseline)")

    # --- 4. Category-confound robustness check (added 2026-09-02, per advisor review) ---
    # patch_token_frac is confounded with object category (tiny COCO instances skew toward
    # person/car/cup/bottle; large ones skew toward bed/train/pizza/dining-table; POPE questions
    # are generated per-category). "Small evidence hurts accuracy" and "hard/rare category hurts
    # accuracy" predict the same marginal curve, and p_yes_blank only partially absorbs this. The
    # discriminating test is within-category: does area still predict correctness once category is
    # held fixed? If the effect washes out within category, the phenomenon is category difficulty,
    # not an evidence-size effect, and Phase 2 (steering) would be chasing an artifact.
    print("\n--- Category-confound check: per-category r(patch_token_frac, correct), n>=30 ---")
    with open(f"{DATA}/pope_coco_area_joined.json") as f:
        joined = json.load(f)
    n_instances_by_key = {(j["split"], str(j["question_id"])): j["n_instances"] for j in joined}

    by_cat = {}
    for r in positives:
        by_cat.setdefault(r["category"], []).append(r)
    sign_pos = sign_neg = sign_flat = sign_nan = 0
    cat_rows = []
    for cat, rows in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < 30:
            continue
        areas_c = [r[area_key] for r in rows]
        correct_c = [1.0 if r["p_yes_real"] > 0.5 else 0.0 for r in rows]
        if len(set(areas_c)) < 2:
            continue
        r_c = pearson_r(areas_c, correct_c)
        cat_rows.append((cat, len(rows), r_c))
        if r_c != r_c:  # nan (zero-variance correct, e.g. category at 100% accuracy)
            sign_nan += 1
        elif r_c > 0.05:
            sign_pos += 1
        elif r_c < -0.05:
            sign_neg += 1
        else:
            sign_flat += 1
    for cat, n, r_c in cat_rows:
        print(f"  {cat:16s} n={n:4d} r(area,correct)={r_c:+.3f}")
    print(f"  sign distribution: {sign_pos} positive / {sign_neg} negative / {sign_flat} ~flat / "
          f"{sign_nan} nan(zero-variance, i.e. 100% accuracy in that category) "
          f"(out of {len(cat_rows)} categories with n>=30)")
    print("  (if a large majority are positive, the area effect survives within-category and is "
          "not just category difficulty; if signs are scattered/mostly flat, the marginal curve "
          "above is likely a category-difficulty artifact)")

    # Category-fixed-effects regression: correct ~ patch_token_frac_z + p_yes_blank_z + category
    # dummies (one-hot, drop first). Same hand-rolled logistic regression as the primary test.
    cats_sorted = sorted(by_cat.keys())
    cat_index = {c: i for i, c in enumerate(cats_sorted[1:])}  # drop first as reference
    X_fe = []
    for r, az, bz in zip(positives, areas_z, blanks_z):
        onehot = [0.0] * len(cat_index)
        if r["category"] in cat_index:
            onehot[cat_index[r["category"]]] = 1.0
        X_fe.append([az, bz] + onehot)
    w_fe, b_fe = logistic_regression(X_fe, correct, iters=300, lr=0.1)
    print(f"\n  category-FE logistic regression: correct ~ patch_token_frac_z + p_yes_blank_z + "
          f"category dummies ({len(cat_index)} categories, 1 reference)")
    print(f"  coef(patch_token_frac_z) = {w_fe[0]:+.4f}   coef(p_yes_blank_z) = {w_fe[1]:+.4f}   "
          f"(compare to no-FE coef(patch_token_frac_z) = {w[0]:+.4f} above -- if it survives "
          f"with similar sign/magnitude, category isn't explaining the effect away)")

    # n_instances covariate note: area is a union-over-instances area, so "is there a person" with
    # 12 people is a very different evidential situation than one tiny person at the same area
    # fraction. Report the raw correlation as a flag, not yet folded into the locked regression.
    n_inst = [n_instances_by_key.get((r["split"], str(r["question_id"])), None) for r in positives]
    if all(v is not None for v in n_inst):
        r_area_ninst = pearson_r(areas, [float(v) for v in n_inst])
        r_ninst_correct = pearson_r([float(v) for v in n_inst], correct)
        print(f"\n  r(patch_token_frac, n_instances) = {r_area_ninst:+.4f}  "
              f"r(n_instances, correct) = {r_ninst_correct:+.4f}  "
              f"(n_instances entangled with area since area=union-over-instances; flagged, not "
              f"yet added as a third regression covariate)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    else:
        main()
