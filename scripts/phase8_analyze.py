"""
Phase 8 analysis: does a grounding-derived presence score beat P(yes) on a BROAD, unbiased
positives-vs-negatives sample, at matched compute?

Reports, in the order a reviewer will demand them:
  1. Headline: AUROC(p_ground) vs AUROC(p_yes) on identical items, with a PAIRED bootstrap CI on
     the difference (not two CIs quoted side by side).
  2. Blank-image null floors for BOTH channels. POPE asks different questions of positives and
     negatives, so a detector can beat chance from category priors with no perception at all. If the
     grounding advantage survives on black images, it is a language prior, not a perception channel.
  3. Per-split AUROC. Winning only on `random` would mean exploiting a co-occurrence prior.
  4. Area stratification -- the dose-response link to the §2.1 scaling law.
  5. Balanced accuracy at the oracle threshold, and the unmatched-compute grounding scores
     (box emission, n_boxes) reported separately so the matched-cost claim stays clean.
"""
import json
import random
import sys
from collections import defaultdict

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
PATH = f"{DATA}/phase8_grounding_detector_results.jsonl"


def auroc(pos, neg):
    """Rank-based AUROC with correct mid-rank handling of ties (ties matter a lot here: p_ground
    saturates at 1.0 for easy positives)."""
    if not pos or not neg:
        return float("nan")
    lab = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    n_pos, n_neg, n = len(pos), len(neg), len(lab)
    rank_sum, i = 0.0, 0
    while i < n:
        j = i
        while j < n and lab[j][0] == lab[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rank_sum += avg * sum(1 for k in range(i, j) if lab[k][1] == 1)
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_balanced_acc(pos, neg):
    """Oracle (best-case, in-sample) balanced accuracy over all thresholds -- an upper bound, and
    labeled as such wherever it is quoted."""
    thr = sorted(set(pos + neg))
    best = (0.0, None)
    for t in thr:
        sens = sum(1 for s in pos if s >= t) / len(pos)
        spec = sum(1 for s in neg if s < t) / len(neg)
        ba = (sens + spec) / 2
        if ba > best[0]:
            best = (ba, t)
    return best


def paired_boot(pos_a, neg_a, pos_b, neg_b, n=3000, seed=11):
    """CI on AUROC(a) - AUROC(b) where a and b are two scores on the SAME items. Positives and
    negatives are resampled independently (each defines one side of the ranking), and both scores
    are evaluated on the identical resample -- so the CI is on the paired difference."""
    rng = random.Random(seed)
    npos, nneg = len(pos_a), len(neg_a)
    diffs = []
    for _ in range(n):
        pi = [rng.randrange(npos) for _ in range(npos)]
        ni = [rng.randrange(nneg) for _ in range(nneg)]
        da = auroc([pos_a[i] for i in pi], [neg_a[i] for i in ni])
        db = auroc([pos_b[i] for i in pi], [neg_b[i] for i in ni])
        diffs.append(da - db)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n)]
    frac = sum(1 for d in diffs if d > 0) / n
    return lo, hi, frac


def split_scores(rows, key):
    pos = [r[key] for r in rows if r["group"] == "positive"]
    neg = [r[key] for r in rows if r["group"] == "negative"]
    return pos, neg


def main():
    rows = [json.loads(l) for l in open(PATH)]
    # de-duplicate on uid (the file is append-mode + resumable; a smoke run precedes the full run)
    seen, uniq = set(), []
    for r in rows:
        if r["uid"] not in seen:
            seen.add(r["uid"])
            uniq.append(r)
    rows = uniq
    has_gen = [r for r in rows if "n_pred_boxes" in r]
    npos = sum(1 for r in rows if r["group"] == "positive")
    print(f"n={len(rows)}  positives={npos}  negatives={len(rows)-npos}  with_generation={len(has_gen)}")

    bad = [r for r in rows if r["ground"]["ground_mass_captured"] < 0.9]
    print(f"\n[coverage check] items where the two-token grounding score captures <90% of the "
          f"first-token mass: {len(bad)}/{len(rows)}")
    if bad:
        from collections import Counter
        print("  argmax tokens on those items:",
              Counter(r["ground"]["argmax_tok"] for r in bad).most_common(8))

    print("\n" + "=" * 78)
    print("1. HEADLINE -- matched compute (one forward pass each), broad unselected sample")
    print("=" * 78)
    py_p, py_n = split_scores(rows, "p_yes_fp16")
    pg_p, pg_n = split_scores(rows, "p_ground")
    a_py, a_pg = auroc(py_p, py_n), auroc(pg_p, pg_n)
    print(f"  AUROC  P(yes)   [standard readout] = {a_py:.4f}")
    print(f"  AUROC  p_ground [grounding channel] = {a_pg:.4f}")
    lo, hi, frac = paired_boot(pg_p, pg_n, py_p, py_n)
    print(f"  paired delta = {a_pg - a_py:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"{frac*100:.1f}% of resamples favor grounding")
    for name, (p, n) in [("P(yes)", (py_p, py_n)), ("p_ground", (pg_p, pg_n))]:
        ba, t = best_balanced_acc(p, n)
        print(f"  oracle-threshold balanced acc, {name:9s} = {ba:.4f} (at t={t:.6g}) [in-sample UPPER BOUND]")

    print("\n" + "=" * 78)
    print("2. BLANK-IMAGE NULL FLOORS -- how much is language prior, not perception?")
    print("=" * 78)
    with_blank = [r for r in rows if r.get("p_yes_blank_cached") is not None]
    byp, byn = split_scores(with_blank, "p_yes_blank_cached")
    print(f"  AUROC  P(yes) on BLANK image  = {auroc(byp, byn):.4f}   (n={len(with_blank)}; "
          f"cached from phase1, 4-bit)")
    bgp, bgn = split_scores(rows, "p_ground_blank")
    a_bg = auroc(bgp, bgn)
    print(f"  AUROC  p_ground on BLANK image = {a_bg:.4f}")
    print(f"  -> perception-attributable gain for grounding: {a_pg:.4f} - {a_bg:.4f} = {a_pg-a_bg:+.4f}")
    lo, hi, frac = paired_boot(pg_p, pg_n, bgp, bgn)
    print(f"     paired CI vs its own blank floor [{lo:+.4f}, {hi:+.4f}], {frac*100:.1f}% favor real image")

    print("\n" + "=" * 78)
    print("3. PER-SPLIT AUROC -- is the win only on the easy `random` split?")
    print("=" * 78)
    print(f"  {'split':<13}{'n_pos':>6}{'n_neg':>6}{'P(yes)':>10}{'p_ground':>10}{'delta':>9}")
    for sp in ["random", "popular", "adversarial"]:
        sub = [r for r in rows if r["split"] == sp]
        p1, n1 = split_scores(sub, "p_yes_fp16")
        p2, n2 = split_scores(sub, "p_ground")
        if not p1 or not n1:
            continue
        print(f"  {sp:<13}{len(p1):>6}{len(n1):>6}{auroc(p1,n1):>10.4f}{auroc(p2,n2):>10.4f}"
              f"{auroc(p2,n2)-auroc(p1,n1):>+9.4f}")

    print("\n" + "=" * 78)
    print("4. AREA STRATIFICATION -- dose-response against the §2.1 scaling law")
    print("=" * 78)
    print("  (negatives have no area; the SAME negative pool is reused for every positive stratum,")
    print("   so strata are comparable in their negative arm by construction)")
    pos_rows = [r for r in rows if r["group"] == "positive" and r.get("pixel_area_frac") is not None]
    neg_rows = [r for r in rows if r["group"] == "negative"]
    pos_rows.sort(key=lambda r: r["pixel_area_frac"])
    q = len(pos_rows) // 4
    print(f"  {'area quartile':<16}{'n':>5}{'med area':>11}{'P(yes)':>10}{'p_ground':>10}{'delta':>9}")
    for i in range(4):
        chunk = pos_rows[i*q: (i+1)*q if i < 3 else len(pos_rows)]
        med = chunk[len(chunk)//2]["pixel_area_frac"]
        a1 = auroc([r["p_yes_fp16"] for r in chunk], [r["p_yes_fp16"] for r in neg_rows])
        a2 = auroc([r["p_ground"] for r in chunk], [r["p_ground"] for r in neg_rows])
        print(f"  Q{i+1} {'(smallest)' if i==0 else '(largest)' if i==3 else '':<12}{len(chunk):>5}"
              f"{med:>11.5f}{a1:>10.4f}{a2:>10.4f}{a2-a1:>+9.4f}")

    if has_gen:
        print("\n" + "=" * 78)
        print("5. UNMATCHED-COMPUTE grounding scores (~100 decode steps) -- reported SEPARATELY;")
        print("   these are NOT part of the matched-cost claim above.")
        print("=" * 78)
        mean_tok = sum(r["n_gen_tokens"] for r in has_gen) / len(has_gen)
        print(f"  mean generated tokens = {mean_tok:.1f}  (vs 1 forward pass for p_yes / p_ground)")
        for key, label in [("n_pred_boxes", "n boxes emitted")]:
            p, n = split_scores(has_gen, key)
            print(f"  AUROC  {label:<18} = {auroc(p,n):.4f}")
        ep = [1.0 if r["n_pred_boxes"] > 0 else 0.0 for r in has_gen if r["group"] == "positive"]
        en = [1.0 if r["n_pred_boxes"] > 0 else 0.0 for r in has_gen if r["group"] == "negative"]
        sens, spec = sum(ep)/len(ep), 1 - sum(en)/len(en)
        print(f"  binary box-emission: sens={sens:.4f} spec={spec:.4f} balanced acc={(sens+spec)/2:.4f}")
        print(f"  AUROC  box emission (binary)  = {auroc(ep, en):.4f}")


if __name__ == "__main__":
    main()
