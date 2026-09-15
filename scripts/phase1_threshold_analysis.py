"""
Threshold analysis on phase1_results.jsonl, run BEFORE committing to the miscalibration framing
or building Phase 2 steering code (per advisor review, 2026-09-02).

Question: is "small-object accuracy is low despite high AUROC" a real miscalibration (a single
global threshold shift recovers small-bin accuracy near-for-free) or just the normal
precision/recall tradeoff along a fixed ROC curve (recovering small-bin recall costs a comparable
rise in false-positive rate on negatives)?

1. Global tau* maximizing overall balanced accuracy across all 5553 items.
2. At tau=0.5 vs tau=tau*: per-bin positive accuracy (recall) AND negative-class accuracy
   (1-FPR), the latter computed once globally (negatives aren't binned by area).
3. Per-bin oracle threshold (maximizes that bin's own balanced accuracy vs all negatives) and its
   accuracy -- the ceiling for any evidence-size-conditioned correction.
4. The discriminating number: threshold that brings smallest-bin recall to ~0.83 (matching bin 2's
   current recall) -- what FPR does that cost on all 1500 negatives?
"""
import json
import random

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BIN_EDGES = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]


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


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def bin_of(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
            return i
    return None


def balanced_acc(pos_scores, neg_scores, tau):
    tpr = sum(1 for s in pos_scores if s > tau) / len(pos_scores) if pos_scores else float("nan")
    tnr = sum(1 for s in neg_scores if s <= tau) / len(neg_scores) if neg_scores else float("nan")
    return (tpr + tnr) / 2, tpr, tnr


def main(path=f"{DATA}/phase1_results.jsonl", area_key="patch_token_frac"):
    recs = load(path)
    positives = [r for r in recs if r["label"] == "yes" and r[area_key] is not None]
    negatives = [r for r in recs if r["label"] == "no"]
    pos_all_scores = [r["p_yes_real"] for r in positives]
    neg_scores = [r["p_yes_real"] for r in negatives]

    print(f"n_positives={len(positives)} n_negatives={len(negatives)}\n")

    # --- 1. Global tau* maximizing overall balanced accuracy ---
    candidates = sorted(set(pos_all_scores + neg_scores))
    best_tau, best_ba = 0.5, -1
    for tau in candidates:
        ba, _, _ = balanced_acc(pos_all_scores, neg_scores, tau)
        if ba > best_ba:
            best_ba, best_tau = ba, tau
    print(f"--- Global tau* (max overall balanced accuracy) ---")
    print(f"tau* = {best_tau:.4f}  overall balanced_acc={best_ba:.4f}")
    ba_05, tpr_05, tnr_05 = balanced_acc(pos_all_scores, neg_scores, 0.5)
    ba_star, tpr_star, tnr_star = balanced_acc(pos_all_scores, neg_scores, best_tau)
    print(f"at tau=0.5   : overall_pos_acc(recall)={tpr_05:.4f} neg_acc(1-FPR)={tnr_05:.4f} balanced={ba_05:.4f}")
    print(f"at tau=tau*  : overall_pos_acc(recall)={tpr_star:.4f} neg_acc(1-FPR)={tnr_star:.4f} balanced={ba_star:.4f}\n")

    # --- 2. Per-bin accuracy at tau=0.5 vs tau=tau*, single shared negative accuracy ---
    print("--- Per-bin positive accuracy (recall) at tau=0.5 vs tau=tau*, with global negative accuracy ---")
    bin_labels = [f"[{BIN_EDGES[i]},{BIN_EDGES[i+1]})" for i in range(len(BIN_EDGES) - 1)]
    for i, lab in enumerate(bin_labels):
        pos_in_bin = [r["p_yes_real"] for r in positives if bin_of(r[area_key], BIN_EDGES) == i]
        if len(pos_in_bin) < 10:
            continue
        rec_05 = sum(1 for s in pos_in_bin if s > 0.5) / len(pos_in_bin)
        rec_star = sum(1 for s in pos_in_bin if s > best_tau) / len(pos_in_bin)
        print(f"  bin {lab}: n={len(pos_in_bin)} recall@0.5={rec_05:.3f} recall@tau*={rec_star:.3f} "
              f"(global neg_acc@0.5={tnr_05:.3f}, @tau*={tnr_star:.3f})")
    print()

    # --- 3. Per-bin oracle threshold (ceiling) ---
    print("--- Per-bin ORACLE threshold (maximizes that bin's own balanced acc vs all negatives) ---")
    for i, lab in enumerate(bin_labels):
        pos_in_bin = [r["p_yes_real"] for r in positives if bin_of(r[area_key], BIN_EDGES) == i]
        if len(pos_in_bin) < 10:
            continue
        cands = sorted(set(pos_in_bin + neg_scores))
        best_t, best_b, best_tpr, best_tnr = 0.5, -1, 0, 0
        for t in cands:
            b, tpr, tnr = balanced_acc(pos_in_bin, neg_scores, t)
            if b > best_b:
                best_b, best_t, best_tpr, best_tnr = b, t, tpr, tnr
        print(f"  bin {lab}: n={len(pos_in_bin)} oracle_tau={best_t:.3f} balanced_acc={best_b:.3f} "
              f"(recall={best_tpr:.3f}, neg_acc={best_tnr:.3f})")
    print()

    # --- 4. The discriminating number (fixed: pick the threshold via the recall-sorted quantile,
    # not a descending-scan-with-break which broke on the first iteration and silently fell
    # through to min(scores) -- caught by advisor review before this was logged as the headline
    # evidence) ---
    print("--- Discriminating check: cost (in negative FPR) of lifting smallest-bin recall to ~0.83 ---")
    smallest_bin_pos = [r["p_yes_real"] for r in positives if bin_of(r[area_key], BIN_EDGES) == 0]
    target_recall = 0.83
    sorted_scores = sorted(smallest_bin_pos)
    idx = int((1 - target_recall) * len(sorted_scores))
    idx = max(0, min(idx, len(sorted_scores) - 1))
    t = sorted_scores[idx]
    recall_at = sum(1 for s in smallest_bin_pos if s > t) / len(smallest_bin_pos)
    fpr_at = sum(1 for s in neg_scores if s > t) / len(neg_scores)
    fpr_05 = sum(1 for s in neg_scores if s > 0.5) / len(neg_scores)
    print(f"  threshold needed = {t:.4f} (achieves recall={recall_at:.3f} on smallest bin)")
    print(f"  FPR on negatives at this threshold = {fpr_at:.4f}  (vs FPR at 0.5 = {fpr_05:.4f})")
    print(f"  -> if FPR roughly triples or worse (>= ~0.30-0.35), this is a plain ROC tradeoff, "
          f"not free miscalibration recovery")

    # --- 5. Oracle-per-bin AGGREGATE balanced accuracy vs global tau* (the number that decides
    # whether Phase 2 is worth GPU time: if area-conditioned oracle ceiling barely beats the
    # single global threshold, no area-conditioned method -- steering included -- has room to
    # matter). Compute each bin's oracle stats once and reuse, rather than recomputing per term. ---
    print("\n--- Oracle-per-bin AGGREGATE balanced accuracy vs single global tau* ---")
    bin_stats = []  # (n_pos_in_bin, oracle_recall, oracle_neg_acc)
    for i, lab in enumerate(bin_labels):
        pos_in_bin = [r["p_yes_real"] for r in positives if bin_of(r[area_key], BIN_EDGES) == i]
        if len(pos_in_bin) < 10:
            continue
        cands = sorted(set(pos_in_bin + neg_scores))
        best_bal, best_tpr, best_tnr = -1, 0, 0
        for tt in cands:
            bal, tpr, tnr = balanced_acc(pos_in_bin, neg_scores, tt)
            if bal > best_bal:
                best_bal, best_tpr, best_tnr = bal, tpr, tnr
        bin_stats.append((len(pos_in_bin), best_tpr, best_tnr))

    total_n_pos = sum(n for n, _, _ in bin_stats)
    oracle_pos_acc = sum(n * tpr for n, tpr, _ in bin_stats) / total_n_pos
    # Negative accuracy isn't naturally per-bin (negatives have no area); report the n-weighted
    # average of each bin's own oracle neg_acc as the idealized companion number -- an upper-bound
    # sense of "if you could always pick the right bin-specific operating point."
    avg_neg_acc_oracle = sum(n * tnr for n, _, tnr in bin_stats) / total_n_pos
    oracle_balanced = (oracle_pos_acc + avg_neg_acc_oracle) / 2
    print(f"  idealized per-bin-oracle: positive accuracy(recall)={oracle_pos_acc:.4f}, "
          f"(n-weighted-avg) neg_acc={avg_neg_acc_oracle:.4f}, balanced~={oracle_balanced:.4f}")
    print(f"  single global tau* balanced accuracy = {best_ba:.4f}")
    gap = oracle_balanced - best_ba
    print(f"  gap = {gap:+.4f}  "
          f"(this is the ceiling for ANY area-conditioned method incl. steering -- if this is "
          f"under ~0.02-0.03, Phase 2 is fighting for noise-level headroom)")

    # --- 6. Bootstrap 95% CI on smallest-bin AUROC (so a future steering delta can be judged
    # against real measurement noise, not eyeballed) ---
    print("\n--- Bootstrap 95% CI on smallest-bin AUROC (n_boot=1000) ---")
    random.seed(0)
    boot_aurocs = []
    n_pos_s, n_neg_s = len(smallest_bin_pos), len(neg_scores)
    for _ in range(1000):
        bp = [smallest_bin_pos[random.randrange(n_pos_s)] for _ in range(n_pos_s)]
        bn = [neg_scores[random.randrange(n_neg_s)] for _ in range(n_neg_s)]
        boot_aurocs.append(auroc(bp, bn))
    boot_aurocs.sort()
    lo = boot_aurocs[int(0.025 * len(boot_aurocs))]
    hi = boot_aurocs[int(0.975 * len(boot_aurocs))]
    point = auroc(smallest_bin_pos, neg_scores)
    print(f"  smallest-bin AUROC = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  -> any steering-vector AUROC delta on this bin should exceed ~{(hi-lo):.3f} "
          f"(the CI width) to be distinguishable from noise; per advisor, treat anything "
          f"below ~+0.04 as noise-level")


if __name__ == "__main__":
    main()
