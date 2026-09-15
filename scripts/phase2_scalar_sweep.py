"""
"Some steering mechanism" satisfied at zero GPU cost (per advisor, 2026-09-02): adding a constant
c to the yes-logit is a monotone transform of P(yes) -- new_p = sigmoid(logit(p) + c) -- so
sweeping c over the cached p_yes_real values retraces the existing ROC curve exactly. This is
mathematically identical to the pre-registered "yes-bias-matched scalar" control, computed directly
from data already in phase1_results.jsonl with no forward passes.

Produces: per-sweep-magnitude per-bin recall (the "beautiful dose-response scaling inversely with
area" curve the original steering plan wanted) AND the global negative-class FPR cost at the same
sweep point, in the same pass -- showing the tradeoff honestly rather than only the flattering half.
"""
import json
import math

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BIN_EDGES = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
EPS = 1e-6


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


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def shifted(p, c):
    return sigmoid(logit(p) + c)


def main():
    recs = load(f"{DATA}/phase1_results.jsonl")
    positives = [r for r in recs if r["label"] == "yes" and r["patch_token_frac"] is not None]
    negatives = [r for r in recs if r["label"] == "no"]
    neg_scores = [r["p_yes_real"] for r in negatives]

    bin_labels = [f"[{BIN_EDGES[i]},{BIN_EDGES[i+1]})" for i in range(len(BIN_EDGES) - 1)]
    pos_by_bin = {}
    for i, lab in enumerate(bin_labels):
        pos_by_bin[i] = [r["p_yes_real"] for r in positives if bin_of(r["patch_token_frac"], BIN_EDGES) == i]

    sweep = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    print(f"{'c':>6s} | " + " | ".join(f"{lab:>13s}" for lab in bin_labels) + f" | {'global_FPR':>10s} | {'balanced_acc':>12s}")
    pos_all = [r["p_yes_real"] for r in positives]
    for c in sweep:
        recalls = []
        for i in range(len(bin_labels)):
            ps = pos_by_bin[i]
            if len(ps) < 10:
                recalls.append(float("nan"))
                continue
            rec = sum(1 for p in ps if shifted(p, c) > 0.5) / len(ps)
            recalls.append(rec)
        fpr = sum(1 for p in neg_scores if shifted(p, c) > 0.5) / len(neg_scores)
        overall_recall = sum(1 for p in pos_all if shifted(p, c) > 0.5) / len(pos_all)
        balanced = (overall_recall + (1 - fpr)) / 2
        row = " | ".join(f"{r:13.3f}" if r == r else f"{'--':>13s}" for r in recalls)
        print(f"{c:6.1f} | {row} | {fpr:10.4f} | {balanced:12.4f}")

    print("\nInterpretation: as c increases, EVERY bin's recall rises (biggest relative gain in the")
    print("smallest-evidence bin, since it starts furthest below ceiling) -- this reproduces the")
    print("'dose-response scaling inversely with area' shape a real steering vector was predicted to")
    print("show. But global_FPR rises in lockstep and balanced_acc peaks near c=0 and DECLINES for")
    print("large c -- confirming this is a pure ROC-curve tradeoff, not a free evidence-size-specific")
    print("fix. A real steering vector would need to beat this curve (i.e. raise a bin's recall at a")
    print("GIVEN global FPR by more than this scalar sweep can), which the ceiling analysis in")
    print("phase1_threshold_analysis.py (gap=+0.016 for even a perfect oracle) says is not there to")
    print("find via any decision-rule-only correction.")


if __name__ == "__main__":
    main()
