"""
Phase 90: does W=0.25 survive HELD-OUT selection, or is it selection bias?

WHY THIS MUST COME FIRST
------------------------
W=0.25 scores 71.7% against the deployed W=0.15's 68.6%, and +7.9pp [-0.5,+16.2] against the
compute-matched bar -- about one point from the claim that is currently 0-for-2. But it was chosen as
the best of five values ON THE SAME 191 ITEMS it is scored on. Phase 74 already showed what that does:
an in-sample "best layer" read 41.9% and collapsed to 36.1% out-of-fold, BELOW the baseline it
appeared to beat. Nothing downstream is worth running until this is settled.

DESIGN
------
Nested selection. For each of 5 folds: choose W on the 4 TRAINING folds, apply it to the held-out
fold, concatenate the held-out predictions. Repeated over 20 seeds so the estimate does not depend on
one partition. The head's proposals are already out-of-fold, so W is the only thing being selected.

ARMS
    W = 0.15 fixed        pre-registered, no selection at all -- the honest baseline
    W = 0.25 fixed        the in-sample winner, scored in-sample (OPTIMISTIC, shown for contrast)
    W chosen per fold     the honest version of "choose W by CV"
    W chosen per fold, per CATEGORY   relational and single-region may want different windows;
                                      selection still happens on training folds only

READING, fixed before the run
    held-out ~= 71.7%  -> W=0.25 is real, the starting point for Track B is +15.2pp over vanilla
    held-out ~= 68.6%  -> the gain was selection bias; the deployed W was already right
    held-out < 68.6%   -> selecting W at all is harmful at this n, and W should stay fixed
"""
import json
import random
import statistics as st

import numpy as np

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase78_w_sweep.jsonl"
WS = [0.15, 0.25, 0.35, 0.50, 0.70]
SEEDS, K = 20, 5


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return round(100 * s[int(.025 * n)], 1), round(100 * s[int(.975 * n)], 1)


def main():
    R = [json.loads(l) for l in open(PATH)]
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    n = len(R)
    H = {W: [hit(r, f"head@{W}") for r in R] for W in WS}
    u6 = [hit(r, "uniform@600") for r in R]
    u3 = [hit(r, "uniform@300") for r in R]
    cat = [r["category"] for r in R]
    print(f"n = {n}   uniform@300 {100*st.mean(u3):.1f}%   uniform@600 (bar) {100*st.mean(u6):.1f}%\n")

    # ---- nested selection, pooled and per-category
    sel_pool = np.zeros(n); sel_cat = np.zeros(n); picks_p, picks_c = [], []
    for sd in range(SEEDS):
        rng = np.random.default_rng(900 + sd)
        order = rng.permutation(n)
        folds = np.array_split(order, K)
        for f in folds:
            te = set(f.tolist())
            tr = [i for i in range(n) if i not in te]
            bw = max(WS, key=lambda W: st.mean([H[W][i] for i in tr]))
            picks_p.append(bw)
            for i in f:
                sel_pool[i] += H[bw][i]
            for c in set(cat):
                trc = [i for i in tr if cat[i] == c]
                tec = [i for i in f if cat[i] == c]
                if not trc or not tec:
                    continue
                bwc = max(WS, key=lambda W: st.mean([H[W][i] for i in trc]))
                picks_c.append((c, bwc))
                for i in tec:
                    sel_cat[i] += H[bwc][i]
    sel_pool /= SEEDS; sel_cat /= SEEDS
    sp = [float(x) for x in sel_pool]; sc = [float(x) for x in sel_cat]

    print(f"  {'arm':34}{'acc':>8}{'vs bar':>10}{'CI':>16}{'vs vanilla':>13}")
    for nm, v in [("W=0.15 fixed (pre-registered)", H[0.15]),
                  ("W=0.25 fixed (IN-SAMPLE pick)", H[0.25]),
                  ("W chosen per fold (held-out)", sp),
                  ("W per fold, per category", sc)]:
        lo, hi = boot(v, u6)
        print(f"  {nm:34}{100*st.mean(v):7.1f}%{100*(st.mean(v)-st.mean(u6)):+9.1f}"
              f"  [{lo:+.1f},{hi:+.1f}]{100*(st.mean(v)-st.mean(u3)):+12.1f}")

    from collections import Counter
    print(f"\n  W picked by the folds (pooled): {dict(Counter(picks_p))}")
    for c in sorted(set(cat)):
        print(f"    {c:20} {dict(Counter(w for cc, w in picks_c if cc == c))}")

    print("\n" + "=" * 70)
    ho, ins, fix = st.mean(sp), st.mean(H[0.25]), st.mean(H[0.15])
    print(f"  in-sample W=0.25 {100*ins:.1f}%  ->  held-out selection {100*ho:.1f}%  "
          f"(fixed W=0.15 {100*fix:.1f}%)")
    print(f"  selection cost: {100*(ins-ho):+.1f}pp")
    if ho >= ins - 0.01:
        print("  => W=0.25 IS REAL. Selecting W generalises; Track B starts from the higher number.")
    elif ho > fix + 0.01:
        print("  => PARTIAL. Selection helps over the fixed window but less than in-sample implied.")
    else:
        print("  => SELECTION BIAS. The apparent gain does not survive; keep W fixed at 0.15 and")
        print("     the starting point for Track B stays +4.7pp against the bar.")


if __name__ == "__main__":
    main()
