"""
Phase 78 analyzer: is there anything for a learned allocation function to learn?

THE ONE NUMBER THAT DECIDES IT
------------------------------
    oracle-per-item-W   (best W for THIS item, at the head's placement)
  - best FIXED W        (one W for everybody, the current design)

That gap is the entire budget available to ANY learned sizer, however clever. It is also an
OPTIMISTIC bound twice over: the per-item W is chosen using the answer key, and chosen on the same
items it is scored on. So if the gap is small, the direction is closed; if it is large, the gap is
an upper bound to chase, not a result.

    gap < ~4pp   -> nothing to learn. Fixed W is near-optimal and a learned sizer cannot pay.
    gap > ~8pp   -> real headroom. Next question: does the attention blob's spatial extent predict
                    the right W? (SS14A/B killed predicting BUDGET from image features, but never
                    tried predicting EXTENT from the local structure of the map.)

SS14D's oracle PLACEMENT headroom was +21.5pp and perfect SIZING on top of perfect placement was
only +3.6pp. So the prior is that sizing is worth little -- but that was sizing to the GT BOX, which
maximises magnification. It is not the same as sizing to beat the SS13B cliff at an IMPERFECT
placement, where a wider window buys robustness to a near-miss. That is the untested case.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase78_w_sweep.jsonl"


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n)]


def main():
    R = [json.loads(l) for l in open(PATH)]
    if len(R) < 40:
        print(f"only {len(R)} rows; wait")
        return
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    WS = sorted({float(k.split("@")[1]) for k in R[0]["probs"] if k.startswith("head@")})
    print(f"n = {len(R)}   cats {dict((c, sum(1 for r in R if r['category']==c)) for c in sorted({r['category'] for r in R}))}")

    print("\n=== BUDGET GATE ===")
    bad = [a for a in R[0]["realized_tokens"]
           if abs(st.median([r["realized_tokens"][a] for r in R]) - (600 if "600" in a else 300))
           / (600 if "600" in a else 300) >= .10]
    print("  all arms within 10% of target" if not bad else f"  !! VOID: {bad}")

    print(f"\n=== ACCURACY vs WINDOW SIZE ===")
    print(f"  {'W':>6}{'head placement':>17}{'oracle placement':>19}{'magnification':>15}")
    for W in WS:
        h = st.mean([hit(r, f"head@{W}") for r in R])
        o = st.mean([hit(r, f"oracle@{W}") for r in R])
        print(f"  {W:6.2f}{100*h:16.1f}%{100*o:18.1f}%{1/W:14.1f}x")
    u3 = st.mean([hit(r, "uniform@300") for r in R])
    u6 = st.mean([hit(r, "uniform@600") for r in R])
    print(f"  {'uniform@300':>6}{100*u3:16.1f}%")
    print(f"  {'@600 (bar)':>6}{100*u6:16.1f}%")

    bestW = max(WS, key=lambda W: st.mean([hit(r, f"head@{W}") for r in R]))
    fixed = [hit(r, f"head@{bestW}") for r in R]
    per_item = [max(hit(r, f"head@{W}") for W in WS) for r in R]
    lo, hi = boot(per_item, fixed)
    print("\n" + "=" * 70)
    print(f"  best FIXED W          = {bestW:.2f}  ->  {100*st.mean(fixed):.1f}%")
    print(f"  ORACLE per-item W                 ->  {100*st.mean(per_item):.1f}%")
    print(f"  GAP (all a learned sizer could win) = {100*(st.mean(per_item)-st.mean(fixed)):+.1f}pp "
          f"CI[{lo:+.1f},{hi:+.1f}]")
    gap = st.mean(per_item) - st.mean(fixed)
    if gap < 0.04:
        print("\n  => NOTHING TO LEARN. Fixed W is near-optimal; a learned sizer cannot pay.")
    elif gap > 0.08:
        print("\n  => REAL HEADROOM (an optimistic upper bound: W chosen with the answer key,")
        print("     on the same items). Worth asking whether attention extent predicts W.")
    else:
        print("\n  => MARGINAL. The gap is smaller than the bound's own optimism.")

    print("\n=== does the best W track TARGET SIZE? (what a sizer would have to learn) ===")
    qs = sorted(R, key=lambda r: r["gt_area_frac"])
    k = max(1, len(qs) // 4)
    for i, nm in enumerate(["smallest 25%", "25-50%", "50-75%", "largest 25%"]):
        S = qs[i * k:(i + 1) * k] if i < 3 else qs[3 * k:]
        if not S:
            continue
        bw = [min((W for W in WS if hit(r, f"head@{W}")), default=float('nan')) for r in S]
        ok = [w for w in bw if w == w]
        print(f"  {nm:14} n={len(S):3d}  median area {st.median([r['gt_area_frac'] for r in S]):.5f}"
              f"  best W (median of solved) {st.median(ok) if ok else float('nan'):.2f}"
              f"  solved {len(ok)}/{len(S)}")


if __name__ == "__main__":
    main()
