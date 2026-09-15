"""
Phase 75 analyzer: does the read-out finding generalise to a DIFFERENT TASK — token pruning?

WHY THIS IS THE HIGHEST-VALUE RUN LEFT
--------------------------------------
Everything so far is about OUR problem (where to crop). FastV and the visual-token-pruning
literature rank image tokens by attention AT ONE LAYER and drop the rest — a different task, a
different metric, an established baseline. If a better read-out improves pruning too, the finding
leaves this paper's problem and becomes a statement about a component the field shares.

ARMS (all prune the SAME number of tokens, so cost is matched by construction)
    none        no pruning — upper bound
    rand        random tokens dropped — the control that makes everything else readable
    layerK      rank by attention at layer 2 alone          (FastV-style)
    blockmean   rank by the mean over layers 16–26          (our deployed read-out)
    linear      rank by the SS14F learned SIGNED combination (the hypothesis)

DECISION ORDER, fixed before the data existed. Stop at the first failure.
 1. Does pruning bite at all? `rand` must fall below `none`, or the intervention is inert and
    nothing below is readable.
 2. Do INFORMED rankings beat `rand`? If not, attention carries no usable pruning signal here and
    the comparison among them is meaningless.
 3. Does `linear` beat `layerK` (the FastV baseline)? THIS IS THE CLAIM.
 4. Does `linear` beat `blockmean`? Separates "signed combination" from "just use more layers".

⚠ The SS14F weights were fitted to predict CROP-WINDOW COVERAGE, not pruning quality, and they come
from Qwen3-VL-2B. Using them unchanged is the honest transfer test; refitting on the pruning task
would answer a much weaker question.

⚠ SS14K rejected signed contrast as a GENERAL mechanism (it buys nothing on Qwen2-VL). So a null
here is unsurprising and must be reported as such, not buried. A POSITIVE here would be the single
strongest result in the project, because it would be the finding working outside its own problem.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase75_token_pruning.jsonl"


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
    KEEP = sorted({float(k.split("@")[1]) for k in R[0]["probs"] if "@" in k})
    CRIT = ["rand", "layerK", "blockmean", "linear"]
    print(f"n = {len(R)}   median image tokens {st.median([r['n_img'] for r in R]):.0f}")
    none = [hit(r, "none") for r in R]
    print(f"\n  no pruning: {100*st.mean(none):.1f}%")

    print(f"\n=== ACCURACY BY KEEP FRACTION ===")
    print(f"  {'keep':>6}" + "".join(f"{c:>12}" for c in CRIT))
    for kf in KEEP:
        row = ""
        for c in CRIT:
            v = [hit(r, f"{c}@{kf}") for r in R]
            row += f"{100*st.mean(v):11.1f}%"
        print(f"  {100*kf:5.0f}%" + row)

    print("\n=== DECISION ORDER ===")
    ok = True
    for kf in KEEP:
        rnd = [hit(r, f"rand@{kf}") for r in R]
        d = 100 * (st.mean(rnd) - st.mean(none))
        print(f"  [1] keep {100*kf:.0f}%: rand vs none {d:+.1f}pp "
              f"{'(pruning bites)' if d < -2 else '(INERT)'}")
    kf = KEEP[0]
    rnd = [hit(r, f"rand@{kf}") for r in R]
    for c in ["layerK", "blockmean", "linear"]:
        v = [hit(r, f"{c}@{kf}") for r in R]
        lo, hi = boot(v, rnd)
        print(f"  [2] keep {100*kf:.0f}%: {c:10} - rand {100*(st.mean(v)-st.mean(rnd)):+6.1f}pp "
              f"CI[{lo:+.1f},{hi:+.1f}]")

    print("\n" + "=" * 70)
    print("[3] THE CLAIM: linear (signed read-out) vs layerK (FastV baseline)")
    wins = 0
    for kf in KEEP:
        a = [hit(r, f"linear@{kf}") for r in R]
        b = [hit(r, f"layerK@{kf}") for r in R]
        lo, hi = boot(a, b)
        sig = "SIG" if lo > 0 else ("neg" if hi < 0 else "n.s.")
        wins += lo > 0
        print(f"  keep {100*kf:3.0f}%:  {100*(st.mean(a)-st.mean(b)):+6.1f}pp  "
              f"CI[{lo:+.1f},{hi:+.1f}]  {sig}")
    print("\n[4] vs blockmean (is it the SIGNED part, or just more layers?)")
    for kf in KEEP:
        a = [hit(r, f"linear@{kf}") for r in R]
        b = [hit(r, f"blockmean@{kf}") for r in R]
        lo, hi = boot(a, b)
        print(f"  keep {100*kf:3.0f}%:  {100*(st.mean(a)-st.mean(b)):+6.1f}pp  CI[{lo:+.1f},{hi:+.1f}]")

    print("\n" + "=" * 70)
    if wins >= 2:
        print("=> TRANSFERS. The read-out finding improves an unrelated, established task.")
        print("   This is the result that lifts the paper out of its own problem.")
    elif wins == 1:
        print("=> PARTIAL. One operating point only; report as suggestive, not as a claim.")
    else:
        print("=> DOES NOT TRANSFER. Report the negative. Consistent with SS14K, which already")
        print("   rejected signed contrast as a general mechanism.")


if __name__ == "__main__":
    main()
