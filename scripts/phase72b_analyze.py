"""
Phase 72b analyzer: did the V*Bench-trained head TRANSFER to HR-Bench 4k?

Nothing is fitted anywhere in phases 72a/72b on HR-Bench data. HR-Bench ships no boxes, so the
coverage labels the head was trained on do not exist here even in principle.

PRE-REGISTERED, fixed before the data existed
---------------------------------------------
SS14D found the head beats the deployed argmax in BOTH V*Bench regimes (coverage +13.0pp on
single-region, +14.5pp on relational), but beats the COMPUTE-MATCHED BAR only on single-region
(+13.0pp [+2.6,+23.5]) and loses on relational (-7.9pp). HR-Bench's `single`/`cross` categories are
the direct analogue of that split. So:

    P1  head > argmax overall                      -- the re-ranking transfers at all
    P2  head > uniform@600 on `single`             -- the method transfers where SS7 says it can
    P3  head <= uniform@600 on `cross`             -- the boundary transfers too

P3 failing POSITIVELY (head wins on cross) would be a surprise worth investigating, not a bonus:
it would mean the coverage account of the boundary is wrong.

CONTEXT: allocation has historically LOST here. SS9B measured our gated method at -0.2pp and Zoom
Eye at -15.3pp against uniform on this benchmark. A null is the default expectation.

NO ORACLE ARM and no coverage strata: HR-Bench ships no boxes. Nothing requiring them is quoted.
"""
import json
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase72b_hrbench_transfer.jsonl"


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n)]


def main():
    R = [json.loads(l) for l in open(PATH)]
    if len(R) < 80:
        print(f"only {len(R)} rows; wait")
        return
    ARMS = list(R[0]["probs"].keys())
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    inst = {r["instance"] for r in R}
    print(f"n = {len(R)} rows over {len(inst)} instances")
    print("  categories:", {c: sum(1 for r in R if r["category"] == c)
                            for c in sorted({r["category"] for r in R})})
    diff = st.mean([r["head_cell"] != r["argmax_cell"] for r in R])
    print(f"  head and argmax propose DIFFERENT cells on {100*diff:.1f}% of rows\n")

    print("=== BUDGET GATE (measured) ===")
    void = set()
    for a in ARMS:
        med = st.median([r["realized_tokens"][a] for r in R])
        tgt = 600 if a == "uniform@600" else 300
        off = abs(med - tgt) / tgt
        if off >= .10:
            void.add(a)
        print(f"  {a:14}{med:6.0f} tok (target {tgt})  {100*off:4.1f}% off  "
              f"{'OK' if off < .10 else '!! VOID'}")

    acc = {a: [hit(r, a) for r in R] for a in ARMS}
    print(f"\n=== PER-ROW ACCURACY (n={len(R)}) ===")
    for a in ARMS:
        print(f"  {a:14}{100*st.mean(acc[a]):6.1f}%")

    print("\n=== CIRCULAR EVAL (instance counts only if ALL 4 cycles correct) ===")
    by = defaultdict(list)
    for i, r in enumerate(R):
        by[r["instance"]].append(i)
    full = [k for k, v in by.items() if len(v) >= 4]
    circ = {}
    for a in ARMS:
        circ[a] = [min(acc[a][i] for i in by[k]) for k in full]
    print(f"  instances with all 4 cycles: {len(full)}/{len(by)}")
    for a in ARMS:
        print(f"  {a:14}{100*st.mean(circ[a]):6.1f}%")

    print("\n=== THE CONTRASTS (per-row) ===")
    for lo, hi, why in [("argmax@0.15", "head@0.15", "P1: does re-ranking transfer at all?"),
                        ("uniform@600", "head@0.15", "vs COMPUTE-MATCHED BAR"),
                        ("uniform@300", "head@0.15", "vs B0 baseline"),
                        ("rand@0.15", "head@0.15", "vs random-placement control")]:
        d = 100 * (st.mean(acc[hi]) - st.mean(acc[lo]))
        l, h = boot(acc[hi], acc[lo])
        flag = " [VOID: budget]" if (lo in void or hi in void) else ""
        print(f"  {hi:12} - {lo:12}{d:+7.1f}pp  CI[{l:+.1f},{h:+.1f}]  {why}{flag}")

    print("\n=== INTERNAL CONTROL: rows where head and argmax chose the SAME cell ===")
    same = [i for i, r in enumerate(R) if r["head_cell"] == r["argmax_cell"]]
    dif = [i for i, r in enumerate(R) if r["head_cell"] != r["argmax_cell"]]
    for nm, idx in [("identical proposal", same), ("different proposal", dif)]:
        if not idx:
            continue
        h = [acc["head@0.15"][i] for i in idx]
        a = [acc["argmax@0.15"][i] for i in idx]
        l, hh = boot(h, a)
        print(f"  {nm:20} n={len(idx):4d}  head-argmax {100*(st.mean(h)-st.mean(a)):+6.1f}pp "
              f"CI[{l:+.1f},{hh:+.1f}]")
    print("  (identical-proposal must be ~0; non-zero voids the headline)")

    print("\n=== BY CATEGORY -- P2 (`single`) and P3 (`cross`) ===")
    for c in sorted({r["category"] for r in R}):
        idx = [i for i, r in enumerate(R) if r["category"] == c]
        h = [acc["head@0.15"][i] for i in idx]
        a = [acc["argmax@0.15"][i] for i in idx]
        b = [acc["uniform@600"][i] for i in idx]
        l, hh = boot(h, b)
        la, ha = boot(h, a)
        print(f"  {c:8} n={len(idx):4d}  head {100*st.mean(h):5.1f}%  argmax {100*st.mean(a):5.1f}%"
              f"  bar {100*st.mean(b):5.1f}%")
        print(f"           head-bar {100*(st.mean(h)-st.mean(b)):+6.1f}pp CI[{l:+.1f},{hh:+.1f}]"
              f"   head-argmax {100*(st.mean(h)-st.mean(a)):+6.1f}pp CI[{la:+.1f},{ha:+.1f}]")

    print("\n" + "=" * 72)
    hm, am, bm = (st.mean(acc[k]) for k in ("head@0.15", "argmax@0.15", "uniform@600"))
    la, ha = boot(acc["head@0.15"], acc["argmax@0.15"])
    print(f"VERDICT  head {100*hm:.1f}%  argmax {100*am:.1f}%  bar {100*bm:.1f}%")
    print(f"  P1 head-argmax = {100*(hm-am):+.1f}pp CI[{la:+.1f},{ha:+.1f}]  "
          f"{'TRANSFERS' if la > 0 else 'DOES NOT TRANSFER'}")
    sing = [i for i, r in enumerate(R) if r["category"] == "single"]
    if sing:
        h = [acc["head@0.15"][i] for i in sing]; b = [acc["uniform@600"][i] for i in sing]
        l, hh = boot(h, b)
        d = 100 * (st.mean(h) - st.mean(b))
        print(f"  P2 single: head-bar = {d:+.1f}pp CI[{l:+.1f},{hh:+.1f}]  "
              f"{'METHOD TRANSFERS' if l > 0 else 'not significant here'}")
    print("  (SS9B context: our gated method -0.2pp, Zoom Eye -15.3pp on this benchmark)")


if __name__ == "__main__":
    main()
