"""
Phase 71b analyzer. Nothing is fitted here; proposals were frozen out-of-fold in phase71a.

PRE-REGISTERED, written before the data existed
-----------------------------------------------
SS6D's coverage strata (miss -15.6pp, full cover +37.1pp, a 52.7pp swing) and Phase 70's
39.3% -> 52.9% coverage shift predict

        head@0.15 - argmax@0.15  =  0.136 x 52.7  =  +7.2pp

THE DECISION RULE
    head > uniform@600 (the compute-matched bar)     -> A METHOD. Allocation beats spending the
                                                        same budget, which nothing in SS2 managed.
    head > argmax but <= uniform@600                 -> the head works, allocation still loses.
                                                        Report as a component, not a method.
    head ~= argmax                                   -> coverage is not the mediator SS6D claims,
                                                        and SS14C's gain does not convert.

INTERNAL CONTROL
    On items where the head and the argmax chose the SAME cell (41.4%), the two arms are the same
    computation and their contrast MUST be ~0. A non-zero split there means run-to-run nondeterminism
    or a bookkeeping error, and voids the headline rather than decorating it.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71b_endtask.jsonl"
PROP = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71a_head_proposals.json"
PREDICTED = 7.2


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
    ARMS = list(R[0]["probs"].keys())
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    print(f"n = {len(R)}\n")

    print("=== BUDGET GATE (measured, never computed) ===")
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
    print(f"\n=== ACCURACY (n={len(R)}) ===")
    for a in ARMS:
        print(f"  {a:14}{100*st.mean(acc[a]):6.1f}%")

    print("\n=== THE CONTRASTS THAT DECIDE IT ===")
    for lo, hi, why in [
            ("argmax@0.15", "head@0.15", f"THE METHOD vs incumbent (predicted {PREDICTED:+.1f}pp)"),
            ("uniform@600", "head@0.15", "THE METHOD vs COMPUTE-MATCHED BAR  <- decides 'method'"),
            ("uniform@300", "head@0.15", "vs B0 baseline"),
            ("rand@0.15", "head@0.15", "vs random-placement control"),
            ("head@0.15", "oracle@0.15", "headroom left at this window size")]:
        d = 100 * (st.mean(acc[hi]) - st.mean(acc[lo]))
        l, h = boot(acc[hi], acc[lo])
        flag = " [VOID: budget]" if (lo in void or hi in void) else ""
        print(f"  {hi:12} - {lo:12} {d:+6.1f}pp  CI[{l:+.1f},{h:+.1f}]  {why}{flag}")

    print("\n=== INTERNAL CONTROL: items where head and argmax chose the SAME cell ===")
    # Compare the CELLS, not their coverage values. Most cells have coverage exactly 0.0, so
    # `head_cov == argmax_cov` would lump every both-missed item into "identical proposal" and
    # make this control meaningless -- it would pass by construction.
    props = json.load(open(PROP))
    def same_cell(r):
        p = props[r["question_id_full"]]
        return tuple(p["head"]) == tuple(p["argmax"])
    same = [r for r in R if same_cell(r)]
    diff = [r for r in R if not same_cell(r)]
    for nm, S in [("identical proposal", same), ("different proposal", diff)]:
        if not S:
            continue
        h = [hit(r, "head@0.15") for r in S]
        a = [hit(r, "argmax@0.15") for r in S]
        l, hh = boot(h, a)
        print(f"  {nm:20} n={len(S):4d}  head-argmax {100*(st.mean(h)-st.mean(a)):+6.1f}pp "
              f"CI[{l:+.1f},{hh:+.1f}]")
    print("  (the identical-proposal split must be ~0; a non-zero value voids the headline)")

    print("\n=== does the gain track COVERAGE, as SS6D says it must? ===")
    for nm, sel in [("head covers, argmax missed", lambda r: r["head_cov"] >= .5 > r["argmax_cov"]),
                    ("argmax covers, head missed", lambda r: r["argmax_cov"] >= .5 > r["head_cov"]),
                    ("both cover", lambda r: min(r["head_cov"], r["argmax_cov"]) >= .5),
                    ("neither covers", lambda r: max(r["head_cov"], r["argmax_cov"]) < .5)]:
        S = [r for r in R if sel(r)]
        if not S:
            continue
        h = [hit(r, "head@0.15") for r in S]
        a = [hit(r, "argmax@0.15") for r in S]
        print(f"  {nm:28} n={len(S):4d}  head {100*st.mean(h):5.1f}%  "
              f"argmax {100*st.mean(a):5.1f}%  {100*(st.mean(h)-st.mean(a)):+6.1f}pp")

    print("\n=== BY CATEGORY (V*Bench is ordered by category: a PARTIAL run is a biased subset) ===")
    cats = sorted({r["category"] for r in R})
    for c in cats:
        S = [r for r in R if r["category"] == c]
        h = [hit(r, "head@0.15") for r in S]
        a = [hit(r, "argmax@0.15") for r in S]
        b = [hit(r, "uniform@600") for r in S]
        l, hh = boot(h, b)
        print(f"  {c:20} n={len(S):4d}  head {100*st.mean(h):5.1f}%  argmax {100*st.mean(a):5.1f}%"
              f"  bar {100*st.mean(b):5.1f}%  head-bar {100*(st.mean(h)-st.mean(b)):+6.1f}pp"
              f" CI[{l:+.1f},{hh:+.1f}]")
    if len(cats) < 2:
        print("  !! ONLY ONE CATEGORY PRESENT -- this run is INCOMPLETE and the pooled numbers")
        print("     above are NOT the benchmark result. SS7 says allocation pays on single-region")
        print("     items and fails on relational ones; reporting now would report the easy half.")

    print("\n" + "=" * 72)
    hm = st.mean(acc["head@0.15"])
    am = st.mean(acc["argmax@0.15"])
    bm = st.mean(acc["uniform@600"])
    dm = 100 * (hm - am)
    lo, hi = boot(acc["head@0.15"], acc["uniform@600"])
    print(f"VERDICT  head {100*hm:.1f}%  argmax {100*am:.1f}%  bar(uniform@600) {100*bm:.1f}%")
    print(f"  head - argmax = {dm:+.1f}pp   (predicted {PREDICTED:+.1f}pp)")
    print(f"  head - bar    = {100*(hm-bm):+.1f}pp  CI[{lo:+.1f},{hi:+.1f}]")
    if hm > bm and lo > 0:
        print("  => A METHOD. Allocation beats spending the same budget -- which nothing in SS2 did.")
    elif dm > 2 and hm > bm:
        print("  => BEATS THE BAR but the CI includes zero. Report the margin with its lower bound.")
    elif dm > 2:
        print("  => THE HEAD CONVERTS but allocation still loses to the budget axis. Component, not")
        print("     a method. Report it as such.")
    else:
        print("  => DOES NOT CONVERT. Coverage is not the mediator SS6D's strata imply, or the")
        print("     window size bounds the gain. Either way, report the negative.")


if __name__ == "__main__":
    main()
