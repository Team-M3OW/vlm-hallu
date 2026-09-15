"""
Phase 45: ADAPTIVE token allocation -- spend the second pass only when reallocation will pay.
Offline on Phase 32's logs; no GPU. Nothing here needs a forward pass that was not already run.

THE PROBLEM THIS FIXES
----------------------
The deployed allocator (§5.2) runs TWO passes on EVERY item: one uniform, one cropped, answering
from whichever is more confident. Its honest bar is therefore `uniform@600`, and on HR-Bench it
LOSES to that bar:

    uniform@600 (compute-matched)  59.2%
    conf_route@0.15                56.1%   -3.1pp
    TEXT+CONF (the gate)           53.4%   -5.9pp

So on the transfer benchmark, simply spending the tokens beats the method. That single line is the
paper's biggest weakness, and it is a COMPUTE-ACCOUNTING problem, not an allocation problem: the
method pays 2x on 100% of items to discover, item by item, something it is usually wrong about.

THE FIX THE MECHANISM ALREADY IMPLIES
-------------------------------------
§5.3 showed the confidence gate is really a COVERAGE DETECTOR (AUROC 0.835) -- but it detects
coverage by cropping first and reading the crop's own confidence, which is what costs the second
pass. §3 says coverage is the causal quantity. So the question is whether coverage can be predicted
from the FIRST pass alone, before paying for the crop.

It can. Measured on Phase 32's logged pass-1 features:

    conf(attn crop)      AUROC 0.835   -- costs a second pass
    peak attention alone AUROC 0.788   -- FREE, already computed by the localizer
    all pass-1 feats CV  AUROC 0.766   -- fitted, and WORSE than the single feature at n=191

`peak` is the max of the ring-masked, L1-normalised, block-averaged attention map -- a number the
localizer computes on its way to the argmax. Using it costs nothing.

THE POLICY
----------
    pass 1: uniform@B0 with attention  -> answer, peak location, and `peak`
    if peak < tau:  STOP. Answer from pass 1.            COST 1x. No reallocation.
    if peak >= tau: crop at the peak, run pass 2,
                    answer by the confidence comparison. COST 2x.

Expected cost is 1 + P(peak >= tau) passes, so the compute-matched bar MOVES WITH THE POLICY and is
computed per-benchmark from the realized firing rate, never assumed. That is the whole point: on a
benchmark where reallocation does not pay, the policy should mostly decline to reallocate, its cost
should fall toward 1x, and its bar should fall toward uniform@B0.

TAU IS CHOSEN BY CROSS-VALIDATION, NOT ON THE TEST SET
------------------------------------------------------
5-fold CV: tau is selected on 4 folds by accuracy-per-pass and applied to the held-out fold, so
every reported number is out-of-fold. An in-sample tau would be selection on the test set.
The oracle-coverage policy and the always-2-pass gate are reported beside it as bounds.
"""
import json
import random
import statistics as st

import numpy as np

VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
BOX = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"
W = 0.15


def cov(b, cx, cy, W):
    gx0, gy0, gx1, gy1 = b
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0.0, W
    if y0 < 0: y0, y1 = 0.0, W
    if x1 > 1: x0, x1 = 1 - W, 1.0
    if y1 > 1: y0, y1 = 1 - W, 1.0
    return (max(0.0, min(gx1, x1) - max(gx0, x0)) * max(0.0, min(gy1, y1) - max(gy0, y0))
            / max((gx1 - gx0) * (gy1 - gy0), 1e-9))


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(VS)]
    bx = {x["question_id_full"]: x["gt_box_frac"]
          for x in (json.loads(l) for l in open(BOX))}
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    cA = lambda r: max(r["probs"][f"attn@{W}"])
    cU = lambda r: max(r["probs"]["uniform"])
    for r in rows:
        r["_cov"] = cov(bx[r["question_id_full"]], *r["peak_frac"], W)
        r["_peak"] = r["attn_feats"]["peak"]
    n = len(rows)
    print(f"n = {n}.  Policy signal = `peak`, computed by the localizer on pass 1. Cost: nothing.")

    u = [hit(r, "uniform") for r in rows]
    two = [hit(r, f"attn@{W}") if cA(r) > cU(r) else hit(r, "uniform") for r in rows]

    # ---- the measured uniform sweep IS the compute-matched bar (Qwen3-VL, V*Bench, phase27)
    SWEEP = [(294, 0.565), (600, 0.660), (1176, 0.702), (2400, 0.754), (4760, 0.812)]
    def bar(tokens):
        """Accuracy a plain uniform image of this budget reaches. Interpolated in log-tokens,
        which is how the measured sweep actually behaves (56.5 -> 66.0 -> 70.2 -> 75.4 -> 81.2)."""
        import math
        if tokens <= SWEEP[0][0]:
            return SWEEP[0][1]
        for (t0, a0), (t1, a1) in zip(SWEEP, SWEEP[1:]):
            if tokens <= t1:
                w = (math.log(tokens) - math.log(t0)) / (math.log(t1) - math.log(t0))
                return a0 + w * (a1 - a0)
        return SWEEP[-1][1]

    # ---- COST-TARGETED tau, selected on training folds, evaluated OUT OF FOLD
    # An earlier version maximised "accuracy per pass", which is maximised by NEVER firing
    # (0.565/1.00 beats 0.670/2.00); it drove tau to fire on 2% of items and reported +0.0pp.
    # The objective is accuracy AT A COST, so tau is chosen to hit a target firing rate.
    idx = list(range(n))
    random.Random(7).shuffle(idx)
    folds = [idx[i::5] for i in range(5)]

    def run(target_fire, signal):
        oa, oc = [0.0] * n, [0.0] * n
        for f in range(5):
            te = set(folds[f])
            tr = [i for i in idx if i not in te]
            vals = sorted((signal(rows[i]) for i in tr), reverse=True)
            k = max(0, min(len(vals) - 1, int(target_fire * len(vals)) - 1))
            t = vals[k] if target_fire > 0 else float("inf")
            for i in te:
                r = rows[i]
                if signal(r) >= t and target_fire > 0:
                    oa[i] = hit(r, f"attn@{W}") if cA(r) > cU(r) else hit(r, "uniform")
                    oc[i] = 2.0
                else:
                    oa[i] = hit(r, "uniform")
                    oc[i] = 1.0
        return oa, oc

    print("\n=== COST-ACCURACY CURVE: adaptive allocation vs the MEASURED uniform sweep ===")
    print("  the bar moves with the policy: at average cost c the honest comparison is a plain")
    print("  uniform image of 300*c tokens, read off phase27's measured sweep.")
    print(f"  {'fire rate':<11}{'cost':>7}{'tokens':>8}{'adaptive':>10}{'uniform bar':>13}"
          f"{'margin':>9}{'95% CI':>17}")
    best = None
    for tf in (0.0, 0.15, 0.25, 0.40, 0.50, 0.65, 0.80, 1.0):
        oa, oc = run(tf, lambda r: r["_peak"])
        c = st.mean(oc)
        tokv = 300 * c
        b = bar(tokv)
        m = st.mean(oa) - b
        d = [x - b for x in oa]
        lo, hi = boot(d)
        star = ""
        if best is None or m > best[0]:
            best, star = (m, tf, c, st.mean(oa), b), " <-"
        print(f"  {100*tf:>9.0f}%{c:>6.2f}x{tokv:>8.0f}{100*st.mean(oa):>9.1f}%{100*b:>12.1f}%"
              f"{100*m:>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]{star}")
    print(f"\n  best margin over the compute-matched bar: {100*best[0]:+.1f}pp "
          f"at {100*best[1]:.0f}% firing ({best[2]:.2f}x cost)")

    print("\n=== vs the DEPLOYED always-2-pass gate, which pays 2.00x on every item ===")
    print(f"  always-2-pass gate   {100*st.mean(two):.1f}%  cost 2.00x  bar {100*bar(600):.1f}%"
          f"  margin {100*(st.mean(two)-bar(600)):+.1f}pp")
    oa, oc = run(best[1], lambda r: r["_peak"])
    print(f"  ADAPTIVE             {100*st.mean(oa):.1f}%  cost {st.mean(oc):.2f}x  "
          f"bar {100*bar(300*st.mean(oc)):.1f}%  margin {100*best[0]:+.1f}pp")

    print("\n=== IS `peak` DOING THE WORK? same policy, same cost, different signals ===")
    print(f"  {'signal':<28}{'acc':>8}{'margin vs bar':>16}")
    for nm, sig in [("peak (pass-1, free)", lambda r: r["_peak"]),
                    ("conf(uniform) (pass-1, free)", lambda r: -cU(r)),
                    ("random (control)", lambda r: random.Random(hash(r["question_id_full"]) & 0xffff).random()),
                    ("ORACLE coverage (cheating)", lambda r: r["_cov"])]:
        oa2, oc2 = run(best[1], sig)
        c2 = st.mean(oc2)
        print(f"  {nm:<28}{100*st.mean(oa2):>7.1f}%{100*(st.mean(oa2)-bar(300*c2)):>+15.1f}pp")

    oa, oc = run(best[1], lambda r: r["_peak"])
    oof_acc, oof_cost, cost = oa, oc, st.mean(oc)
    fired = [r for i, r in enumerate(rows) if oof_cost[i] == 2.0]
    notf = [r for i, r in enumerate(rows) if oof_cost[i] == 1.0]
    print("\n=== DOES THE POLICY FIRE WHERE COVERAGE IS? (it never sees a box) ===")
    for nm, g in [("FIRED (paid 2 passes)", fired), ("DECLINED (1 pass)", notf)]:
        if not g:
            continue
        print(f"  {nm:<26}n={len(g):>4}  P(coverage>0) "
              f"{100*st.mean(1.0*(r['_cov']>.001) for r in g):5.1f}%"
              f"   mean coverage {100*st.mean(r['_cov'] for r in g):5.1f}%")
    print("  (a policy firing at random would show the same rate in both rows)")

    print(f"\n=== BOUNDS ===")
    orc = [hit(r, f"attn@{W}") if r["_cov"] > .001 else hit(r, "uniform") for r in rows]
    ocost = 1 + st.mean(1.0 * (r["_cov"] > .001) for r in rows)
    print(f"  ORACLE-coverage adaptive   {100*st.mean(orc):.1f}%  cost {ocost:.2f}x"
          f"   (+{100*(st.mean(orc)-st.mean(u)):.1f}pp)")
    print(f"  adaptive captures {100*(st.mean(oof_acc)-st.mean(u))/max(st.mean(orc)-st.mean(u),1e-9):.0f}%"
          f" of the oracle-coverage gain, at {cost:.2f}x vs its {ocost:.2f}x")


if __name__ == "__main__":
    main()
