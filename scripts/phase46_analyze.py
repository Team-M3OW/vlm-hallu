"""
Phase 46 analyzer: does ADAPTIVE allocation rescue the HR-Bench transfer result?

THE TEST
--------
On HR-Bench the deployed always-2-pass gate loses to its own compute-matched control:
    uniform@600  59.2%   |  conf_route 56.1% (-3.1pp)  |  TEXT+CONF 53.4% (-5.9pp)

The adaptive policy pays the second pass only when pass-1 `peak` clears a threshold, so its cost --
and therefore its bar -- moves with the firing rate. The uniform sweep measured on THIS benchmark
supplies the bar: uniform@300 = 52.5%, uniform@600 = 59.2%.

    adaptive beats its bar at some cost   -> the compute fix rescues the transfer result; the
                                             method wins on both benchmarks and §5 can claim it.
    adaptive loses at every cost          -> the fix does not rescue HR-Bench. Report it; the
                                             method remains V*Bench-only and the paper says so.

THE THRESHOLD IS TRANSFERRED, NOT REFIT. tau comes from the V*Bench firing rate that was best
there. Refitting tau on HR-Bench would answer a weaker question, so the transferred value is the
headline and the HR-Bench-optimal value is shown only as a bound.

RANDOM ROUTING AT THE SAME COST IS THE CONTROL THAT MATTERS. Any partial firing earns a margin
purely from the concavity of the uniform curve, so a policy is only doing work if it beats random
routing at the same cost.
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

HR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33_hrbench_transfer.jsonl"
PK = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase46_hrbench_peak.jsonl"
SWEEP = [(300, 0.525), (600, 0.592)]      # measured on HR-Bench in phase33
REL = ("left", "right", "above", "below", "under", "over", "behind",
       "front", "side", "between", "beneath", "underneath", "top of", "bottom of")


def bar(tok):
    if tok <= SWEEP[0][0]:
        return SWEEP[0][1]
    if tok >= SWEEP[-1][0]:
        return SWEEP[-1][1]
    (t0, a0), (t1, a1) = SWEEP[0], SWEEP[1]
    w = (math.log(tok) - math.log(t0)) / (math.log(t1) - math.log(t0))
    return a0 + w * (a1 - a0)


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(HR)]
    pk = {json.loads(l)["instance"]: json.loads(l) for l in open(PK)}
    rows = [r for r in rows if r["instance"] in pk]
    for r in rows:
        r["_peak"] = pk[r["instance"]]["peak"]
    n = len(rows)
    print(f"n = {n} rows over {len({r['instance'] for r in rows})} instances")
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    cA = lambda r: max(r["probs"]["attn@0.15"])
    cU = lambda r: max(r["probs"]["uniform"])
    is_rel = lambda q: any(k in str(q).lower() for k in REL)

    u = [hit(r, "uniform") for r in rows]
    u2 = [hit(r, "uniform2x") for r in rows]
    conf = [hit(r, "attn@0.15") if cA(r) > cU(r) else hit(r, "uniform") for r in rows]
    tc = [hit(r, "uniform") if is_rel(r["question"])
          else (hit(r, "attn@0.15") if cA(r) > cU(r) else hit(r, "uniform")) for r in rows]
    print("\n=== THE PROBLEM, restated ===")
    print(f"  uniform@300 {100*st.mean(u):.1f}%   uniform@600 (compute-matched) {100*st.mean(u2):.1f}%")
    for nm, v in [("conf_route (2.00x)", conf), ("TEXT+CONF gate (2.00x)", tc)]:
        print(f"  {nm:<26}{100*st.mean(v):.1f}%   vs its bar {100*(st.mean(v)-st.mean(u2)):+.1f}pp")

    # policy applied per INSTANCE (the proposal is per instance, not per row)
    insts = sorted({r["instance"] for r in rows})
    peak_of = {i: pk[i]["peak"] for i in insts}

    def run(fire, signal):
        vals = sorted((signal(i) for i in insts), reverse=True)
        k = max(0, min(len(vals) - 1, int(fire * len(vals)) - 1))
        t = vals[k] if fire > 0 else float("inf")
        acc, cost = [], []
        for r in rows:
            if fire > 0 and signal(r["instance"]) >= t:
                acc.append(hit(r, "attn@0.15") if cA(r) > cU(r) else hit(r, "uniform"))
                cost.append(2.0)
            else:
                acc.append(hit(r, "uniform"))
                cost.append(1.0)
        return acc, cost

    print("\n=== COST-ACCURACY CURVE (bar from HR-Bench's OWN measured sweep) ===")
    print(f"  {'fire':<7}{'cost':>7}{'tokens':>8}{'adaptive':>10}{'bar':>8}{'margin':>9}"
          f"{'95% CI':>17}{'random@same cost':>19}")
    best = None
    for fr in (0.0, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0):
        a, c = run(fr, lambda i: peak_of[i])
        cc = st.mean(c)
        b = bar(300 * cc)
        m = st.mean(a) - b
        lo, hi = boot([x - b for x in a])
        ar, cr = run(fr, lambda i: random.Random(5000 + i).random())
        mr = st.mean(ar) - bar(300 * st.mean(cr))
        if best is None or m > best[0]:
            best = (m, fr, cc, st.mean(a), b)
        print(f"  {100*fr:>5.0f}%{cc:>6.2f}x{300*cc:>8.0f}{100*st.mean(a):>9.1f}%{100*b:>7.1f}%"
              f"{100*m:>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]{100*mr:>+18.1f}")

    print(f"\n  best: {100*best[0]:+.1f}pp over the bar at {100*best[1]:.0f}% firing "
          f"({best[2]:.2f}x cost, {100*best[3]:.1f}% vs bar {100*best[4]:.1f}%)")

    print("\n=== TRANSFERRED THRESHOLD (V*Bench's best firing rate was 40%) ===")
    a, c = run(0.40, lambda i: peak_of[i])
    cc = st.mean(c)
    b = bar(300 * cc)
    lo, hi = boot([x - b for x in a])
    print(f"  adaptive@40%  {100*st.mean(a):.1f}%  cost {cc:.2f}x  bar {100*b:.1f}%  "
          f"margin {100*(st.mean(a)-b):+.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]")
    print(f"  the always-2-pass gate's margin for comparison: "
          f"{100*(st.mean(conf)-st.mean(u2)):+.1f}pp at 2.00x")

    print("\n=== CIRCULAR EVAL (HR-Bench's intended metric) ===")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["instance"]].append(i)
    full = [k for k, v in by.items() if len(v) >= 4]
    circ = lambda v: st.mean([min(v[i] for i in by[k]) for k in full])
    a40, _ = run(0.40, lambda i: peak_of[i])
    print(f"  uniform@300 {100*circ(u):.1f}%   uniform@600 {100*circ(u2):.1f}%"
          f"   conf_route {100*circ(conf):.1f}%   adaptive@40% {100*circ(a40):.1f}%")

    print("\n" + "=" * 72)
    print("VERDICT -- the HEADLINE is the TRANSFERRED threshold, not the best one")
    print("=" * 72)
    a40, c40 = run(0.40, lambda i: peak_of[i])
    m40 = st.mean(a40) - bar(300 * st.mean(c40))
    print(f"  always-2-pass (deployed)          {100*(st.mean(conf)-st.mean(u2)):+.1f}pp   at 2.00x")
    print(f"  adaptive @ V*Bench-transferred 40% {100*m40:+.1f}pp   at {st.mean(c40):.2f}x")
    print(f"  adaptive @ HR-Bench-best 20%       {100*best[0]:+.1f}pp   at {best[2]:.2f}x"
          f"   <- TUNED ON THIS BENCHMARK, not a transfer result")
    print()
    if m40 > 0:
        print("  => the TRANSFERRED policy beats its bar. Genuine rescue.")
    elif m40 > -1.5:
        print("  => the transferred policy is at BREAK-EVEN, up from a clear loss. The adaptive fix")
        print("     removes the transfer failure but does NOT turn it into a win. The +1.9pp figure")
        print("     requires choosing the firing rate ON HR-Bench and is reported as a bound only.")
    else:
        print("  => ADAPTIVE DOES NOT RESCUE HR-BENCH. The method stays V*Bench-only; report it.")
    print()
    print("  note: on V*Bench the margin is positive at EVERY firing rate tested (+1.1 to +5.2pp),")
    print("  so the conclusion there does not depend on the rate. On HR-Bench it is positive only")
    print("  at 10-30% and negative from 40% up, so here the rate choice does matter.")


if __name__ == "__main__":
    main()
