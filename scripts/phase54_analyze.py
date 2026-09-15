"""
Phase 54 analyzer: does the scene+crop COMPOSITE beat the budget axis at 4K?

The crop-only policy loses by 17.3pp at 4K because a W=0.15 window keeps 2.25% of a 4032^2 image and
throws the rest away. The composite passes the SCENE and the CROP together as two images in ONE
forward pass, so a bad crop costs resolution instead of the whole question.

Each composite arm is charged for the localiser pass AND the composite pass. The localiser pass
also produces the uniform@300 answer, so the marginal cost of the composite is only its own tokens;
the full 300 is still charged, which is the conservative reading.

The bar is the uniform sweep measured IN THIS RUN on THESE items, log-interpolated, extended past
the last anchor with the last measured slope (marked EXT).

The peak gate is applied OFFLINE at the V*Bench-TRANSFERRED firing rate, so the adaptive variant of
each composite pays its cost only when it fires. Refitting the rate on HR-Bench would not be a
transfer result and is shown only as a bound.
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase54_composite_4k.jsonl"
FIRE = 0.40
COMPS = ["comp_150_150_w15", "comp_300_300_w15", "comp_300_300_w35", "comp_150_450_w15"]


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 100:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    arms = [a for a in rows[0]["probs"]]
    anchors = sorted((st.mean(r["cost"][a]["tokens"] for r in rows),
                      st.mean(hit(r, a) for r in rows))
                     for a in ("uniform@300", "uniform@600", "uniform@1200"))

    def bar(t):
        if t <= anchors[0][0]:
            return anchors[0][1]
        for (t0, a0), (t1, a1) in zip(anchors, anchors[1:]):
            if t <= t1:
                w = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
                return a0 + w * (a1 - a0)
        (t0, a0), (t1, a1) = anchors[-2], anchors[-1]
        sl = (a1 - a0) / (math.log(t1) - math.log(t0))
        return a1 + sl * (math.log(t) - math.log(t1))

    print(f"n = {n} rows / {len({r['instance'] for r in rows})} instances, HR-Bench 4k")
    print("Bar (in-run): " + ", ".join(f"{t:.0f}tok->{100*a:.1f}%" for t, a in anchors))
    print("Reference from Phase 53, same benchmark: crop-only ours_attn@0.15 = 42.6%, "
          "-17.3pp vs its bar.\n")

    print("=== ALWAYS-ON COMPOSITE (pays on every item) ===")
    print(f"  {'arm':<20}{'psses':>6}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    res = {}
    for a in arms:
        v = [hit(r, a) for r in rows]
        ps = st.mean(r["cost"][a]["passes"] for r in rows)
        tk = st.mean(r["cost"][a]["tokens"] for r in rows)
        b = bar(tk)
        lo, hi = boot([x - b for x in v])
        res[a] = (st.mean(v), ps, tk, b, st.mean(v) - b, lo, hi)
        ext = "  EXT" if tk > anchors[-1][0] else ""
        print(f"  {a:<20}{ps:>6.2f}{tk:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%"
              f"{100*(st.mean(v)-b):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]{ext}")

    # ---- adaptive: pay the composite only when peak clears the transferred threshold
    insts = sorted({r["instance"] for r in rows})
    pk = {r["instance"]: r["peak"] for r in rows}
    vals = sorted((pk[i] for i in insts), reverse=True)
    tau = vals[max(0, int(FIRE * len(vals)) - 1)]
    print(f"\n=== PEAK-GATED COMPOSITE (transferred firing rate {100*FIRE:.0f}%, tau={tau:.4f}) ===")
    print(f"  {'arm':<20}{'psses':>6}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    best = None
    for a in COMPS:
        if a not in rows[0]["probs"]:
            continue
        v, tks, pss = [], [], []
        for r in rows:
            if pk[r["instance"]] >= tau:
                v.append(hit(r, a)); tks.append(r["cost"][a]["tokens"])
                pss.append(r["cost"][a]["passes"])
            else:
                v.append(hit(r, "uniform@300")); tks.append(r["cost"]["uniform@300"]["tokens"])
                pss.append(r["cost"]["uniform@300"]["passes"])
        tk = st.mean(tks); b = bar(tk)
        lo, hi = boot([x - b for x in v])
        m = st.mean(v) - b
        if best is None or m > best[0]:
            best = (m, a, st.mean(v), tk, lo, hi)
        print(f"  {'gated ' + a:<20}{st.mean(pss):>6.2f}{tk:>8.0f}{100*st.mean(v):>7.1f}%"
              f"{100*b:>7.1f}%{100*m:>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== CIRCULAR EVAL (the benchmark's own metric) ===")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["instance"]].append(i)
    full = [k for k, v in by.items() if len(v) >= 4]
    circ = {a: st.mean([min(hit(rows[i], a) for i in by[k]) for k in full]) for a in arms}
    for a in arms:
        print(f"  {a:<20}{100*circ[a]:>7.1f}%   vs uniform@1200 "
              f"{100*(circ[a]-circ['uniform@1200']):+.1f}pp")

    print("\n=== by category (per-row) ===")
    for cat in sorted({r["category"] for r in rows}):
        g = [r for r in rows if r["category"] == cat]
        print(f"  {cat:<8}n={len(g):<4}" + "  ".join(
            f"{a[:12]} {100*st.mean(hit(r,a) for r in g):.1f}%" for a in arms))

    print("\n" + "=" * 74)
    win = [a for a in arms if a not in ("uniform@300", "uniform@600", "uniform@1200")
           and res[a][4] > 0]
    sig = [a for a in win if res[a][5] > 0]
    if sig:
        print(f"  => {', '.join(sig)} BEAT the budget axis at 4K, significantly.")
    elif best and best[0] > 0 and best[4] > 0:
        print(f"  => the GATED composite {best[1]} beats the bar at 4K: {100*best[0]:+.1f}pp "
              f"CI[{100*best[4]:+.1f},{100*best[5]:+.1f}] at {best[3]:.0f} tokens.")
    elif win or (best and best[0] > 0):
        print(f"  => positive margins appear (best {100*best[0]:+.1f}pp for {best[1]}) but none is")
        print(f"     significant at n={n}. Report as directional, not as a win.")
    else:
        print("  => THE COMPOSITE ALSO LOSES AT 4K. No allocation policy we have beats spending the")
        print("     budget there; the method is V*Bench-only and the paper must say so plainly.")


if __name__ == "__main__":
    main()
