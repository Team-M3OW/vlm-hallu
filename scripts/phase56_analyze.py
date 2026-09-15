"""
Phase 56 analyzer: does localising TWICE at increasing resolution make the method win at 4K?

Arms, with every pass charged:
    uniform@300 / @600 / @1200   the bar, measured on these items (600/1200 joined from phase53)
    mid@0.35                     answer from the mid-level crop around the COARSE peak
                                 (cost: coarse localise + this answer)
    coarse_fine@0.15             tight window around the COARSE peak -- the deployed policy
    c2f_fine@0.15                tight window around the FINE peak, found by localising inside the
                                 mid crop (cost: coarse localise + fine localise + this answer)

Plus peak-gated variants at the V*Bench-transferred firing rate, and a confidence route over the
passes each policy has already paid for (answers are free once the pass is run).

The comparison that isolates the contribution is `c2f_fine@0.15` vs `coarse_fine@0.15`: same window
size, same budget for the final answer, differing ONLY in whether the centre came from one
localisation or two. If refinement does not move accuracy, the attention map at 4K does not carry
usable localisation at any resolution we can afford, and the paper says so.
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase56_coarse_to_fine.jsonl"
FIRE = 0.40


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    rows = [r for r in rows if "uniform@1200" in r["probs"]]
    if len(rows) < 100:
        print(f"only {len(rows)} usable rows; wait")
        return
    n = len(rows)
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    conf = lambda r, a: max(r["probs"][a])
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
        return a1 + (a1 - a0) / (math.log(t1) - math.log(t0)) * (math.log(t) - math.log(t1))

    print(f"n = {n} rows / {len({r['instance'] for r in rows})} instances, HR-Bench 4k")
    print("  bar (in-run): " + ", ".join(f"{t:.0f}tok->{100*a:.1f}%" for t, a in anchors))
    print(f"  localiser cell size: coarse {st.median([r['cell_px_coarse'] for r in rows]):.0f}px"
          f" -> fine {st.median([r['cell_px_fine'] for r in rows]):.0f}px "
          f"({st.median([r['cell_px_coarse']/r['cell_px_fine'] for r in rows]):.1f}x finer)")
    d = [math.hypot(r["coarse_xy"][0] - r["fine_xy"][0], r["coarse_xy"][1] - r["fine_xy"][1])
         for r in rows]
    print(f"  refinement moves the peak by a median {st.median(d):.3f} of image width"
          f" (~{st.median(d)*st.median([r['img_wh'][0] for r in rows]):.0f}px)\n")

    arms = [a for a in ("uniform@300", "uniform@600", "uniform@1200", "mid@0.35",
                        "coarse_fine@0.15", "c2f_fine@0.15") if a in rows[0]["probs"]]
    print(f"  {'arm':<20}{'psses':>6}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    res = {}
    for a in arms:
        v = [hit(r, a) for r in rows]
        ps = st.mean(r["cost"][a]["passes"] for r in rows)
        tk = st.mean(r["cost"][a]["tokens"] for r in rows)
        b = bar(tk)
        lo, hi = boot([x - b for x in v])
        res[a] = (st.mean(v), tk, b, st.mean(v) - b, lo, hi)
        print(f"  {a:<20}{ps:>6.2f}{tk:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%"
              f"{100*(st.mean(v)-b):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== THE ISOLATING CONTRAST: same window, one localisation vs two ===")
    dd = [hit(r, "c2f_fine@0.15") - hit(r, "coarse_fine@0.15") for r in rows]
    lo, hi = boot(dd)
    print(f"  c2f_fine - coarse_fine = {100*st.mean(dd):+.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  "
          f"{'SIG' if (lo>0 or hi<0) else 'n.s.'}")

    print("\n=== GATED + CONFIDENCE-ROUTED POLICIES (transferred firing rate) ===")
    insts = sorted({r["instance"] for r in rows})
    pk = {r["instance"]: r["peak_coarse"] for r in rows}
    vals = sorted(pk.values(), reverse=True)
    tau = vals[max(0, int(FIRE * len(vals)) - 1)]
    print(f"  {'policy':<30}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    best = None
    POLICIES = {
        "gated mid@0.35": ["mid@0.35"],
        "gated coarse_fine": ["coarse_fine@0.15"],
        "gated c2f_fine": ["c2f_fine@0.15"],
        "gated {mid, c2f_fine}": ["mid@0.35", "c2f_fine@0.15"],
    }
    for nm, cand in POLICIES.items():
        v, tks = [], []
        for r in rows:
            if pk[r["instance"]] >= tau:
                pool = ["uniform@300"] + cand
                pick = max(pool, key=lambda a: conf(r, a))
                v.append(hit(r, pick))
                tks.append(max(r["cost"][a]["tokens"] for a in cand))
            else:
                v.append(hit(r, "uniform@300"))
                tks.append(r["cost"]["uniform@300"]["tokens"])
        tk = st.mean(tks); b = bar(tk)
        lo, hi = boot([x - b for x in v])
        m = st.mean(v) - b
        if best is None or m > best[0]:
            best = (m, nm, st.mean(v), tk, lo, hi)
        print(f"  {nm:<30}{tk:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%{100*m:>+8.1f}"
              f"   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== CIRCULAR EVAL ===")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["instance"]].append(i)
    full = [k for k, v in by.items() if len(v) >= 4]
    for a in arms:
        c = st.mean([min(hit(rows[i], a) for i in by[k]) for k in full])
        print(f"  {a:<20}{100*c:>7.1f}%")

    print("\n" + "=" * 74)
    if best and best[0] > 0 and best[4] > 0:
        print(f"  => {best[1]} BEATS the budget axis at 4K: {100*best[0]:+.1f}pp "
              f"CI[{100*best[4]:+.1f},{100*best[5]:+.1f}] at {best[3]:.0f} tokens. SIGNIFICANT.")
    elif best and best[0] > 0:
        print(f"  => best is {best[1]} at {100*best[0]:+.1f}pp CI[{100*best[4]:+.1f},"
              f"{100*best[5]:+.1f}] -- positive but NOT significant. Directional only.")
    else:
        print("  => COARSE-TO-FINE DOES NOT MAKE THE METHOD WIN AT 4K. Refining the peak is not the")
        print("     missing piece; the attention map does not carry usable localisation at 4K at any")
        print("     resolution we can afford. The method is V*Bench-only and the paper says so.")


if __name__ == "__main__":
    main()
