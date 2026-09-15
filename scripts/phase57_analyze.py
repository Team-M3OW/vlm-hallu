"""
Phase 57 analyzer: the allocation margin as a function of IMAGE SCALE, everything else held fixed.

Same 800 rows / 200 instances, same questions, same answers, at 1008 / 2016 / 4032 px. The 4032
row is joined from Phase 56 (identical items, identical model, deterministic decoding). At every
scale the bar is the uniform sweep measured AT THAT SCALE, so each margin is internally valid.

§7's prediction, fixed before the run:
    as scale falls, (a) the budget axis becomes LESS productive and (b) the localiser's grid cell
    shrinks in absolute pixels, so the proposal gets more precise. Both favour allocation, so the
    margin should RISE MONOTONICALLY as scale falls and cross zero below 4032px.
If the margin does not track scale, §7's account is wrong.

Two policies are reported at each scale:
    attn@0.15    always crop            (2 passes: localise + answer)
    GATED        crop only when `peak` clears the V*Bench-TRANSFERRED threshold (40% firing),
                 then answer from whichever of {uniform, crop} is more confident
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase57_scale_sweep.jsonl"
P56 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase56_coarse_to_fine.jsonl"
FIRE = 0.40


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def analyse(rows, scale, hit, conf):
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

    crop = "attn@0.15" if "attn@0.15" in rows[0]["probs"] else "coarse_fine@0.15"
    pk = {}
    for r in rows:
        pk.setdefault(r["instance"], r.get("peak", r.get("peak_coarse")))
    vals = sorted(pk.values(), reverse=True)
    tau = vals[max(0, int(FIRE * len(vals)) - 1)]

    out = {"scale": scale, "n": len(rows),
           "cell": st.median([r.get("cell_px", r.get("cell_px_coarse")) for r in rows]),
           "u300": anchors[0][1], "u1200": anchors[-1][1],
           "axis_gain": anchors[-1][1] - anchors[0][1]}
    v = [hit(r, crop) for r in rows]
    tk = st.mean(r["cost"][crop]["tokens"] for r in rows)
    b = bar(tk)
    lo, hi = boot([x - b for x in v])
    out["crop"] = (st.mean(v), tk, b, st.mean(v) - b, lo, hi)
    vg, tg = [], []
    for r in rows:
        if pk[r["instance"]] >= tau:
            vg.append(hit(r, crop) if conf(r, crop) > conf(r, "uniform@300") else hit(r, "uniform@300"))
            tg.append(r["cost"][crop]["tokens"])
        else:
            vg.append(hit(r, "uniform@300"))
            tg.append(r["cost"]["uniform@300"]["tokens"])
    tkg = st.mean(tg); bg = bar(tkg)
    lo2, hi2 = boot([x - bg for x in vg])
    out["gated"] = (st.mean(vg), tkg, bg, st.mean(vg) - bg, lo2, hi2)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["instance"]].append(i)
    full = [k for k, x in by.items() if len(x) >= 4]
    out["circ_u300"] = st.mean([min(hit(rows[i], "uniform@300") for i in by[k]) for k in full])
    out["circ_gated"] = st.mean([min(vg[i] for i in by[k]) for k in full])
    return out


def main():
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    conf = lambda r, a: max(r["probs"][a])
    rows = [json.loads(l) for l in open(PATH)]
    byscale = defaultdict(list)
    for r in rows:
        byscale[r["scale"]].append(r)
    try:
        n4 = [json.loads(l) for l in open(P56)]
        n4 = [r for r in n4 if "uniform@1200" in r["probs"]]
        if len(n4) >= 400:
            byscale[4032] = n4
    except FileNotFoundError:
        pass

    res = []
    for S in sorted(byscale):
        g = byscale[S]
        if len(g) < 200:
            print(f"  scale {S}: only {len(g)} rows, skipping")
            continue
        res.append(analyse(g, S, hit, conf))

    print("THE SCALE SWEEP: same 800 rows, same questions, only input resolution varies.")
    print("Bar measured at each scale from that scale's own uniform arms.\n")
    print(f"  {'scale':<8}{'n':>5}{'cell px':>9}{'u@300':>8}{'u@1200':>9}{'axis gain':>11}"
          f"{'crop margin':>14}{'GATED margin':>16}{'95% CI':>17}")
    for o in res:
        print(f"  {o['scale']:<8}{o['n']:>5}{o['cell']:>9.0f}{100*o['u300']:>7.1f}%"
              f"{100*o['u1200']:>8.1f}%{100*o['axis_gain']:>+10.1f}"
              f"{100*o['crop'][3]:>+13.1f}{100*o['gated'][3]:>+15.1f}"
              f"   [{100*o['gated'][4]:+.1f},{100*o['gated'][5]:+.1f}]")

    print("\n=== §7's PREDICTION: the margin should RISE as scale FALLS ===")
    ok_mono = all(res[i]["gated"][3] >= res[i + 1]["gated"][3] - 1e-9
                  for i in range(len(res) - 1))
    print(f"  gated margin by scale: " + "  ".join(
        f"{o['scale']}px {100*o['gated'][3]:+.1f}pp" for o in res))
    print(f"  localiser cell by scale: " + "  ".join(f"{o['scale']}px {o['cell']:.0f}px" for o in res))
    print(f"  budget-axis gain (1x->4x): " + "  ".join(
        f"{o['scale']}px {100*o['axis_gain']:+.1f}pp" for o in res))
    print(f"  -> monotone in the predicted direction: {'YES' if ok_mono else 'NO'}")

    sig = [o for o in res if o["gated"][4] > 0]
    print("\n" + "=" * 74)
    if sig:
        b = sig[0]
        print(f"  => the gated method BEATS the budget axis at {b['scale']}px: "
              f"{100*b['gated'][3]:+.1f}pp CI[{100*b['gated'][4]:+.1f},{100*b['gated'][5]:+.1f}], "
              f"n={b['n']}. SIGNIFICANT.")
        print(f"     Scales where it is significant: "
              f"{', '.join(str(o['scale'])+'px' for o in sig)}")
    else:
        best = max(res, key=lambda o: o["gated"][3])
        print(f"  => best margin {100*best['gated'][3]:+.1f}pp at {best['scale']}px, "
              f"CI[{100*best['gated'][4]:+.1f},{100*best['gated'][5]:+.1f}] -- still not significant.")
        print("     The method's own win remains unestablished at n=800; report it as directional.")

    print("\n=== STRATIFIED BY CATEGORY: is the boundary really about EVIDENCE DISPERSION? ===")
    print("  §3 says coverage governs the sign, and HR-Bench is 50% `cross` (multi-region) where a")
    print("  single crop CANNOT cover the evidence set. If allocation wins on `single` at some")
    print("  scale, the 'scale boundary' dissolves into the coverage account and §7 is unnecessary.")
    print(f"  {'scale':<8}{'category':<9}{'n':>5}{'u@300':>8}{'gated':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    for S in sorted(byscale):
        g0 = byscale[S]
        if len(g0) < 200:
            continue
        for cat in sorted({r["category"] for r in g0}):
            g = [r for r in g0 if r["category"] == cat]
            if len(g) < 50:
                continue
            o = analyse(g, S, hit, conf)
            print(f"  {S:<8}{cat:<9}{o['n']:>5}{100*o['u300']:>7.1f}%"
                  f"{100*o['gated'][0]:>7.1f}%{100*o['gated'][2]:>7.1f}%"
                  f"{100*o['gated'][3]:>+8.1f}   [{100*o['gated'][4]:+.1f},{100*o['gated'][5]:+.1f}]")

    print("\n=== CIRCULAR EVAL by scale (gated vs uniform@300) ===")
    for o in res:
        print(f"  {o['scale']}px   uniform {100*o['circ_u300']:.1f}%   "
              f"gated {100*o['circ_gated']:.1f}%   ({100*(o['circ_gated']-o['circ_u300']):+.1f}pp)")


if __name__ == "__main__":
    main()
