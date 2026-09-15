"""
Phase 39: four bookkeeping/strength items that decide how §3 and §5 may be stated. Offline.

  A. ARM MISMATCH (blocking). §5.2 headlines TEXT+CONF (68.6%, +12.0pp); §5.3's "deployed gate" was
     conf-only (67.0%, +10.5pp). The "83% of ceiling" figure divided one arm by the other's ceiling.
     Recomputed here against the SAME arm. This can INVERT the closing claim: if the headlined arm
     reaches ~95% of the coverage ceiling, the remaining headroom is in the PROPOSAL, not detection.

  B. EXOGENOUS COVERAGE (the instrument Phase 38 wanted). phase32 drew its random centres from a
     single seeded stream, `random.Random(32)`, two draws per item, in dataset order, with every
     skip occurring BEFORE the draw. So the centres are reconstructible and coverage under RANDOM
     placement is assigned by the generator, not by the model -- exogenous by construction.

     ONE-SIDED BY DESIGN: if phase32 was resumed, a fresh replay desyncs and pairs wrong centres
     with wrong items. A desynced replay CANNOT manufacture a dose-response -- it would be pure
     noise. So a POSITIVE result is valid evidence; a NULL is inconclusive between "replay wrong"
     and "hypothesis wrong", and is reported as inconclusive rather than as a refutation.
     A replay-integrity check is run first and printed either way.

  C. THE W SWEEP HAS NO CIs. §3.4(b) is the one genuinely interventional leg (we set W; the model
     does not), and "interior optimum" currently rests on four unbootstrapped points.

  D. WHAT IS IN THE 0% BIN? n=96 is the bulk of the negative mass, but "coverage exactly 0"
     conflates a one-pixel near-miss with a wrong-side-of-the-image miss: the median box is 0.06% of
     the image and the window 2.25%. Near-misses point at a larger/box-aware window; far misses
     point at multi-window search. This decides which GPU run is worth doing.
"""
import json
import random
import statistics as st

VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
BOX = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"
REL = ("left", "right", "above", "below", "under", "over", "behind",
       "front", "side", "between", "beneath", "underneath", "top of", "bottom of")


def is_rel(q):
    return any(k in str(q).lower() for k in REL)


def cov(b, cx, cy, W):
    gx0, gy0, gx1, gy1 = b
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0.0, W
    if y0 < 0: y0, y1 = 0.0, W
    if x1 > 1: x0, x1 = 1 - W, 1.0
    if y1 > 1: y0, y1 = 1 - W, 1.0
    ix = max(0.0, min(gx1, x1) - max(gx0, x0))
    iy = max(0.0, min(gy1, y1) - max(gy0, y0))
    return ix * iy / max((gx1 - gx0) * (gy1 - gy0), 1e-9)


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(VS)]
    bx = {x["question_id_full"]: x["gt_box_frac"] for x in (json.loads(l) for l in open(BOX))}
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    cA = lambda r: max(r["probs"]["attn@0.15"])
    cU = lambda r: max(r["probs"]["uniform"])
    for r in rows:
        r["_cov"] = cov(bx[r["question_id_full"]], *r["peak_frac"], 0.15)

    # ---------------------------------------------------------------- A
    print("=" * 78)
    print("A. THE ARM MISMATCH: ceiling recomputed against the arm actually headlined")
    print("=" * 78)
    u = [hit(r, "uniform") for r in rows]
    conf_only = [hit(r, "attn@0.15") if cA(r) > cU(r) else hit(r, "uniform") for r in rows]
    textconf = [hit(r, "uniform") if is_rel(r["question"])
                else (hit(r, "attn@0.15") if cA(r) > cU(r) else hit(r, "uniform")) for r in rows]
    # the matching ceiling for EACH arm: route on TRUE coverage, keeping that arm's other rules
    orc_conf = [hit(r, "attn@0.15") if r["_cov"] > .001 else hit(r, "uniform") for r in rows]
    orc_tc = [hit(r, "uniform") if is_rel(r["question"])
              else (hit(r, "attn@0.15") if r["_cov"] > .001 else hit(r, "uniform")) for r in rows]
    best = [max(hit(r, "attn@0.15"), hit(r, "uniform")) for r in rows]
    ub = st.mean(u)
    print(f"  {'arm':<34}{'acc':>8}{'vs uniform':>12}{'its own coverage ceiling':>26}")
    for nm, v, c in [("conf-only gate", conf_only, orc_conf),
                     ("TEXT+CONF gate (HEADLINED)", textconf, orc_tc)]:
        cap = (st.mean(v) - ub) / max(st.mean(c) - ub, 1e-9)
        print(f"  {nm:<34}{100*st.mean(v):>7.1f}%{100*(st.mean(v)-ub):>+11.1f}pp"
              f"{100*(st.mean(c)-ub):>+16.1f}pp  = {100*cap:.0f}%")
    print(f"  {'per-item best (NOT reachable)':<34}{100*st.mean(best):>7.1f}%"
          f"{100*(st.mean(best)-ub):>+11.1f}pp")
    tc_cap = (st.mean(textconf) - ub) / max(st.mean(orc_tc) - ub, 1e-9)
    print(f"\n  => the headlined arm captures {100*tc_cap:.0f}% of its coverage ceiling.")
    print("     >=90% means the DETECTOR is essentially solved and the residual headroom is in the")
    print("     PROPOSAL (the per-item best line); <90% means detection is still the bottleneck.")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78)
    print("B. EXOGENOUS COVERAGE: replaying the seeded random centres")
    print("=" * 78)
    rng = random.Random(32)
    for r in rows:
        r["_rx"], r["_ry"] = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        r["_rcov"] = cov(bx[r["question_id_full"]], r["_rx"], r["_ry"], 0.15)
    nz = sum(1 for r in rows if r["_rcov"] > .001)
    exp = st.mean([min(1.0, (0.15 ** 2) / max((bx[r['question_id_full']][2]-bx[r['question_id_full']][0])
                  * (bx[r['question_id_full']][3]-bx[r['question_id_full']][1]), 1e-9)) and
                  min(1.0, 0.15 ** 2 / 0.64) for r in rows])
    print(f"  replay integrity: {nz}/{len(rows)} random windows overlap the box "
          f"({100*nz/len(rows):.1f}%); a W=0.15 window is 2.25% of the image, and centres are")
    print(f"  drawn in [0.1,0.9]^2, so a few percent is the expected rate for tiny boxes.")
    print(f"\n  {'random-placement coverage':<28}{'n':>4}{'uniform':>9}{'rand@0.15':>11}"
          f"{'delta':>8}{'95% CI':>18}")
    for lo_, hi_, nm in [(-.001, .001, "0% (missed)"), (.001, 1.01, "hit (>0)")]:
        g = [r for r in rows if lo_ < r["_rcov"] <= hi_]
        if len(g) < 8:
            print(f"  {nm:<28}{len(g):>4}   too few -- INCONCLUSIVE")
            continue
        uu = [hit(r, "uniform") for r in g]
        rr = [hit(r, "rand@0.15") for r in g]
        d = [x - y for x, y in zip(rr, uu)]
        l, h = boot(d)
        print(f"  {nm:<28}{len(g):>4}{100*st.mean(uu):>8.1f}%{100*st.mean(rr):>10.1f}%"
              f"{100*st.mean(d):>+8.1f}   [{100*l:+.1f},{100*h:+.1f}]")
    print("\n  ONE-SIDED: a positive separation here is valid causal evidence (a desynced replay")
    print("  could not produce one). A null is INCONCLUSIVE, not a refutation.")

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78)
    print("C. THE W SWEEP WITH CIs -- the one interventional leg of the causal argument")
    print("=" * 78)
    Ws = ["0.15", "0.25", "0.35", "0.5"]
    print(f"  {'W':<8}{'attn delta':>12}{'95% CI':>20}{'mean coverage':>16}")
    ds = {}
    for w in Ws:
        d = [hit(r, f"attn@{w}") - hit(r, "uniform") for r in rows]
        ds[w] = d
        l, h = boot(d)
        c = st.mean([cov(bx[r["question_id_full"]], *r["peak_frac"], float(w)) for r in rows])
        print(f"  {w:<8}{100*st.mean(d):>+11.1f}{'':>3}[{100*l:+.1f},{100*h:+.1f}]{100*c:>15.1f}%")
    print(f"\n  does the W=0.25 peak SEPARATE from its neighbours? (paired, same items)")
    for a, b in [("0.25", "0.15"), ("0.25", "0.35"), ("0.25", "0.5")]:
        d = [x - y for x, y in zip(ds[a], ds[b])]
        l, h = boot(d)
        sig = "SEPARATES" if (l > 0 or h < 0) else "n.s."
        print(f"    W={a} vs W={b}: {100*st.mean(d):+.1f}pp  CI[{100*l:+.1f},{100*h:+.1f}]  {sig}")

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 78)
    print("D. WHAT IS IN THE 0%-COVERAGE BIN? near-miss or far-miss decides the next GPU run")
    print("=" * 78)
    miss = [r for r in rows if r["_cov"] <= .001]
    gap = []
    for r in miss:
        gx0, gy0, gx1, gy1 = bx[r["question_id_full"]]
        cx, cy = r["peak_frac"]
        dx = max(gx0 - (cx + .075), (cx - .075) - gx1, 0.0)
        dy = max(gy0 - (cy + .075), (cy - .075) - gy1, 0.0)
        gap.append(max(dx, dy))
    gap.sort()
    print(f"  n = {len(miss)} missed items; GAP = distance from the window EDGE to the box")
    for q, nm in [(.25, "p25"), (.5, "median"), (.75, "p75"), (.9, "p90")]:
        print(f"    {nm:<8}{gap[int(q*(len(gap)-1))]:.3f} of image width")
    for t, nm in [(.05, "within 0.05 (one third of a window)"),
                  (.10, "within 0.10 (two thirds of a window)"),
                  (.20, "within 0.20")]:
        print(f"    {nm:<40}{100*sum(1 for g in gap if g <= t)/len(gap):5.1f}%")
    W35 = sum(1 for r in miss if cov(bx[r["question_id_full"]], *r["peak_frac"], 0.35) > .001)
    print(f"\n  of these 96, a W=0.35 window at the SAME centre would reach {W35} "
          f"({100*W35/len(miss):.0f}%)")
    print("  -> mostly NEAR misses argue for a larger / box-aware window (and match the W sweep);")
    print("     mostly FAR misses argue for multi-window search. This picks the next experiment.")


if __name__ == "__main__":
    main()
