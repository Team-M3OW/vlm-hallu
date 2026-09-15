"""
Phase 30d: offline read-out study on the Phase 30c attention dumps. No GPU, no forward passes.

THE HYPOTHESIS BEING TESTED
---------------------------
30b established that the localizer IS in the attention map (GT-centre cell at the 16th percentile
against an exact chance of 50th) but that the ARGMAX IS NEVER THE TARGET (0/191, every layer), and
that the tokens holding the top ranks are positionally stable corner sinks absorbing 41-45% of the
selected mass. If that background is content-independent, dividing it out should promote the
target -- and the test of "promote" is not a box metric, it is whether the GT cell's RANK improves
and whether the argmax starts landing on the target.

LEAVE-ONE-OUT BY CONSTRUCTION
-----------------------------
The background for item i is estimated from every item EXCEPT i. This is not a refinement to apply
later; it is how the background is computed here, because a background fitted on all 191 items and
then evaluated on those same items would be using the test set to build its own normaliser. LOO is
free offline, so there is no reason to ever quote the leaky version.

Grids differ across items (14x21, 15x20, ...), so the background lives on a fixed RELATIVE grid and
each cell is assigned by its centre's normalised coordinates.

READ-OUTS COMPARED (all scored on the same items, same layers)
--------------------------------------------------------------
    raw        attention as-is                                     -- the 30a/30b baseline
    loo_norm   a_i / background(relative position of i)             -- the hypothesis
    loo_z      (a_i - mu_pos) / sigma_pos                           -- same idea, additive form
    no_border  raw, with the outermost ring of cells masked out     -- the crude version; if this
                                                                       matches loo_norm then the
                                                                       whole effect is just "edges"

PRIMARY OUTCOME: median gt_pct (chance 0.500) and P(argmax inside GT) (chance = GT area share,
~1/925). A read-out that moves the argmax onto the target is a localizer; one that only improves a
box statistic is not.
"""
import json
import math
import statistics as st
from collections import defaultdict

import os
ALL = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"
PATH = ALL if os.path.exists(ALL) and sum(1 for _ in open(ALL)) >= 191 else \
    "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps.jsonl"
BG_R, BG_C = 8, 8          # background resolution; coarse on purpose -- it models position, not content


def rel_cell(r, c, gh, gw):
    return (min(BG_R - 1, int(BG_R * (r + 0.5) / gh)),
            min(BG_C - 1, int(BG_C * (c + 0.5) / gw)))


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 30:
        print(f"only {len(rows)} rows; wait for more")
        return
    layers = sorted(rows[0]["attn"].keys(), key=lambda s: int(s[1:]))
    n = len(rows)
    print(f"n = {len(rows)} items, layers {layers}")
    print(f"  chance gt_pct = 0.500 exactly;  chance P(argmax in GT) = "
          f"{st.median([r['gt_area_frac'] for r in rows]):.5f} "
          f"(~1 in {1/max(st.median([r['gt_area_frac'] for r in rows]),1e-9):.0f})")
    _g = rows[0]["grid"]
    print(f"  chance P(near, 3x3 window) = {9/(_g[0]*_g[1]):.4f};  "
          f"chance median dist (uniform peak) ~ 0.38")

    results = defaultdict(dict)
    for L in layers:
        # --- accumulate the background over ALL items (per-item normalised so no image dominates)
        tot = [[0.0] * BG_C for _ in range(BG_R)]
        totsq = [[0.0] * BG_C for _ in range(BG_R)]
        cnt = [[0] * BG_C for _ in range(BG_R)]
        contrib = []            # per item, its own contribution -- needed to remove it for LOO
        for r in rows:
            gh, gw = r["grid"]
            a = r["attn"][L]
            s = sum(a) or 1.0
            cc = defaultdict(float); ccsq = defaultdict(float); nn = defaultdict(int)
            for i, v in enumerate(a):
                br, bc = rel_cell(i // gw, i % gw, gh, gw)
                p = v / s
                tot[br][bc] += p; totsq[br][bc] += p * p; cnt[br][bc] += 1
                cc[(br, bc)] += p; ccsq[(br, bc)] += p * p; nn[(br, bc)] += 1
            contrib.append((cc, ccsq, nn))

        acc = defaultdict(lambda: {"gt_pct": [], "in_gt": [], "dist": [], "near": []})
        for idx, r in enumerate(rows):
            gh, gw = r["grid"]
            a = r["attn"][L]
            s = sum(a) or 1.0
            cc, ccsq, nn = contrib[idx]
            gx0, gy0, gx1, gy1 = r["gt_box_frac"]
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            gt_cell = min(gh - 1, max(0, int(cy * gh))) * gw + min(gw - 1, max(0, int(cx * gw)))

            _bgcache = {}

            def bg_at(br, bc):
                """LOO mean and std at this relative position: this item's own contribution out."""
                if (br, bc) in _bgcache:
                    return _bgcache[(br, bc)]
                t = tot[br][bc] - cc.get((br, bc), 0.0)
                t2 = totsq[br][bc] - ccsq.get((br, bc), 0.0)
                k = cnt[br][bc] - nn.get((br, bc), 0)
                if k <= 1:
                    return 1e-9, 1e-9
                mu = t / k
                var = max(t2 / k - mu * mu, 0.0)
                _bgcache[(br, bc)] = (mu, math.sqrt(var) + 1e-12)
                return _bgcache[(br, bc)]

            scores = {}
            raw = [v / s for v in a]
            scores["raw"] = raw
            norm, z, nob = [], [], []
            for i, p in enumerate(raw):
                rr, ccol = i // gw, i % gw
                br, bc = rel_cell(rr, ccol, gh, gw)
                mu, sd = bg_at(br, bc)
                norm.append(p / max(mu, 1e-12))
                z.append((p - mu) / sd)
                border = rr == 0 or ccol == 0 or rr == gh - 1 or ccol == gw - 1
                nob.append(-1.0 if border else p)
            scores["loo_norm"] = norm
            scores["loo_z"] = z
            scores["no_border"] = nob

            for name, sc in scores.items():
                order = sorted(range(len(sc)), key=lambda i: -sc[i])
                rank = order.index(gt_cell) + 1
                acc[name]["gt_pct"].append(rank / len(sc))
                t1 = order[0]
                tx, ty = ((t1 % gw) + .5) / gw, ((t1 // gw) + .5) / gh
                inside = gx0 <= tx <= gx1 and gy0 <= ty <= gy1
                acc[name]["in_gt"].append(bool(inside))
                d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                acc[name]["dist"].append(d)
                # "near" = peak within one grid cell of the GT centre; at ~294 tokens a cell is
                # ~1/17 of the image per side, and a crop centred there still contains the target.
                acc[name]["near"].append(abs(tx - cx) <= 1.5 / gw and abs(ty - cy) <= 1.5 / gh)
        results[L] = acc

    # --- compact layer curve first: a real layer effect has NEIGHBOURS, an outlier does not.
    print("\n" + "=" * 74)
    print("LAYER CURVE on the primary metric (median gt_pct; chance 0.500; LOWER is better)")
    print("=" * 74)
    lo = min(st.median(results[L]["loo_norm"]["gt_pct"]) for L in layers)
    hi = max(st.median(results[L]["raw"]["gt_pct"]) for L in layers)
    for L in layers:
        rw = st.median(results[L]["raw"]["gt_pct"])
        nm = st.median(results[L]["loo_norm"]["gt_pct"])
        bar = "#" * max(1, int(40 * (rw - lo) / max(hi - lo, 1e-9)))
        print(f"  {L:>4} raw {rw:.3f}  loo_norm {nm:.3f}  |{bar}")
    ranked = sorted(layers, key=lambda L: st.median(results[L]["raw"]["gt_pct"]))
    print(f"\n  best 5 layers by raw gt_pct: "
          + ", ".join(f"{L}({st.median(results[L]['raw']['gt_pct']):.3f})" for L in ranked[:5]))
    bl = ranked[0]
    bi = layers.index(bl)
    nb = [layers[i] for i in (bi - 1, bi + 1) if 0 <= i < len(layers)]
    print(f"  NEIGHBOUR CHECK for best layer {bl}: "
          + ", ".join(f"{L}={st.median(results[L]['raw']['gt_pct']):.3f}" for L in nb))
    print("     a real layer effect has neighbours close to the peak; an isolated spike is noise.")

    print("\n" + "=" * 74)
    print("PRIMARY: does removing the positional background promote the target?")
    print("=" * 74)
    print(f"  {'layer':6} {'readout':12} {'med gt_pct':>11} {'p10':>7} {'in GT':>7} "
          f"{'near':>7} {'med dist':>9}")
    best = None
    for L in layers:
        for name in ("raw", "loo_norm", "loo_z", "no_border"):
            g = results[L][name]["gt_pct"]
            ig = results[L][name]["in_gt"]
            pct = 100 * sum(ig) / len(ig)
            nr = results[L][name]["near"]; dd = results[L][name]["dist"]
            print(f"  {L:6} {name:12} {st.median(g):11.3f} {st.quantiles(g,n=10)[0]:7.3f} "
                  f"{pct:6.1f}% {100*sum(nr)/len(nr):6.1f}% {st.median(dd):9.3f}")
            if best is None or st.median(g) < best[0]:
                best = (st.median(g), L, name, pct)
        print()
    bm, bl, bn, bp = best
    print(f"  BEST: {bl}/{bn}  median gt_pct {bm:.3f}  argmax-in-GT {bp:.1f}%")
    raw_med = st.median(results[bl]["raw"]["gt_pct"])
    print(f"  raw at the same layer: {raw_med:.3f}  ->  improvement {raw_med - bm:+.3f}")
    if bp >= 20:
        print("  => LOCALIZER. The argmax lands on the target often enough to drive a proposer.")
    elif bm < 0.08:
        print("  => STRONG RANKING, weak peak. Target is top-8% but the argmax still is not it;")
        print("     a proposer must aggregate ranks, not trust the peak.")
    else:
        print("  => Background subtraction does not convert the ranking into a peak. A learned")
        print("     read-out is the remaining option, and this is the evidence justifying it.")


if __name__ == "__main__":
    main()
