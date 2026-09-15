"""
Phase 34: the mask the MECHANISM predicts. Offline, no GPU, no forward passes.

WHAT CHANGED
------------
The sink was described in Phase 30b/§3 as sitting at the image CORNERS. Re-measuring it stratified
by grid shape shows that reading was wrong, and wrong in a way that matters for the method:

    region              mass    area   enrichment
    LAST  COL (gw-1)   25.3%    5.0%      5.1x
    FIRST COL (0)      14.2%    5.0%      2.9x
    TOP ROW             5.5%    6.3%      0.9x     <- NOT enriched
    BOTTOM ROW          5.3%    6.3%      0.8x     <- NOT enriched
    INTERIOR           49.7%   77.5%      0.6x

The effect is COLUMNAR, not a border. Rows carry no excess mass at all. "Corners" was the
intersection of a strong column effect with a null row effect. Two further tests agree:

  TRANSPOSE (14x21 vs 21x14, identical n=294, geometry transposed): sink positions are identical in
  (row,col) space -- col = gw-1 in both -- but only 4/8 absolute indices coincide. A register token
  would reuse the same absolute slots; it does not.

  INTERIOR REPLAY: modal-grid sink indices, replayed on other grid shapes where they land in the
  INTERIOR, re-fire at 1.1% against 3.3% chance -- BELOW chance. Not fixed slots.

This is the signature of RASTER SERIALIZATION, not of 2-D image position: col gw-1 is the token
before a row wrap and col 0 the token after it, while top and bottom rows are ordinary raster
positions and duly show nothing.

THE PREDICTION BEING TESTED HERE
--------------------------------
The deployed localizer (phase31/32/33) masks the whole OUTER RING. If the mechanism is columnar,
that mask is throwing away the top and bottom rows for nothing -- unbiased image area, 0.9x and
0.8x -- and any target whose centre lies there is made unreachable. A COLUMN-ONLY mask should
therefore be >= the ring mask on both read-outs, with the gain concentrated on exactly those items.

Pre-registered reading, fixed before running:
    col-only >= ring on BOTH gt_pct and argmax-in-GT   -> mechanism-derived improvement; adopt.
    col-only ~= ring                                   -> the discarded rows were empty anyway;
                                                          the mechanism claim stands, the method
                                                          gain does not. Report as such.
    col-only <  ring                                   -> the ring mask is doing something the
                                                          column account does not explain. Report.

PRIMARY METRIC: median gt_pct, the rank of the GT-centre cell as a fraction of n_img.
Chance is 0.500 EXACTLY by construction. Secondary: P(argmax inside GT).
Read-out is the selection-free L16-26 block mean from §4W -- no layer is chosen on this data.
"""
import json
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"
BLOCK = [f"L{i}" for i in range(16, 27)]


def block_mean(r):
    n = len(r["attn"]["L16"])
    acc = [0.0] * n
    for k in BLOCK:
        v = r["attn"][k]
        s = sum(v) or 1.0
        for i in range(n):
            acc[i] += v[i] / s
    return [x / len(BLOCK) for x in acc]


MASKS = {
    "raw (no mask)":        lambda rr, cc, gh, gw: True,
    "ring (deployed)":      lambda rr, cc, gh, gw: 0 < rr < gh - 1 and 0 < cc < gw - 1,
    "columns only":         lambda rr, cc, gh, gw: 0 < cc < gw - 1,
    "last column only":     lambda rr, cc, gh, gw: cc < gw - 1,
    "rows only (placebo)":  lambda rr, cc, gh, gw: 0 < rr < gh - 1,
}


def main():
    rows = [json.loads(l) for l in open(PATH)]
    n = len(rows)
    print(f"n = {n} items; read-out = selection-free block mean {BLOCK[0]}-{BLOCK[-1]}")

    # how much evidence does each mask make UNREACHABLE?
    print("\n=== COST OF EACH MASK: items whose GT centre is masked away ===")
    print(f"  {'mask':<22}{'kept area':>11}{'GT centre masked':>19}")
    for name, keep in MASKS.items():
        ka = lost = 0.0
        for r in rows:
            gh, gw = r["grid"]
            k = sum(1 for rr in range(gh) for cc in range(gw) if keep(rr, cc, gh, gw))
            ka += k / (gh * gw) / n
            gx0, gy0, gx1, gy1 = r["gt_box_frac"]
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            rr, cc = min(gh - 1, int(cy * gh)), min(gw - 1, int(cx * gw))
            lost += (not keep(rr, cc, gh, gw)) / n
        print(f"  {name:<22}{100*ka:>10.1f}%{100*lost:>18.1f}%")

    print("\n=== READ-OUT QUALITY (chance gt_pct 0.500; LOWER better) ===")
    chance_in = st.mean([(r["gt_box_frac"][2] - r["gt_box_frac"][0]) *
                         (r["gt_box_frac"][3] - r["gt_box_frac"][1]) for r in rows])
    print(f"  chance P(argmax in GT) = {100*chance_in:.2f}%")
    print(f"  {'mask':<22}{'med gt_pct':>12}{'p25':>8}{'argmax-in-GT':>15}{'x chance':>10}")
    out = {}
    for name, keep in MASKS.items():
        gp, ing = [], []
        for r in rows:
            gh, gw = r["grid"]
            a = block_mean(r)
            sc = []
            for i in range(gh * gw):
                rr, cc = i // gw, i % gw
                sc.append(a[i] if keep(rr, cc, gh, gw) else -1.0)
            gx0, gy0, gx1, gy1 = r["gt_box_frac"]
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            gr, gc = min(gh - 1, int(cy * gh)), min(gw - 1, int(cx * gw))
            gv = sc[gr * gw + gc]
            gp.append(sum(1 for v in sc if v > gv) / len(sc))
            j = max(range(len(sc)), key=lambda k: sc[k])
            px, py = ((j % gw) + .5) / gw, ((j // gw) + .5) / gh
            ing.append(1.0 * (gx0 <= px <= gx1 and gy0 <= py <= gy1))
        out[name] = (gp, ing)
        print(f"  {name:<22}{st.median(gp):>12.3f}"
              f"{sorted(gp)[len(gp)//4]:>8.3f}{100*st.mean(ing):>14.1f}%"
              f"{st.mean(ing)/chance_in:>9.0f}x")

    print("\n=== PAIRED TEST: columns-only vs ring (deployed) ===")
    import random
    a_gp, a_in = out["columns only"]
    b_gp, b_in = out["ring (deployed)"]
    for lbl, A, B, sign in [("gt_pct (lower better)", a_gp, b_gp, -1),
                            ("argmax-in-GT", a_in, b_in, +1)]:
        d = [x - y for x, y in zip(A, B)]
        rng = random.Random(0)
        s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
        lo, hi = s[250], s[9750]
        better = (sign * st.mean(d)) > 0
        print(f"  {lbl:<24} delta {st.mean(d):+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
              f"{'col-only BETTER' if better else 'ring better/equal'}"
              f"{'  (significant)' if lo > 0 or hi < 0 else '  (n.s.)'}")

    print("\n=== SUBGROUP the mechanism names: items whose GT centre is in the top/bottom row ===")
    idx = []
    for i, r in enumerate(rows):
        gh, gw = r["grid"]
        gx0, gy0, gx1, gy1 = r["gt_box_frac"]
        rr = min(gh - 1, int(((gy0 + gy1) / 2) * gh))
        if rr in (0, gh - 1):
            idx.append(i)
    print(f"  n = {len(idx)} items ({100*len(idx)/n:.1f}%) -- the ring mask deletes their target")
    if idx:
        for name in ("ring (deployed)", "columns only"):
            gp, ing = out[name]
            print(f"    {name:<20} med gt_pct {st.median([gp[i] for i in idx]):.3f}"
                  f"   argmax-in-GT {100*st.mean([ing[i] for i in idx]):5.1f}%")


if __name__ == "__main__":
    main()
