"""
Phase 30 analyzer: apply the pre-registered viability rule to the attention-proposal pre-check.

Written BEFORE the data existed, so the thresholds cannot drift toward whatever the numbers turn
out to be. The rule, restated from the experiment docstring:

    median containment >= 0.5 AND median area_frac <= 0.5  -> VIABLE, run the full arm matrix
    median area_frac   >  0.8                              -> DEAD (proposal ~= whole image)
    containment high AND area_frac high                    -> attention is diffuse; the rank signal
                                                              of Phase 22 does not convert into a
                                                              REGION. Report as a negative.

The headline configuration is L2 @ r=0.25 and only that one. L4/L8 and the other keep rates are
printed as a sensitivity table -- they are NOT eligible to become the headline, because picking the
best cell of a 3x3 grid after seeing it is fitting the selector to the test set.

`headroom` = containment / area_frac is the quantity that decides whether allocation can pay. A
proposal covering the whole image scores 1.0 by construction (it contains everything and costs
everything). Only headroom > 1 means the proposal concentrates evidence faster than it spends
budget, and the bigger it is the more resolution a matched-budget re-render buys on the target.
"""
import json
import statistics as st
import sys
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30_proposal_quality.jsonl"
HEAD_L, HEAD_R = 2, 0.25


def q(v, p):
    return st.quantiles(v, n=100)[p - 1] if len(v) > 2 else (v[0] if v else float("nan"))


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if not rows:
        print("no rows yet")
        return
    print(f"n = {len(rows)} items")
    bad = [r for r in rows if "note" in r]
    if bad:
        print(f"  WARNING: {len(bad)} rows with token-count mismatch -- inspect before trusting")
    cats = defaultdict(int)
    for r in rows:
        cats[r["category"]] += 1
    print("  categories:", dict(cats))
    print(f"  GT box area as frac of image: median {st.median([r['gt_area_frac'] for r in rows]):.5f}")
    print(f"  realized tokens: median {st.median([r['realized_tokens'] for r in rows]):.0f}"
          f"  (spread {min(r['realized_tokens'] for r in rows)}"
          f"-{max(r['realized_tokens'] for r in rows)})")

    key = f"L{HEAD_L}@{HEAD_R}"
    print(f"\n=== HEADLINE (pre-registered): {key} ===")
    cont = [r["proposal"][key]["containment"] for r in rows]
    area = [r["proposal"][key]["area_frac"] for r in rows]
    head = [r["proposal"][key]["headroom"] for r in rows]
    mass = [r["proposal"][key]["attn_mass_in_gt"] for r in rows]
    for nm, v in [("containment", cont), ("area_frac", area), ("headroom", head),
                  ("attn_mass_in_gt", mass)]:
        print(f"  {nm:16} median {st.median(v):8.4f}  mean {st.mean(v):8.4f}"
              f"  p10 {q(v,10):8.4f}  p90 {q(v,90):8.4f}")
    print(f"  containment == 1.0 (GT fully inside proposal): "
          f"{100*sum(1 for c in cont if c >= 0.999)/len(cont):.1f}% of items")
    print(f"  containment >= 0.5:  {100*sum(1 for c in cont if c >= 0.5)/len(cont):.1f}%")
    print(f"  area_frac  <= 0.25:  {100*sum(1 for a in area if a <= 0.25)/len(area):.1f}%")

    mc, ma = st.median(cont), st.median(area)
    print("\n=== PRE-REGISTERED VERDICT ===")
    if ma > 0.8:
        print(f"  DEAD. median area_frac {ma:.3f} > 0.8 -- the proposal is essentially the whole")
        print("  image, so a matched-budget re-render is the uniform baseline. Report as negative.")
    elif mc >= 0.5 and ma <= 0.5:
        print(f"  VIABLE. median containment {mc:.3f} >= 0.5 and median area_frac {ma:.3f} <= 0.5.")
        print(f"  Expected resolution gain on target at matched budget: {1/max(ma,1e-9):.1f}x area,"
              f" ~{(1/max(ma,1e-9))**0.5:.1f}x linear. Proceed to the full arm matrix.")
    elif mc >= 0.5:
        print(f"  DIFFUSE. containment {mc:.3f} is fine but area_frac {ma:.3f} is too large:")
        print("  attention ranks the target highly (Phase 22) yet its top tokens are scattered, so a")
        print("  BOUNDING BOX is the wrong read-out. Negative, with the read-out named as the cause.")
    else:
        print(f"  DEAD. median containment {mc:.3f} < 0.5 -- the proposal misses the evidence.")

    print("\n=== sensitivity (NOT eligible to be the headline) ===")
    print(f"  {'config':12} {'med cont':>9} {'med area':>9} {'med head':>9}")
    for L in (2, 4, 8):
        for r in (0.10, 0.25, 0.50):
            k = f"L{L}@{r}"
            if k not in rows[0]["proposal"]:
                continue
            c = [x["proposal"][k]["containment"] for x in rows]
            a = [x["proposal"][k]["area_frac"] for x in rows]
            h = [x["proposal"][k]["headroom"] for x in rows]
            star = "  <- headline" if (L, r) == (HEAD_L, HEAD_R) else ""
            print(f"  {k:12} {st.median(c):9.4f} {st.median(a):9.4f} {st.median(h):9.4f}{star}")

    print("\n=== by category ===")
    for cat in sorted(cats):
        g = [r for r in rows if r["category"] == cat]
        c = [x["proposal"][key]["containment"] for x in g]
        a = [x["proposal"][key]["area_frac"] for x in g]
        print(f"  {cat:20} n={len(g):3d}  med cont {st.median(c):.3f}  med area {st.median(a):.3f}")

    print("\n=== by GT object size (the axis the whole paper runs on) ===")
    srt = sorted(rows, key=lambda r: r["gt_area_frac"])
    nb = 4
    for i in range(nb):
        g = srt[i*len(srt)//nb:(i+1)*len(srt)//nb]
        c = [x["proposal"][key]["containment"] for x in g]
        a = [x["proposal"][key]["area_frac"] for x in g]
        h = [x["proposal"][key]["headroom"] for x in g]
        lo, hi = g[0]["gt_area_frac"], g[-1]["gt_area_frac"]
        print(f"  gt_area {lo:.5f}-{hi:.5f} n={len(g):3d}  cont {st.median(c):.3f}"
              f"  area {st.median(a):.3f}  headroom {st.median(h):6.2f}")


if __name__ == "__main__":
    main()
