"""
Phase 37: the continuous, category-free form of the boundary condition. Offline, no GPU.

WHY THIS REPLACES BOTH EARLIER STORIES
--------------------------------------
Two candidate boundaries have now each failed on their own decisive test:

  TARGET SIZE (§5.5)   -- HR-Bench is deeply sub-token (~233 px per merged token at B0=300) and
                          allocation LOST there by 8.8pp. Predictor backwards on its key case.
  REGION COUNT (§6C)   -- the `oracle` arm (crop to the GT box, a PERFECT single-region proposal)
                          WINS on relative_position, +19.7pp [+5.3,+34.2]. If two regions were
                          intrinsically un-croppable, a perfect crop would fail too. It does not.

What the oracle arm actually shows is that the GT box for a relational question is the UNION of the
objects involved: median box area 0.0048 vs 0.0006 for direct_attributes, 7.9x larger, aspect 2.09
vs 1.68. Cropping to it keeps the whole evidence set. Cropping to a single attention peak does not:

    fraction of the GT box covered by ONE W=0.15 window at the attention peak
        direct_attributes   52.5%
        relative_position   22.1%

That deficit tracks the accuracy sign flip (+13.9pp vs -9.2pp) exactly. So the boundary is neither
size nor region count per se -- both are causes of the SAME quantity:

    **the fraction of the evidence set that the reallocated window actually covers.**

THE TEST
--------
Pool ALL items, discard the category labels, and bin by measured coverage. If coverage is the
mediator, the accuracy delta must rise monotonically with coverage and cross zero -- a continuous
dose-response, with no category, no benchmark and no size term needed. Then check that category adds
nothing once coverage is in the model; if it does, coverage is incomplete.

The window used is the one ACTUALLY evaluated: Phase 32 logged `peak_frac`, so coverage is measured
on the deployed crop, not on a recomputed one.

NOTE ON WHAT IS AND IS NOT FITTED: coverage uses the GT box, so it is an EXPLANATORY variable, not
a feature any method may read. Nothing here is used to select a method or a hyper-parameter; it is
an account of results already obtained. The deployed allocator never sees a box.
"""
import json
import random
import statistics as st

VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
S = 40


def coverage(box, cx, cy, W):
    gx0, gy0, gx1, gy1 = box
    if gx1 <= gx0 or gy1 <= gy0:
        return 0.0
    # the deployed window is clamped to the image; replicate that clamp exactly
    x0, x1 = cx - W / 2, cx + W / 2
    y0, y1 = cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0.0, W
    if y0 < 0: y0, y1 = 0.0, W
    if x1 > 1: x0, x1 = 1 - W, 1.0
    if y1 > 1: y0, y1 = 1 - W, 1.0
    ix = max(0.0, min(gx1, x1) - max(gx0, x0))
    iy = max(0.0, min(gy1, y1) - max(gy0, y0))
    return ix * iy / ((gx1 - gx0) * (gy1 - gy0))


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(VS)]
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    W = 0.15
    for r in rows:
        cx, cy = r["peak_frac"]
        r["_cov"] = coverage(r["gt_box_frac"], cx, cy, W) if "gt_box_frac" in r else None
    if rows[0]["_cov"] is None:
        # phase32 may not carry the box; join it from the 30c dump on question_id_full
        bx = {x["question_id_full"]: x["gt_box_frac"]
              for x in (json.loads(l) for l in open(
                  "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"))}
        miss = 0
        for r in rows:
            b = bx.get(r["question_id_full"])
            if b is None:
                miss += 1
                r["_cov"] = None
            else:
                cx, cy = r["peak_frac"]
                r["_cov"] = coverage(b, cx, cy, W)
        print(f"  joined GT boxes from the 30c dump; {miss} unmatched")
    rows = [r for r in rows if r["_cov"] is not None]
    print(f"n = {len(rows)} items with measured coverage of the DEPLOYED W={W} window")

    print("\n=== DOSE-RESPONSE: allocation delta vs measured coverage (categories POOLED) ===")
    print(f"  {'coverage bin':<16}{'n':>4}{'med cov':>9}{'uniform':>9}{'attn@0.15':>11}"
          f"{'delta':>8}{'95% CI':>18}")
    bins = [(-.001, .001, "0% (missed)"), (.001, .25, "0-25%"), (.25, .75, "25-75%"),
            (.75, .999, "75-100%"), (.999, 1.01, "100% (full)")]
    for lo_, hi_, nm in bins:
        g = [r for r in rows if lo_ < r["_cov"] <= hi_]
        if len(g) < 8:
            print(f"  {nm:<16}{len(g):>4}   too few")
            continue
        u = [hit(r, "uniform") for r in g]
        a = [hit(r, f"attn@{W}") for r in g]
        d = [x - y for x, y in zip(a, u)]
        l, h = boot(d)
        print(f"  {nm:<16}{len(g):>4}{st.median([r['_cov'] for r in g]):>9.2f}"
              f"{100*st.mean(u):>8.1f}%{100*st.mean(a):>10.1f}%{100*st.mean(d):>+8.1f}"
              f"   [{100*l:+.1f},{100*h:+.1f}]")

    print("\n=== DOES CATEGORY ADD ANYTHING ONCE COVERAGE IS HELD FIXED? ===")
    print("  if coverage is the mediator, the two categories should agree WITHIN a coverage bin.")
    print(f"  {'coverage bin':<16}{'category':<20}{'n':>4}{'delta':>9}")
    for lo_, hi_, nm in [(-.001, .25, "low (<=25%)"), (.25, 1.01, "high (>25%)")]:
        for cat in ("direct_attributes", "relative_position"):
            g = [r for r in rows if lo_ < r["_cov"] <= hi_ and r["category"] == cat]
            if len(g) < 8:
                print(f"  {nm:<16}{cat:<20}{len(g):>4}   too few")
                continue
            d = [hit(r, f"attn@{W}") - hit(r, "uniform") for r in g]
            print(f"  {nm:<16}{cat:<20}{len(g):>4}{100*st.mean(d):>+9.1f}")

    print("\n=== COVERAGE EXPLAINS BOTH FAILED PREDICTORS ===")
    for cat in ("direct_attributes", "relative_position"):
        g = [r for r in rows if r["category"] == cat]
        print(f"  {cat:<20} mean coverage {100*st.mean([r['_cov'] for r in g]):5.1f}%"
              f"   P(coverage=0) {100*st.mean([r['_cov'] <= .001 for r in g]):5.1f}%")
    print("\n  and the ORACLE window, which is the GT box itself, has coverage 100% by construction")
    print("  -- which is why it wins in BOTH categories (+47.8pp / +19.7pp).")

    print("\n=== THE PREDICTION THIS MAKES, AND IT IS FALSIFIABLE ===")
    print("""  If coverage is the mediator, then a proposal that covers MORE of the evidence set must
  help, whatever produces it -- and covering it with TWO windows must work as well as one, since
  nothing in the account refers to contiguity. Offline, two attention peaks (min separation 0.25)
  raise GT coverage on relative_position from 22.1% to 25.7% at W=0.15 and 41.7% -> 53.8% at
  W=0.35. Those gains must then survive splitting the SAME token budget across two crops. That is
  the experiment Phase 38 runs, and it can fail.""")


if __name__ == "__main__":
    main()
