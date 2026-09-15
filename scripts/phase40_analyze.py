"""
Phase 40 analyzer: does a SECOND window at a DIFFERENT place buy anything at a fixed budget?

Nothing is fitted here. W=0.15, the L16-26 block, the ring mask and MIN_SEP all come from earlier
phases; this file only scores. The verdict rule below was written into phase40_multiwindow.py
before the run and is reproduced verbatim.

    two_diff > two_same  (paired, CI excludes 0)   -> a genuinely different second region ADDS
                                                      accuracy at fixed budget. Coverage is doing
                                                      causal work; §3 upgrades from mediator.
    two_diff ~= two_same                           -> the second window adds nothing beyond format
                                                      and resolution. The coverage account FAILS as
                                                      an intervention and stays correlational.
    two_diff < two_same                            -> a second location actively hurts.

WHY two_same IS THE COMPARISON AND NOT `one`
--------------------------------------------
Against `two_same`, `two_diff` holds the image count (2), the total realized budget (300), the
per-image resolution (150) and the two-image prompt format all constant. Only the second window's
LOCATION varies. Comparing against `one` instead would confound a second region with halved
resolution and with the two-image format, which is exactly the confound that sank the naive reading
of the W sweep in §3.4(b).

Coverage is recomputed offline from the logged boxes and peaks -- the GT box is an EXPLANATORY
variable only; the proposal never saw it.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase40_multiwindow.jsonl"
W = 0.15


def cov1(b, cx, cy, W):
    gx0, gy0, gx1, gy1 = b
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0.0, W
    if y0 < 0: y0, y1 = 0.0, W
    if x1 > 1: x0, x1 = 1 - W, 1.0
    if y1 > 1: y0, y1 = 1 - W, 1.0
    return (max(0.0, min(gx1, x1) - max(gx0, x0)), max(0.0, min(gy1, y1) - max(gy0, y0)),
            (gx1 - gx0) * (gy1 - gy0))


def cov_union(b, pts, W):
    """Union coverage on a fine grid -- two windows may overlap, so areas cannot simply be added."""
    gx0, gy0, gx1, gy1 = b
    if gx1 <= gx0 or gy1 <= gy0:
        return 0.0
    boxes = []
    for cx, cy in pts:
        x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
        if x0 < 0: x0, x1 = 0.0, W
        if y0 < 0: y0, y1 = 0.0, W
        if x1 > 1: x0, x1 = 1 - W, 1.0
        if y1 > 1: y0, y1 = 1 - W, 1.0
        boxes.append((x0, y0, x1, y1))
    S, hit = 40, 0
    for i in range(S):
        px = gx0 + (i + .5) / S * (gx1 - gx0)
        for j in range(S):
            py = gy0 + (j + .5) / S * (gy1 - gy0)
            if any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in boxes):
                hit += 1
    return hit / (S * S)


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 40:
        print(f"only {len(rows)} rows so far; wait")
        return
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    arms = ["uniform", "one", "two_diff", "two_same", "two_rand"]
    n = len(rows)
    sep = sum(r["peaks_separated"] for r in rows)
    print(f"n = {n} items;  {sep} ({100*sep/n:.0f}%) admitted a separated second peak")

    print("\n=== BUDGET GATE (all arms must sit within 10%) ===")
    med = {a: st.median([r["realized_tokens"][a] for r in rows]) for a in arms}
    for a in arms:
        print(f"  {a:12}{med[a]:6.0f} tok")
    spread = (max(med.values()) - min(med.values())) / min(med.values())
    print(f"  spread {100*spread:.1f}%  -> {'OK' if spread < .10 else '!! CONTRAST VOID'}")
    if spread >= .10:
        print("  refusing to report contrasts on a voided budget gate.")
        return

    for r in rows:
        r["_c1"] = cov_union(r["gt_box_frac"], [r["peak1"]], W)
        r["_c2"] = cov_union(r["gt_box_frac"], [r["peak1"], r["peak2"]], W)

    print("\n=== COVERAGE ACHIEVED (explanatory only; the proposal never saw a box) ===")
    print(f"  one window   mean {100*st.mean([r['_c1'] for r in rows]):5.1f}%   "
          f"P(cov=0) {100*st.mean([r['_c1'] <= .001 for r in rows]):5.1f}%")
    print(f"  two windows  mean {100*st.mean([r['_c2'] for r in rows]):5.1f}%   "
          f"P(cov=0) {100*st.mean([r['_c2'] <= .001 for r in rows]):5.1f}%")
    gained = [r for r in rows if r["_c2"] > r["_c1"] + .001]
    rescued = [r for r in rows if r["_c1"] <= .001 < r["_c2"]]
    print(f"  the 2nd window ADDS coverage on {len(gained)} items ({100*len(gained)/n:.0f}%), "
          f"and RESCUES {len(rescued)} items from zero coverage ({100*len(rescued)/n:.0f}%)")

    print("\n=== ACCURACY ===")
    u = [hit(r, "uniform") for r in rows]
    print(f"  {'arm':<12}{'acc':>8}{'vs uniform':>13}{'95% CI':>18}")
    A = {}
    for a in arms:
        v = [hit(r, a) for r in rows]
        A[a] = v
        d = [x - y for x, y in zip(v, u)]
        lo, hi = boot(d)
        print(f"  {a:<12}{100*st.mean(v):>7.1f}%{100*st.mean(d):>+12.1f}pp"
              f"   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== THE PRE-REGISTERED CONTRAST (paired, same items) ===")
    for a, b, note in [("two_diff", "two_same", "** THE DECIDER: only the 2nd window's PLACE differs"),
                       ("two_diff", "two_rand", "placement control for window 2"),
                       ("two_diff", "one", "does multi-window beat the deployed method outright?"),
                       ("two_same", "one", "cost of halving resolution + 2-image format alone")]:
        d = [x - y for x, y in zip(A[a], A[b])]
        lo, hi = boot(d)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {a:<9} - {b:<9}{100*st.mean(d):>+7.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  "
              f"{sig:<12}{note}")

    print("\n=== WHERE THE ACCOUNT SAYS THE GAIN MUST LIVE ===")
    print("  if coverage is causal, two_diff - two_same must be concentrated on the items where the")
    print("  second window actually ADDED coverage, and absent where it did not.")
    print(f"  {'subgroup':<34}{'n':>4}{'two_diff - two_same':>21}{'95% CI':>18}")
    for nm, idx in [("2nd window ADDED coverage", [i for i, r in enumerate(rows)
                                                   if r["_c2"] > r["_c1"] + .001]),
                    ("2nd window added NOTHING", [i for i, r in enumerate(rows)
                                                  if r["_c2"] <= r["_c1"] + .001]),
                    ("RESCUED from zero coverage", [i for i, r in enumerate(rows)
                                                    if r["_c1"] <= .001 < r["_c2"]])]:
        if len(idx) < 8:
            print(f"  {nm:<34}{len(idx):>4}   too few")
            continue
        d = [A["two_diff"][i] - A["two_same"][i] for i in idx]
        lo, hi = boot(d)
        print(f"  {nm:<34}{len(idx):>4}{100*st.mean(d):>+20.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== by category ===")
    for cat in sorted({r["category"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["category"] == cat]
        cells = "  ".join(f"{a} {100*st.mean([A[a][i] for i in idx]):5.1f}%" for a in arms)
        print(f"  {cat:<20}n={len(idx):<4}{cells}")

    print("\n" + "=" * 78)
    print("VERDICT vs the rule fixed before the run")
    print("=" * 78)
    d = [x - y for x, y in zip(A["two_diff"], A["two_same"])]
    lo, hi = boot(d)
    if lo > 0:
        print(f"  two_diff - two_same = {100*st.mean(d):+.1f}pp CI[{100*lo:+.1f},{100*hi:+.1f}]")
        print("  => A DIFFERENT SECOND REGION ADDS ACCURACY AT FIXED BUDGET.")
        print("     Coverage moved by intervention, with resolution and format held constant.")
        print("     §3 upgrades from mediator toward cause.")
    elif hi < 0:
        print(f"  two_diff - two_same = {100*st.mean(d):+.1f}pp CI[{100*lo:+.1f},{100*hi:+.1f}]")
        print("  => A SECOND LOCATION HURTS. Contiguity matters in a way the account denies.")
    else:
        print(f"  two_diff - two_same = {100*st.mean(d):+.1f}pp CI[{100*lo:+.1f},{100*hi:+.1f}]")
        print("  => NULL. The second window buys nothing beyond format and resolution.")
        print("     The coverage account FAILS as an intervention and remains correlational.")
        print("     Report as a negative; §3 stays a mediator and §9 must be rewritten.")


if __name__ == "__main__":
    main()
