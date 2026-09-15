"""
Phase 38: is coverage CAUSAL, or does attention just look at easy items? Offline, no GPU.

THE CONFOUND IN PHASE 37
------------------------
Phase 37's dose-response bins items by how much of the GT box the ATTENTION-placed window covers.
But the attention peak is chosen by the model, so coverage is not assigned independently of the
item: the model may simply attend accurately on items it was going to get right anyway. Under that
account coverage predicts the delta without causing it, and the whole mechanism is post-hoc.

THE INSTRUMENT ALREADY IN THE DATA
----------------------------------
Phase 32 logged `rand@W` at four window sizes: a crop of the SAME size at a UNIFORMLY RANDOM centre,
same budget, same pipeline, same items. There, coverage is assigned by the random number generator
and is therefore INDEPENDENT of item difficulty, of saliency, and of whatever the model finds easy.

    If the dose-response survives under RANDOM placement, coverage is doing causal work.
    If it flattens, Phase 37 was selection and the mechanism claim must be withdrawn.

This is the same logic as the rand control that decided V*Bench (§4) and HR-Bench (§5B), used here
as an instrument rather than as a baseline.

A SECOND, INDEPENDENT HANDLE
----------------------------
Window size W is also set by us, not by the model. Growing W raises coverage mechanically while
LOWERING the resolution gain (same tokens over more area). If coverage is causal, delta should track
coverage across the W sweep on items where the window would otherwise miss -- and the two effects
should trade off, giving an interior optimum rather than a monotone preference for large W.

Difficulty is controlled directly: `uniform` accuracy on the same items is reported in every bin, so
a bin that is simply easier is visible rather than inferred.
"""
import json
import random
import statistics as st

VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
BOX = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30c_attn_maps_all.jsonl"


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
    Ws = [w for w in ("0.15", "0.25", "0.35", "0.5") if f"rand@{w}" in rows[0]["probs"]]
    if "rand_frac" not in rows[0]:
        print("  NOTE: `rand_frac` (the random centre actually used) is not logged in phase32.")
        print("  Coverage under random placement cannot be measured exactly from this file.")
        print("  Falling back to the EXPECTED-coverage instrument, which needs no centre: for a")
        print("  uniformly placed window the chance of covering a box is a known function of the")
        print("  box's size, so box size is a valid instrument for coverage under RANDOM placement.")
        print()

    print("=" * 80)
    print("INSTRUMENT 1: RANDOM PLACEMENT. Coverage assigned by the RNG, not by the model.")
    print("=" * 80)
    print("  P(a uniformly placed W-window covers the box centre) ~ (1-W)^-2 x W^2 area overlap,")
    print("  which is a function of BOX SIZE alone -- so we bin on box size, which the model does")
    print("  not choose, and read the RANDOM arm's delta. Difficulty is shown via `uniform`.")
    print()
    for r in rows:
        b = bx[r["question_id_full"]]
        r["_area"] = (b[2] - b[0]) * (b[3] - b[1])
    qs = sorted(r["_area"] for r in rows)
    cut = [qs[len(qs) // 3], qs[2 * len(qs) // 3]]
    bands = [(-1, cut[0], "small box"), (cut[0], cut[1], "medium box"), (cut[1], 1e9, "large box")]
    W = "0.15"
    print(f"  {'box band':<13}{'n':>4}{'med area':>10}{'uniform':>9}"
          f"{'rand@W delta':>14}{'95% CI':>18}{'attn@W delta':>14}")
    for lo_, hi_, nm in bands:
        g = [r for r in rows if lo_ < r["_area"] <= hi_]
        u = [hit(r, "uniform") for r in g]
        rd = [hit(r, f"rand@{W}") - hit(r, "uniform") for r in g]
        at = [hit(r, f"attn@{W}") - hit(r, "uniform") for r in g]
        l, h = boot(rd)
        print(f"  {nm:<13}{len(g):>4}{st.median([r['_area'] for r in g]):>10.4f}"
              f"{100*st.mean(u):>8.1f}%{100*st.mean(rd):>+13.1f}   [{100*l:+.1f},{100*h:+.1f}]"
              f"{100*st.mean(at):>+13.1f}")
    print("\n  a bigger box is MORE likely to be hit by a random window. If coverage is causal the")
    print("  RANDOM arm's delta must rise with box size -- with no help from attention at all.")

    print("\n" + "=" * 80)
    print("INSTRUMENT 2: WINDOW SIZE. W is set by us; growing it raises coverage and lowers")
    print("              the resolution gain. Causal coverage predicts an INTERIOR optimum.")
    print("=" * 80)
    print(f"  {'arm':<10}" + "".join(f"{'W='+w:>11}" for w in Ws))
    for arm in ("attn", "rand"):
        cells = []
        for w in Ws:
            d = [hit(r, f"{arm}@{w}") - hit(r, "uniform") for r in rows]
            cells.append(100 * st.mean(d))
        print(f"  {arm:<10}" + "".join(f"{c:>+10.1f}" for c in cells))
    print("\n  mean GT coverage achieved by the ATTENTION window at each W:")
    line = []
    for w in Ws:
        c = st.mean([cov(bx[r["question_id_full"]], *r["peak_frac"], float(w)) for r in rows])
        line.append(100 * c)
    print(f"  {'coverage':<10}" + "".join(f"{c:>10.1f}%" for c in line))

    print("\n" + "=" * 80)
    print("CONTROLLING DIFFICULTY DIRECTLY: dose-response WITHIN a difficulty stratum")
    print("=" * 80)
    print("  stratify on `uniform` correctness -- the purest difficulty proxy available -- and")
    print("  re-run the Phase 37 dose-response inside each stratum. If coverage only marked easy")
    print("  items, it would carry no signal once difficulty is held fixed.")
    for r in rows:
        r["_cov"] = cov(bx[r["question_id_full"]], *r["peak_frac"], 0.15)
    print(f"\n  {'uniform was':<14}{'coverage':<14}{'n':>4}{'attn@0.15 acc':>15}{'95% CI':>18}")
    for ok, nm in [(1.0, "CORRECT"), (0.0, "WRONG")]:
        for lo_, hi_, cn in [(-.001, .001, "missed (0%)"), (.001, 1.01, "covered (>0)")]:
            g = [r for r in rows if hit(r, "uniform") == ok and lo_ < r["_cov"] <= hi_]
            if len(g) < 8:
                print(f"  {nm:<14}{cn:<14}{len(g):>4}   too few")
                continue
            a = [hit(r, "attn@0.15") for r in g]
            l, h = boot(a)
            print(f"  {nm:<14}{cn:<14}{len(g):>4}{100*st.mean(a):>14.1f}%"
                  f"   [{100*l:+.1f},{100*h:+.1f}]")
    print("\n  on items uniform got WRONG, coverage is the chance to RECOVER;")
    print("  on items uniform got RIGHT, missing is the chance to BREAK something that worked.")


if __name__ == "__main__":
    main()
