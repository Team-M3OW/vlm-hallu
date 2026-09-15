"""
Phase 47 analyzer: every prior-art policy scored against what its OWN token spend buys uniformly.

THE COMPARISON NOBODY RUNS
--------------------------
Each arm is charged for every image token it processes across every pass, and compared against the
MEASURED uniform sweep at that same total spend (Qwen3-VL, V*Bench, phase27):

    150 -> 46.6%   294 -> 56.5%   600 -> 66.0%   1176 -> 70.2%
    2400 -> 75.4%  4760 -> 81.2%  7957 -> 84.8%

So a method spending 2700 tokens on tree search is asked to beat a single uniform image of 2700
tokens, which is ~75%. That is the question the crop/zoom literature does not ask, and it is the
only question that separates "this search policy works" from "this method spends more".

Interpolation is in log-tokens, which is how the measured sweep behaves. Arms are also reported at
their raw accuracy so a reader who rejects the interpolation can still read the table.

REIMPLEMENTATION CAVEAT, repeated here because the table will be read on its own: these are the
published POLICIES on one common backbone, not the released checkpoints. Methods whose gains come
from fine-tuning (Chain-of-Spot, Visual CoT, DualFocus) are represented by their inference-time
mechanism only.
"""
import json
import math
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase47_prior_art.jsonl"
SWEEP = [(150, .466), (294, .565), (600, .660), (1176, .702),
         (2400, .754), (4760, .812), (7957, .848)]
FAMILY = {
    "uniform@300": "budget axis (1x)",
    "uniform@600": "budget axis (2x)",
    "uniform@1200": "budget axis (4x)",
    "grounding_crop": "Chain-of-Spot / Visual CoT / DualFocus family",
    "zoom_eye": "Zoom Eye (tree search)",
    "ours_attn@0.15": "ours, fixed policy",
    "ours_adaptive": "ours, peak-gated adaptive",
    "oracle": "ceiling (uses GT box)",
    "rand@0.15": "floor (random placement)",
}


def make_bar(anchors):
    """Build the compute-matched bar from the uniform arms measured IN THIS RUN, on THESE items.

    Using phase27's sweep would confound the comparison with item subset: phase27 covers all 191
    items and reads uniform@300 = 56.5%, while a partial or reordered subset here can read 49.0%
    for the very same arm. Every arm must be judged against what a uniform image of the same spend
    scores ON THE SAME ITEMS, so the anchors come from uniform@300/600/1200 in this file.

    Beyond the largest in-run anchor, the curve is extended using phase27's MEASURED increments in
    log-tokens (1176->2400: +5.2pp, 2400->4760: +5.8pp), i.e. only the SHAPE is borrowed and the
    LEVEL stays anchored to these items. Arms needing that extension are flagged in the table.
    """
    anchors = sorted(anchors)
    ext = []
    for (t0, a0), (t1, a1) in zip(SWEEP, SWEEP[1:]):
        ext.append((t0, t1, a1 - a0))

    def f(tokens):
        if tokens <= anchors[0][0]:
            return anchors[0][1]
        for (t0, a0), (t1, a1) in zip(anchors, anchors[1:]):
            if tokens <= t1:
                w = (math.log(tokens) - math.log(t0)) / (math.log(t1) - math.log(t0))
                return a0 + w * (a1 - a0)
        # past the last measured anchor: walk phase27's measured increments from there
        t, a = anchors[-1]
        for (s0, s1, d) in ext:
            if s1 <= t:
                continue
            lo = max(t, s0)
            if tokens <= lo:
                break
            hi = min(tokens, s1)
            a += d * (math.log(hi) - math.log(lo)) / (math.log(s1) - math.log(s0))
            if tokens <= s1:
                break
        return a
    return f, anchors[-1][0]


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 40:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    arms = [a for a in FAMILY if a in rows[0]["probs"]]

    # ours_adaptive is assembled from logged arms: fire on the top-40% `peak`, else keep uniform
    peaks = sorted((r["peak"] for r in rows if "peak" in r), reverse=True)
    tau = peaks[int(0.40 * len(peaks)) - 1] if peaks else float("inf")
    for r in rows:
        fired = r.get("peak", -1) >= tau
        r["probs"]["ours_adaptive"] = r["probs"]["ours_attn@0.15"] if fired else r["probs"]["uniform@300"]
        c = r["cost"]["ours_attn@0.15"] if fired else r["cost"]["uniform@300"]
        r["cost"]["ours_adaptive"] = c
    arms = [a for a in FAMILY if a in rows[0]["probs"]]

    anchors = []
    for a in ("uniform@300", "uniform@600", "uniform@1200"):
        if a in rows[0]["probs"]:
            anchors.append((st.mean(r["cost"][a]["tokens"] for r in rows),
                            st.mean(hit(r, a) for r in rows)))
    bar, last_anchor = make_bar(anchors)
    print(f"n = {n} V*Bench items, Qwen3-VL-2B. Every arm charged for every image token it uses.")
    print("Bar = uniform arms measured IN THIS RUN on THESE items: "
          + ", ".join(f"{t:.0f}tok->{100*a:.1f}%" for t, a in anchors))
    print(f"Beyond {last_anchor:.0f} tokens the curve is extended with phase27's MEASURED increments")
    print("(shape borrowed, level anchored here); arms needing that are marked EXT.\n")
    print(f"  {'arm':<17}{'family':<44}{'psses':>6}{'tokens':>8}{'acc':>8}"
          f"{'bar':>8}{'margin':>9}{'95% CI':>17}")
    res = {}
    for a in arms:
        v = [hit(r, a) for r in rows]
        ps = st.mean(r["cost"][a]["passes"] for r in rows)
        tk = st.mean(r["cost"][a]["tokens"] for r in rows)
        b = bar(tk)
        m = st.mean(v) - b
        lo, hi = boot([x - b for x in v])
        res[a] = (st.mean(v), ps, tk, b, m)
        note = "  (not a method)" if a == "oracle" else ("  EXT" if tk > last_anchor else "")
        print(f"  {a:<17}{FAMILY[a]:<44}{ps:>6.2f}{tk:>8.0f}{100*st.mean(v):>7.1f}%"
              f"{100*b:>7.1f}%{100*m:>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]{note}")

    print("\n=== HEAD-TO-HEAD: which policies beat the budget axis at their own spend? ===")
    win = [a for a in arms if a not in ("oracle", "rand@0.15") and not a.startswith("uniform")
           and res[a][4] > 0]
    lose = [a for a in arms if a not in ("oracle", "rand@0.15") and not a.startswith("uniform")
            and res[a][4] <= 0]
    for nm, g in [("BEAT the uniform bar", win), ("LOSE to the uniform bar", lose)]:
        print(f"  {nm}:")
        for a in g:
            print(f"     {a:<17}{100*res[a][4]:+6.1f}pp   at {res[a][2]:.0f} tokens "
                  f"/ {res[a][1]:.2f} passes")

    print("\n=== PAIRWISE, paired on the same items ===")
    for a, b_ in [("ours_adaptive", "grounding_crop"), ("ours_adaptive", "zoom_eye"),
                  ("ours_attn@0.15", "grounding_crop"), ("grounding_crop", "zoom_eye")]:
        if a not in res or b_ not in res:
            continue
        d = [hit(r, a) - hit(r, b_) for r in rows]
        lo, hi = boot(d)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {a:<17} - {b_:<17}{100*st.mean(d):>+7.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  {sig}"
              f"   ({res[a][2]:.0f} vs {res[b_][2]:.0f} tokens)")

    print("\n=== COST-NORMALISED: accuracy per 1000 image tokens ===")
    for a in sorted(arms, key=lambda x: -res[x][0] / max(res[x][2], 1)):
        print(f"  {a:<17}{1000*res[a][0]/max(res[a][2],1):>7.2f} acc/1k tok"
              f"   ({100*res[a][0]:.1f}% at {res[a][2]:.0f} tok)")

    print("\n=== DIAGNOSTIC: does the grounding arm actually find the target? ===")
    import re
    BB = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*"
                    r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")
    par = iou = 0
    ious = []
    for r in rows:
        m = BB.search(r.get("ground_raw", "") or "")
        if not m:
            continue
        par += 1
        x0, y0, x1, y1 = [float(x) / 1000 for x in m.groups()]
        g = r["gt_box_frac"]
        ix = max(0, min(g[2], x1) - max(g[0], x0))
        iy = max(0, min(g[3], y1) - max(g[1], y0))
        inter = ix * iy
        un = (x1 - x0) * (y1 - y0) + (g[2] - g[0]) * (g[3] - g[1]) - inter
        v = inter / un if un > 0 else 0
        ious.append(v)
        iou += v > 0.5
    if par:
        print(f"  parsed a box on {par}/{n} ({100*par/n:.0f}%);  median IoU with GT "
              f"{st.median(ious):.3f};  IoU>0.5 on {100*iou/par:.0f}%")
        print("  (a grounding policy that cannot localise is being charged 2 passes for nothing)")


if __name__ == "__main__":
    main()
