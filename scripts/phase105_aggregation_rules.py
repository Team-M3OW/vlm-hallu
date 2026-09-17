"""
Phase 105: CLAA's aggregation rule, applied to VLM localisation.

McDanel, Li & Khaitan, "CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill"
(arXiv 2602.16054, Feb 2026) diagnose exactly our depth problem on the LLM side:

  - they build an Answer-Informed Oracle (true token importance = attention from the GENERATED
    ANSWER back to the prompt) and use it to score ranking heuristics layer by layer;
  - they find "layer-wise ranking instability": rankings "degrade sharply at specific layers, a
    failure mode invisible to end-to-end benchmarks", and "early layers consistently show unreliable
    token rankings ... particularly layers 0-4";
  - their fix is "do not trust any single layer" -- aggregate across a window of consecutive layers,
    and crucially they aggregate by **MAX**, not mean, "to preserve tokens deemed important by any
    recent layer while filtering layer-specific noise", then 1D average-pool (kernel 7) to stabilise.

Our deployed read-out takes the **MEAN** over L16-26. Phase 91 already found that all layers beat any
contiguous block, i.e. aggregation breadth helps -- but the RULE has never been varied. CLAA says the
rule matters as much as the range, and gives a specific alternative with an independent justification.

This runs entirely on disk. Arms, all scored on the deployed metric (top-1 coverage, ring-masked):

    mean_block      mean over the deployed block                  <- THE INCUMBENT
    mean_all        mean over all layers                          <- known-bad control
    max_block       MAX over the deployed block                   <- CLAA's rule, our range
    max_all         MAX over all layers
    max_win4        max over the best 4-layer consecutive window  <- CLAA as specified (n=4)
    max_win4_pool   the same, then 3x3 average pooling            <- CLAA's stabilisation step
    topk_mean       mean of each cell's top-4 layer values        <- soft max, between the two
    median_block    median over the block                         <- robustness control
    mean_block_pool mean over the block, then 3x3 pooling         <- isolates pooling from the rule

The window position for max_win4 is chosen OUT-OF-FOLD by item, never on the evaluation fold.
"""
import json, sys
import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
CFG = {"qwen3": (f"{D}/phase30c_attn_maps_all.jsonl", 28, (16, 27)),
       "qwen2": (f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", 28, (15, 27))}
W, COV_HIT, NW = 0.25, 0.5, 4


def coverage(cx, cy, gt, w=W):
    x0, x1, y0, y1 = cx - w/2, cx + w/2, cy - w/2, cy + w/2
    if x0 < 0: x0, x1 = 0.0, w
    if y0 < 0: y0, y1 = 0.0, w
    if x1 > 1: x0, x1 = 1-w, 1.0
    if y1 > 1: y0, y1 = 1-w, 1.0
    gx0, gy0, gx1, gy1 = gt
    inter = max(0.0, min(gx1,x1)-max(gx0,x0)) * max(0.0, min(gy1,y1)-max(gy0,y0))
    return inter / max((gx1-gx0)*(gy1-gy0), 1e-12)


def pool3(m, gh, gw):
    g = m.reshape(gh, gw); p = np.pad(g, 1, mode="edge")
    return (sum(p[i:i+gh, j:j+gw] for i in range(3) for j in range(3)) / 9.0).ravel()


for which in ["qwen3", "qwen2"]:
    src, NL, (b0, b1) = CFG[which]
    rows = [json.loads(l) for l in open(src)]
    items = []
    for r in rows:
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
        cov = np.array([coverage(float(fx[i]), float(fy[i]), r["gt_box_frac"]) for i in range(n)])
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        items.append((A, cov, rm.ravel(), gh, gw))
    N = len(items)

    def score(fn):
        return np.array([float(c[int(np.argmax(np.where(rmv, fn(A, gh, gw), -1e9)))] >= COV_HIT)
                         for A, c, rmv, gh, gw in items])

    arms = {
        "mean_block":      lambda A, gh, gw: A[b0:b1].mean(0),
        "mean_all":        lambda A, gh, gw: A.mean(0),
        "max_block":       lambda A, gh, gw: A[b0:b1].max(0),
        "max_all":         lambda A, gh, gw: A.max(0),
        "topk_mean":       lambda A, gh, gw: np.sort(A, 0)[-NW:].mean(0),
        "median_block":    lambda A, gh, gw: np.median(A[b0:b1], 0),
        "mean_block_pool": lambda A, gh, gw: pool3(A[b0:b1].mean(0), gh, gw),
    }
    print(f"\n=== {which}: {N} items, layers {NL}, block L{b0}-{b1-1}, W={W} ===")
    res = {k: score(f) for k, f in arms.items()}

    # CLAA as specified: max over a 4-layer consecutive window, window chosen OUT-OF-FOLD
    K = 5
    order = np.arange(N); np.random.default_rng(7).shuffle(order)
    fold = np.zeros(N, int)
    for i, o in enumerate(order): fold[o] = i % K
    for tag, post in [("max_win4", False), ("max_win4_pool", True)]:
        got = np.zeros(N)
        for k in range(K):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            best, bs = None, -1
            for s in range(NL - NW + 1):
                def f(A, gh, gw, s=s, post=post):
                    m = A[s:s+NW].max(0)
                    return pool3(m, gh, gw) if post else m
                acc = np.mean([float(items[i][1][int(np.argmax(np.where(items[i][2],
                               f(items[i][0], items[i][3], items[i][4]), -1e9)))] >= COV_HIT)
                               for i in tr])
                if acc > bs: bs, best = acc, s
            for i in te:
                A, c, rmv, gh, gw = items[i]
                m = A[best:best+NW].max(0)
                if post: m = pool3(m, gh, gw)
                got[i] = float(c[int(np.argmax(np.where(rmv, m, -1e9)))] >= COV_HIT)
        res[tag] = got

    base = res["mean_block"]
    rng = np.random.default_rng(105)
    print(f"  {'arm':>17} {'cov':>7}   vs mean_block")
    for k in ["mean_block", "mean_all", "max_block", "max_all", "max_win4", "max_win4_pool",
              "topk_mean", "median_block", "mean_block_pool"]:
        d = res[k] - base
        b = np.array([d[rng.integers(0, N, N)].mean() for _ in range(4000)])
        lo, hi = np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
        tag = "CLEARS" if lo > 0 else ("WORSE" if hi < 0 else "")
        print(f"  {k:>17} {res[k].mean()*100:6.1f}%   {d.mean()*100:+6.1f} [{lo:+5.1f},{hi:+5.1f}] {tag}")
