"""
Phase 110: THE THIRD ARCHITECTURE FAMILY. Train the re-ranking head on LLaVA.

The method's end-task win is currently two Qwen checkpoints -- one vendor, one tokenisation scheme.
Phase 82 already extracted per-layer attention for LLaVA-NeXT (32 layers, 552 visual tokens) and
LLaVA-OneVision (28 layers, 540), with `image_newline` separators stripped and the grid rebuilt from
the modal row length. So the head can be fitted here with NO GPU at all, and only the end task needs
one.

WHAT WOULD REJECT IT. SS14N found the read-out defect holds on LLaVA-NeXT (+6.3pp) but is a NULL on
LLaVA-OneVision (+1.0pp [-2.6,+5.2]), and that absolute localisation is far worse on both
(12.6% / 25.1% against Qwen3-VL's 39.3%). If the head cannot raise coverage on a family whose
proposal signal is 2-3x weaker, the method is Qwen-specific and the paper must say so.

Everything is phase 70's machinery verbatim -- same features, same out-of-fold GroupKFold grouped by
item, same negative subsampling, same ring mask -- pointed at a different attention file, so the
three families' heads cannot drift apart in implementation.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, COV_HIT = 0.25, 0.5
CFG = {"LLaVA-NeXT-7B": (f"{D}/phase82_llavanext.jsonl", 32),
       "LLaVA-OneVision-7B": (f"{D}/phase82_onevision.jsonl", 28)}


def build(src, NL, b0, b1):
    rows = [json.loads(l) for l in open(src)]
    X, Y, G, DEP, RING = [], [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]; n = r["n_img_tokens"]
        if gh * gw != n:                      # separator bookkeeping must be exact
            continue
        A = np.stack([np.asarray(r["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A.reshape(NL, gh, gw)
        dep = M[b0:b1].mean(0)
        R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)).reshape(NL, gh, gw)
        pad = np.pad(dep, 1, mode="edge")
        nb = sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3)) / 9.0
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = (yy + .5) / gh, (xx + .5) / gw
        X.append(np.concatenate([
            M.reshape(NL, -1).T, R.reshape(NL, -1).T,
            nb.reshape(-1, 1), dep.reshape(-1, 1),
            fx.reshape(-1, 1), fy.reshape(-1, 1),
            np.sqrt((fx-.5)**2 + (fy-.5)**2).reshape(-1, 1),
            np.minimum(np.minimum(fx, 1-fx), np.minimum(fy, 1-fy)).reshape(-1, 1),
            (xx == gw-1).astype(float).reshape(-1, 1),
            (yy == gh-1).astype(float).reshape(-1, 1),
            (xx == 0).astype(float).reshape(-1, 1)], axis=1))
        P70.W = W
        Y.append(np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"])
                           for i in range(n)]))
        G.append(np.full(n, gi)); DEP.append(dep.flatten())
        m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        RING.append(m.ravel())
    return (np.vstack(X), np.concatenate(Y), np.concatenate(G), DEP, RING, rows)


for name, (src, NL) in CFG.items():
    b0, b1 = int(.57 * NL), int(.93 * NL) + 1        # same fraction of the stack as Qwen's L16-26
    X, Y, G, DEP, RING, rows = build(src, NL, b0, b1)
    nI = len(np.unique(G))
    P = P70.oof(X, Y, G, seeds=3)
    hit, cov, dep_hit, dep_cov = P70.evaluate(P, G, Y, DEP, RING)
    # ceiling: does ANY ring-masked cell cover?
    ceil = np.mean([float(Y[G == gi].max() >= COV_HIT) for gi in np.unique(G)])
    d = np.array([float(Y[G == gi][int(np.argmax(np.where(RING[gi], P[G == gi], -1e9)))] >= COV_HIT)
                  for gi in np.unique(G)])
    b = np.array([float(Y[G == gi][int(np.argmax(np.where(RING[gi], DEP[gi], -1e9)))] >= COV_HIT)
                  for gi in np.unique(G)])
    rng = np.random.default_rng(110); n = len(d)
    bs = np.array([(d - b)[rng.integers(0, n, n)].mean() for _ in range(6000)])
    print(f"\n{name}: {nI} items, block L{b0}-{b1-1} of {NL}, W={W}")
    print(f"   deployed argmax (block mean) {b.mean()*100:5.1f}%")
    print(f"   learned head (OOF)           {d.mean()*100:5.1f}%")
    print(f"   head - argmax  {(d-b).mean()*100:+5.1f}pp "
          f"[{np.percentile(bs,2.5)*100:+5.1f},{np.percentile(bs,97.5)*100:+5.1f}]"
          f"{'  CLEARS' if np.percentile(bs,2.5) > 0 else ''}")
    print(f"   ceiling (some cell covers)   {ceil*100:5.1f}%")
