"""Phase 121 analysis: OOF head + deployed argmax coverage per localisation-prompt variant, both models."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
P70.W = 0.25
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
NL = 28

def build(recs, vn, b0, b1):
    X, Y, G, DEP, RING = [], [], [], [], []
    for gi, r in enumerate(recs):
        v = r["variants"][vn]; gh, gw = v["grid"]; n = v["n_img_tokens"]
        A = np.stack([np.asarray(v["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = A.reshape(NL, gh, gw); dep = M[b0:b1].mean(0)
        R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n-1, 1)).reshape(NL, gh, gw)
        pad = np.pad(dep, 1, mode="edge"); nb = sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3))/9.0
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = (yy+.5)/gh, (xx+.5)/gw
        X.append(np.concatenate([M.reshape(NL, -1).T, R.reshape(NL, -1).T, nb.reshape(-1, 1), dep.reshape(-1, 1),
            fx.reshape(-1, 1), fy.reshape(-1, 1), np.sqrt((fx-.5)**2+(fy-.5)**2).reshape(-1, 1),
            np.minimum(np.minimum(fx, 1-fx), np.minimum(fy, 1-fy)).reshape(-1, 1),
            (xx == gw-1).astype(float).reshape(-1, 1), (yy == gh-1).astype(float).reshape(-1, 1),
            (xx == 0).astype(float).reshape(-1, 1)], axis=1))
        Y.append(np.array([P70.coverage(float(fx.flat[i]), float(fy.flat[i]), r["gt_box_frac"]) for i in range(n)]))
        G.append(np.full(n, gi)); DEP.append(dep.ravel())
        m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        RING.append(m.ravel())
    return np.vstack(X), np.concatenate(Y), np.concatenate(G), DEP, RING

def top1(P, Y, G, RING):
    return np.array([float(Y[G == gi][int(np.argmax(np.where(RING[gi], P[G == gi], -1e9)))] >= P70.COV_HIT) for gi in np.unique(G)])

for which, b0, b1 in [("qwen3", 16, 27), ("qwen2", 15, 27)]:
    p = f"{D}/phase121_prompts_{which}.jsonl"
    try: recs = [json.loads(l) for l in open(p)]
    except FileNotFoundError: print(f"{which}: no data"); continue
    vns = list(recs[0]["variants"].keys()); n = len(recs)
    res_h, res_a = {}, {}
    for vn in vns:
        X, Y, G, DEP, RING = build(recs, vn, b0, b1)
        res_h[vn] = top1(P70.oof(X, Y, G, seeds=3), Y, G, RING)
        res_a[vn] = top1(np.concatenate(DEP), Y, G, RING)
    rng = np.random.default_rng(121)
    def ci(d):
        m = len(d); b = np.array([d[rng.integers(0, m, m)].mean() for _ in range(6000)])
        return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
    print(f"\n{which}: n={n}  (W=0.25, top-1 coverage)")
    print(f"  {'variant':>11} {'argmax':>7} {'head OOF':>9}   head vs V0            argmax vs V0")
    for vn in vns:
        a, lo, hi = ci(res_h[vn] - res_h["V0_full"]); a2, lo2, hi2 = ci(res_a[vn] - res_a["V0_full"])
        print(f"  {vn:>11} {res_a[vn].mean()*100:6.1f}% {res_h[vn].mean()*100:8.1f}%   {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{' CLEARS' if lo>0 else ''}   {a2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]{' CLEARS' if lo2>0 else ''}")
