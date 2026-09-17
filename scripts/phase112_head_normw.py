"""
Phase 112: give the head the signals that WORKED -- norm weighting and max aggregation.

Phase 104 found norm-weighted attention (Kobayashi et al., EMNLP'20: the attention weight is only
half of what determines the output; the other half is ||v||) beats raw attention by +4.7pp, and
phase 105 found MAX over layers (CLAA, 2602.16054) beats MEAN by +7.3pp. Together, +10.5pp.

The head has never seen either. It reads RAW per-layer attention and RAW within-layer ranks, and a
gradient-boosted tree splits on individual features -- it cannot construct a max across 28 columns,
and it cannot construct alpha*||v|| at all because the value norms were never given to it.

ARMS (all out-of-fold, GroupKFold grouped by item, identical folds, W=0.25)
    raw65        the deployed head: 28 raw + 28 ranks + nb + dep + 7 geometry   <- INCUMBENT
    +max         raw65 plus the max over the block and over all layers          <- gives it CLAA
    +nw          raw65 plus 28 norm-weighted layers and their ranks             <- gives it Kobayashi
    +nw+max      both, plus the norm-weighted maxima
    nw_only      the same shape as raw65 but built from norm-weighted attention

PRE-REGISTERED: the arm to beat is raw65 at W=0.25 (63.4% on Qwen3-VL). Any winner must clear zero
against it, and must then replicate on Qwen2-VL before it is claimed -- phase 104b for Qwen2 is
queued. No GPU: phase 104b already stored per-layer raw and norm-weighted maps from one forward pass.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
SRC = f"{D}/phase104b_locators_{WHICH}.jsonl"
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json",
        "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}[WHICH]
B0, B1 = (16, 27) if WHICH == "qwen3" else (15, 27)
W, COV_HIT = 0.25, 0.5
P70.W = W

props = json.load(open(PROP))
rows = [json.loads(l) for l in open(SRC)]
rows = [r for r in rows if r["question_id_full"] in props]
NL = len(rows[0]["raw_layers"])
print(f"{WHICH}: {len(rows)} items, {NL} layers, block L{B0}-{B1-1}, W={W}")


def geom_block(dep, gh, gw):
    pad = np.pad(dep.reshape(gh, gw), 1, mode="edge")
    nb = sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3)) / 9.0
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = (yy+.5)/gh, (xx+.5)/gw
    return np.concatenate([
        nb.reshape(-1, 1), dep.reshape(-1, 1), fx.reshape(-1, 1), fy.reshape(-1, 1),
        np.sqrt((fx-.5)**2 + (fy-.5)**2).reshape(-1, 1),
        np.minimum(np.minimum(fx, 1-fx), np.minimum(fy, 1-fy)).reshape(-1, 1),
        (xx == gw-1).astype(float).reshape(-1, 1), (yy == gh-1).astype(float).reshape(-1, 1),
        (xx == 0).astype(float).reshape(-1, 1)], axis=1)


def ranks(A):
    return np.argsort(np.argsort(-A, axis=1), axis=1) / max(A.shape[1]-1, 1)


blocks, Y, G, DEP, RING = {k: [] for k in ["raw65", "maxf", "nwf", "nwmaxf", "nw65"]}, [], [], [], []
for gi, r in enumerate(rows):
    gh, gw = r["grid"]; n = r["n_img"]
    RA = np.asarray(r["raw_layers"], float); NA = np.asarray(r["nw_layers"], float)
    gt = props[r["question_id_full"]]["gt_box_frac"]
    dep, ndep = RA[B0:B1].mean(0), NA[B0:B1].mean(0)
    blocks["raw65"].append(np.concatenate([RA.T, ranks(RA).T, geom_block(dep, gh, gw)], 1))
    blocks["maxf"].append(np.stack([RA[B0:B1].max(0), RA.max(0), RA[B0:B1].max(0)-dep], 1))
    blocks["nwf"].append(np.concatenate([NA.T, ranks(NA).T, ndep.reshape(-1, 1)], 1))
    blocks["nwmaxf"].append(np.stack([NA[B0:B1].max(0), NA.max(0), NA[B0:B1].max(0)-ndep], 1))
    blocks["nw65"].append(np.concatenate([NA.T, ranks(NA).T, geom_block(ndep, gh, gw)], 1))
    yy, xx = np.mgrid[0:gh, 0:gw]
    fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
    Y.append(np.array([P70.coverage(float(fx[i]), float(fy[i]), gt) for i in range(n)]))
    G.append(np.full(n, gi)); DEP.append(dep)
    m = np.zeros((gh, gw), bool)
    if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
    else: m[:] = True
    RING.append(m.ravel())
B = {k: np.vstack(v) for k, v in blocks.items()}
Y, G = np.concatenate(Y), np.concatenate(G)

ARMS = {"raw65 (INCUMBENT)": ["raw65"], "+max": ["raw65", "maxf"], "+nw": ["raw65", "nwf"],
        "+nw+max": ["raw65", "nwf", "maxf", "nwmaxf"], "nw_only": ["nw65"]}
res = {}
for name, parts in ARMS.items():
    X = np.hstack([B[p] for p in parts])
    P = P70.oof(X, Y, G, seeds=3)
    res[name] = np.array([float(Y[G == gi][int(np.argmax(np.where(RING[gi], P[G == gi], -1e9)))] >= COV_HIT)
                          for gi in np.unique(G)])
    print(f"\r  {name:>18} {X.shape[1]:4d} feats  {res[name].mean()*100:5.1f}%", flush=True)
base = res["raw65 (INCUMBENT)"]
rng = np.random.default_rng(112); n = len(base)
print(f"\n  {'arm':>18} {'cov':>7}   vs incumbent")
for k, v in res.items():
    d = v - base
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(6000)])
    lo, hi = np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
    print(f"  {k:>18} {v.mean()*100:6.1f}%   {d.mean()*100:+6.1f} [{lo:+5.1f},{hi:+5.1f}] "
          f"{'CLEARS' if lo > 0 else ('WORSE' if hi < 0 else '')}")
