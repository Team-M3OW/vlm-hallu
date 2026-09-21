"""
Phase 206 (CPU): WHEN IS THE TARGET'S LOCATION LINEARLY DECODABLE FROM THE ANSWER POSITION?

Task 2 of the mechanist-interpretability request: a linear probe alongside the logit lens (phase 203) and the
attention read-out (§48/§51). Ridge probe from the last-position hidden state at every layer to the GT box
centre, OOF (KFold 5 x 3 seeds, alpha=1000), both models, n=191.

NOTE on the metric: placement coverage at the predicted centre is NOT the right score here -- V*Bench targets
are centred, so the constant predictor (the dataset-mean centre) already covers 0.956 and a noise-regularised
linear probe lands *below* it. The measured quantity is therefore the correlation between predicted and true
centres, with a label-shuffle control at the same layer.

Pre-registered shape (PREREG_MECH_THEORY.md): decodability should rise across the transport window and peak in
the read-out band; if it peaks early, the band's advantage has a different cause and we report that.
"""
import json, numpy as np
from sklearn.model_selection import KFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; W = 0.25; ALPHA = 1000.0
MAPS = {"qwen3": "phase30c_attn_maps_all.jsonl", "qwen2": "phase74_Qwen2_VL_7B_Instruct.jsonl"}
SEEDS = (700, 701, 702)


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def oof_pred(Xl, Y):
    n = len(Y); P = np.zeros((n, 2))
    for s in SEEDS:
        for tr, te in KFold(5, shuffle=True, random_state=s).split(Xl):
            mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-9
            Zt = np.c_[(Xl[tr] - mu) / sd, np.ones(len(tr))]
            A_ = Zt.T @ Zt + ALPHA * np.eye(Zt.shape[1]); A_[-1, -1] -= ALPHA
            Wt = np.linalg.solve(A_, Zt.T @ Y[tr])
            P[te] += np.c_[(Xl[te] - mu) / sd, np.ones(len(te))] @ Wt
    return P / len(SEEDS)


for which, fn in MAPS.items():
    z = np.load(f"{D}/data/phase109_probe_{which}.npz", allow_pickle=True)
    X = z["X"][:, 0].astype(np.float32); ids = [str(i) for i in z["ids"]]
    maps = {}
    for l in open(f"{D}/data/{fn}"):
        r = json.loads(l)
        if "attn" in r and "grid" in r: maps[r["question_id_full"]] = r
    keep = [i for i, q in enumerate(ids) if q in maps]
    X = X[keep]; ids = [ids[i] for i in keep]
    n, NL, Dd = X.shape
    gt = np.array([maps[q]["gt_box_frac"] for q in ids])
    Y = np.array([[(g[0] + g[2]) / 2, (g[1] + g[3]) / 2] for g in gt])
    mean_cov = np.mean([cov(Y[i, 0], Y[i, 1], gt[i]) for i in range(n)])
    rng = np.random.default_rng(206)
    print(f"\n================ {which}  n={n}  dim {Dd}  alpha={ALPHA} ================")
    print(f"constant mean-centre coverage {mean_cov:.3f} (the prior a probe must beat to place better)")
    print("layer :  probe r(pred,true)   shuffled-control r")
    rs, shuf = [], []
    for l in range(NL):
        P = oof_pred(X[:, l], Y)
        r = float(np.corrcoef(P.ravel(), Y.ravel())[0, 1]); rs.append(r)
        Ps = oof_pred(X[:, l], Y[rng.permutation(n)])
        shuf.append(float(np.corrcoef(Ps.ravel(), Y.ravel())[0, 1]))
    for l in range(NL):
        print(f"  L{l:2d} :   {rs[l]:+.3f}              {shuf[l]:+.3f}")
    best = int(np.argmax(rs))
    print(f"  peak r at L{best} ({rs[best]:+.3f}); inside window L0-15 max {max(rs[:16]):+.3f}; after L16+ max {max(rs[16:]):+.3f}")
    print(f"  shuffle control max |r| {max(abs(x) for x in shuf):.3f}")
    json.dump({"which": which, "n": n, "r": rs, "shuffle": shuf, "mean_centre_cov": float(mean_cov)},
              open(f"{D}/data/phase206_probe_loc_{which}.json", "w"), indent=1)
