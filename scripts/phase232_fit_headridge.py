"""
Phase 232 (CPU): fit the head-augmented ridge for deployment.

Feature set from phase 231's winner (AH+geo, cov25 label):
   LA  28  log of the head-mean attention map (deployed read-out)
   HM  28  log of the per-cell MAX over heads (per-head maps normalised over cells first)
   HS  28  per-cell STD over heads
   geo  7  log 3x3 neighbourhood of the block-mean map, centre/edge/position/sink flags
= 91 features + intercept. Fitted on ALL V*Bench items (every grid), ridge alpha=1,
train-fold standardisation saved with the weights.

Saves data/fig_ridge_w_head_{qwen3,qwen2}.npz  (w, mu, sd) and prints the training coverage.
"""
import json, os, numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; W = 0.25
MAPS = {"qwen3": ("phase30c_attn_maps_all.jsonl", (16, 27), "phase170_perhead_qwen3.npy", "phase170_perhead_qwen3_index.json"),
        "qwen2": ("phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27), "phase170_perhead_qwen2.npy", "phase170_perhead_qwen2_index.json")}


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


for which, (fn, BLK, phf, phidx) in MAPS.items():
    rows = [json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows = [r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    PH = np.load(f"{D}/data/{phf}", mmap_mode="r"); pidx = json.load(open(f"{D}/data/{phidx}"))
    ppos = {it["question_id_full"]: (it["offset"], it["n_cells"]) for it in pidx["items"]}
    H = pidx["n_heads"]; NL = 28
    Xs, Ys, Gs, valid = [], [], [], []
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]; nc = gh * gw
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
        A = np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        off, ncl = ppos[r["question_id_full"]]
        if ncl != nc:                      # 1 item per model has a grid mismatch between dumps
            continue
        P = np.asarray(PH[:, off:off + ncl], np.float32).reshape(NL, H, ncl)
        Ps = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
        LA = np.log(A + 1e-12).T
        HM = np.log(Ps.max(1) + 1e-12).T
        HS = Ps.std(1).T
        dep = A[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
        nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
        geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                    np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                    (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
        Xs.append(np.c_[LA, HM, HS, geo])
        Ys.append(np.array([cov(float(fx[c]), float(fy[c]), r["gt_box_frac"]) for c in range(nc)]))
        Gs.append(np.full(nc, gi)); valid.append((gh, gw, r["gt_box_frac"]))
    X = np.vstack(Xs); Y = np.concatenate(Ys)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xt = np.c_[(X - mu) / sd, np.ones(len(X))]
    A_ = Xt.T @ Xt + np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
    w = np.linalg.solve(A_, Xt.T @ Y)
    # ring-masked argmax coverage (in-sample, diagnostic only)
    P = (np.c_[(X - mu) / sd, np.ones(len(X))] @ w)
    ccs = []
    k = 0
    for gh, gw, gt in valid:
        nc = gh * gw
        Pi = P[k:k + nc].reshape(gh, gw); k += nc
        rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True
        j = np.argmax(np.where(rm, Pi, -1e9))
        cy, cx = np.unravel_index(j, (gh, gw))
        ccs.append(cov((cx + .5) / gw, (cy + .5) / gh, gt))
    np.savez(f"{D}/data/fig_ridge_w_head_{which}.npz", w=w, mu=mu, sd=sd)
    print(f"{which}: {X.shape[1]} features, in-sample coverage {np.mean(ccs):.3f} ({100*np.mean(np.array(ccs)>=.5):.1f}%) -> fig_ridge_w_head_{which}.npz")
