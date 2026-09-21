"""
Phase 185: OPTIMISE THE HEAD SCORE FOR THE VERIFICATION ROLE, not the grounding role.

VRH selects heads by attention mass on the GROUND-TRUTH referent. But §20H/§20I use those heads for a
different job: judging whether DPR's proposed window deserves the crop. Nobody has selected heads FOR
that job. Five rules, same verifier function throughout (score = mass inside the DPR window, summed
over the selected heads); only the SELECTION changes.

    vrh       S_h = E_s[ mass on the GT referent ]                    <- the original, reference
    dvrh_z    S_h = (mu+ - mu-) / sqrt((var+ + var-)/2)  on window mass, split by y
    auc_dvrh  S_h = AUROC( window mass , y )                          <- rank-based twin of dvrh_z
    margin    S_h = E[D|y=1] - E[D|y=0],  D = m_(1) - m_(2) over the top-3 NMS regions of the map
    entvrh    S_h = E_s[ window mass ] / (E_s[ spatial entropy ] + eps)   <- LABEL-FREE

TWO DEFINITIONS OF y, both reported:
    y_cov  the DPR window really covers the evidence        (needs GT boxes -- same cost as today)
    y_acc  the DPR crop arm answered correctly              (needs ONLY answer labels -- NO BOXES)
y_acc matters: if it works, head selection becomes box-free and the annotation question disappears.

FOLD HONESTY IS THE WHOLE EXPERIMENT. dvrh/auc/margin select heads USING y and are then scored on
predicting y. Selection happens on TRAINING folds only (GroupKFold(5), phase-70 convention). Phase
108's precedent: scoring heads on each item's own box read 60.7% and collapsed to 58.1% when fixed.
Hard top-K only, no head weighting -- at 448-784 heads and n=191, weighting manufactures results.

PRE-REGISTERED: verification AUROC is the SELECTOR. Exactly ONE configuration is promoted to the gate,
chosen on AUROC BEFORE its gate number is looked at. That keeps §20I's contrast a confirmation rather
than a search. CPU only.
"""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, K_REG, NMS = 0.25, 3, 3
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
OUTC = {"qwen3": f"{D}/phase78_w_sweep.jsonl", "qwen2": f"{D}/phase97m_merged_qwen2vl.jsonl"}
rng = np.random.default_rng(185)
EPS = 1e-9


def auroc(s, p):
    p = np.asarray(p, bool)
    if p.all() or not p.any(): return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = p.sum(), (~p).sum()
    return (r[p].sum() - n1*(n1+1)/2) / (n1*n0)


def auroc_cols(M, y):
    """AUROC of every column of M (n x k) against y, vectorised."""
    y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.zeros(M.shape[1])
    R = np.argsort(np.argsort(M, axis=0), axis=0) + 1.0
    return (R[y].sum(0) - n1*(n1+1)/2) / (n1*n0)


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    out = {r["question_id_full"]: r for r in (json.loads(l) for l in open(OUTC[which]))}
    idx = [e for e in idx if e["question_id_full"] in props and e["question_id_full"] in out]
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)

    m = np.zeros((N, NL, H), np.float32)     # mass in the DPR window
    g = np.zeros((N, NL, H), np.float32)     # mass on the GT referent
    ent = np.zeros((N, NL, H), np.float32)   # spatial entropy
    marg = np.zeros((N, NL, H), np.float32)  # top-1 minus top-2 region mass
    y_cov = np.zeros(N, bool); y_acc = np.zeros(N, bool)
    WIN = []
    for i, e in enumerate(idx):
        q = e["question_id_full"]; p = props[q]; r = out[q]
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        cx, cy = p["head"]
        x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
        win = (fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W)
        gt = p["gt_box_frac"]
        ref = (fx >= gt[0]) & (fx <= gt[2]) & (fy >= gt[1]) & (fy <= gt[3])
        A = imap(e)
        m[i] = A[:, :, win].sum(-1)
        g[i] = A[:, :, ref].sum(-1) if ref.any() else 0.0
        P = np.clip(A, EPS, None)
        ent[i] = -(P * np.log(P)).sum(-1)
        # candidate regions: top-K NMS peaks of the all-head gated-max map
        s = A.mean(1)[GATE[which]].max(0)
        order = np.argsort(-s); picked = []
        for c in order:
            if len(picked) >= K_REG: break
            cy_, cx_ = divmod(int(c), gw)
            if all(abs(cy_-a) + abs(cx_-b) >= NMS for a, b in picked): picked.append((cy_, cx_))
        masses = []
        for (a, b) in picked:
            rx, ry = (b+.5)/gw, (a+.5)/gh
            rx0 = min(max(0, rx-W/2), 1-W); ry0 = min(max(0, ry-W/2), 1-W)
            sel = (fx >= rx0) & (fx <= rx0+W) & (fy >= ry0) & (fy <= ry0+W)
            masses.append(A[:, :, sel].sum(-1))
        Mst = np.sort(np.stack(masses), axis=0)[::-1] if len(masses) >= 2 else None
        marg[i] = (Mst[0] - Mst[1]) if Mst is not None else 0.0
        WIN.append(win)
        y_cov[i] = p["head_cov"] >= 0.5
        lab = r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
        y_acc[i] = int(np.argmax(r["probs"]["head@0.25"]) == lab)

    k = max(1, int(round(0.25*H)))
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))

    def topk(sc):
        msk = np.zeros((NL, H), bool)
        for l in range(NL): msk[l, np.argsort(-sc[l])[:k]] = True
        return msk

    def score_rule(rule, tr, y):
        if rule == "vrh": return g[tr].mean(0)
        X = m[tr].reshape(len(tr), -1)
        yy_ = y[tr]
        if rule == "dvrh_z":
            a, b = X[yy_], X[~yy_]
            if len(a) < 2 or len(b) < 2: return np.zeros(NL*H).reshape(NL, H)
            s = (a.mean(0)-b.mean(0)) / (np.sqrt((a.var(0)+b.var(0))/2)+EPS)
            return s.reshape(NL, H)
        if rule == "auc_dvrh": return auroc_cols(X, yy_).reshape(NL, H)
        if rule == "margin":
            Z = marg[tr].reshape(len(tr), -1)
            a, b = Z[yy_], Z[~yy_]
            if len(a) < 2 or len(b) < 2: return np.zeros((NL, H))
            return (a.mean(0)-b.mean(0)).reshape(NL, H)
        if rule == "entvrh":
            return (m[tr].mean(0)) / (ent[tr].mean(0) + EPS)
        raise ValueError(rule)

    RULES = ["vrh", "dvrh_z", "auc_dvrh", "margin", "entvrh"]
    print(f"\n{'='*92}\n{which}  n={N}  {NL}x{H} heads, top-{k}/layer, fold-honest selection")
    print(f"  base rates: y_cov {100*y_cov.mean():.1f}%   y_acc {100*y_acc.mean():.1f}%")
    res = {}
    for yname, y in [("y_cov (needs boxes)", y_cov), ("y_acc (NO boxes)", y_acc)]:
        print(f"  --- selection target: {yname}")
        for rule in RULES:
            V = np.zeros(N)
            for tr, te in folds:
                msk = topk(score_rule(rule, tr, y))
                for i in te:
                    A = imap(idx[i])
                    sel = (A*msk[:, :, None]).sum(1)/np.maximum(msk.sum(1)[:, None], 1)
                    gm = sel[GATE[which]].mean(0)
                    V[i] = gm[WIN[i]].sum()/max(gm.sum(), EPS)
            a_cov, a_acc = auroc(V, y_cov), auroc(V, y_acc)
            res[(yname, rule)] = (V, a_cov, a_acc)
            tag = "  (label-free)" if rule in ("vrh", "entvrh") and "NO boxes" in yname else ""
            print(f"      {rule:10} AUROC vs y_cov {a_cov:.3f} | vs y_acc {a_acc:.3f}{tag}")
        base = res[(yname, "vrh")][0]
        for rule in RULES[1:]:
            Vr = res[(yname, rule)][0]
            d = auroc(Vr, y_cov) - auroc(base, y_cov)
            b = []
            for _ in range(4000):
                i = rng.integers(0, N, N)
                x = auroc(Vr[i], y_cov[i]) - auroc(base[i], y_cov[i])
                if not np.isnan(x): b.append(x)
            lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
            print(f"      {rule:10} ΔAUROC(y_cov) vs original VRH {d:+.3f} [{lo:+.3f},{hi:+.3f}]"
                  f"{'  BETTER' if lo > 0 else ''}")
    np.save(f"{D}/phase185_scores_{which}.npy",
            np.stack([res[(yn, r)][0] for yn in ["y_cov (needs boxes)", "y_acc (NO boxes)"]
                      for r in RULES]))
