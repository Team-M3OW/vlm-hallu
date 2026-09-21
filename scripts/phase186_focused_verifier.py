"""
Phase 186: ENTROPY-PENALISED VERIFIER — the one non-monotone variant.

§20L: the verifier is rank-invariant to which heads are selected, so head-score normalisations cannot
move it. And AUROC + a median threshold are both RANK-based, so any strictly monotone transform of the
verifier score (log-odds, key-set-size normalisation, lift, and — since our window is a fixed 6.25% of
image area so |K| is near-constant — the contrastive form) is provably inert too.

`Θ_focused = V·(1 − H/log n)` is the exception: H varies per item independently of where the mass sits,
so it is not monotone in V. It also fuses the two signals that independently work — window mass
(AUROC 0.762/0.755, §20H) and spatial concentration (0.638/0.636, the §18E routing signal).

ARMS (head set = the original VRH selection, fold-honest, unchanged)
    V          mass inside the DPR window                       <- the incumbent verifier
    H_only     1 − H/log n  (concentration alone)               <- the §18E signal, reference
    focused    V·(1 − H/log n)                                  <- THE TEST
    ratio      V/(H+eps)                                        <- secondary form
PRE-REGISTERED: `focused` − `V` paired ΔAUROC on y_cov must clear zero on BOTH models. Only then is it
taken to the gate; the gate number is not looked at first. CPU only.
"""
import json
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, EPS = 0.25, 1e-12
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
OUTC = {"qwen3": f"{D}/phase78_w_sweep.jsonl", "qwen2": f"{D}/phase97m_merged_qwen2vl.jsonl"}
rng = np.random.default_rng(186)


def auroc(s, p):
    p = np.asarray(p, bool)
    if p.all() or not p.any(): return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = p.sum(), (~p).sum()
    return (r[p].sum() - n1*(n1+1)/2) / (n1*n0)


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H_, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    out = {r["question_id_full"]: r for r in (json.loads(l) for l in open(OUTC[which]))}
    idx = [e for e in idx if e["question_id_full"] in props and e["question_id_full"] in out]
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H_, -1)

    WIN, REF, y_cov, y_acc = [], [], np.zeros(N, bool), np.zeros(N, bool)
    for i, e in enumerate(idx):
        q = e["question_id_full"]; p = props[q]; r = out[q]
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        cx, cy = p["head"]
        x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
        WIN.append((fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W))
        g = p["gt_box_frac"]
        REF.append((fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3]))
        y_cov[i] = p["head_cov"] >= 0.5
        lab = r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
        y_acc[i] = int(np.argmax(r["probs"]["head@0.25"]) == lab)

    # original VRH head set, fold-honest
    ref_mass = np.zeros((N, NL, H_), np.float32)
    for i, e in enumerate(idx):
        A = imap(e)
        ref_mass[i] = A[:, :, REF[i]].sum(-1) if REF[i].any() else 0.0
    k = max(1, int(round(0.25*H_)))
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))
    V = np.zeros(N); Hn = np.zeros(N)
    for tr, te in folds:
        sc = ref_mass[tr].mean(0)
        msk = np.zeros((NL, H_), bool)
        for l in range(NL): msk[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            A = imap(idx[i])
            sel = (A*msk[:, :, None]).sum(1)/np.maximum(msk.sum(1)[:, None], 1)
            gm = sel[GATE[which]].mean(0)
            gm = gm/max(gm.sum(), EPS)
            V[i] = gm[WIN[i]].sum()
            P = np.clip(gm, EPS, None)
            Hn[i] = -(P*np.log(P)).sum()/np.log(len(gm))          # normalised entropy in [0,1]
    arms = {"V (incumbent)": V, "H_only (1-H/log n)": 1-Hn,
            "focused = V*(1-H/log n)": V*(1-Hn), "ratio = V/H": V/(Hn+1e-9)}
    print(f"\n{'='*84}\n{which}  n={N}   head set = original VRH, fold-honest")
    print(f"  mean V {V.mean():.4f} | mean normalised entropy {Hn.mean():.4f} "
          f"(sd {Hn.std():.4f})  | corr(V, 1-H) = {np.corrcoef(V, 1-Hn)[0,1]:+.3f}")
    for nm, s in arms.items():
        print(f"    {nm:26} AUROC y_cov {auroc(s, y_cov):.3f} | y_acc {auroc(s, y_acc):.3f}")
    for nm in list(arms)[1:]:
        d = auroc(arms[nm], y_cov) - auroc(V, y_cov)
        b = []
        for _ in range(4000):
            i = rng.integers(0, N, N)
            x = auroc(arms[nm][i], y_cov[i]) - auroc(V[i], y_cov[i])
            if not np.isnan(x): b.append(x)
        lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
        print(f"    {nm:26} ΔAUROC vs V {d:+.3f} [{lo:+.3f},{hi:+.3f}]{'  CLEARS' if lo > 0 else ''}")
