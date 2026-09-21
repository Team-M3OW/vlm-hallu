"""
Phase 182: does a BETTER VERIFIER make a BETTER GATE?

§20H: VRH mass inside the DPR window predicts whether that window really contains the evidence at
AUROC 0.762 / 0.755, beating the naive all-head mass on both models. §18E already shows a gate built
on a WEAKER signal (map dispersion, AUROC 0.636) clears pooled on both models at +5.8 / +5.8. So the
question is whether the stronger signal converts.

COMPOSITION (phase 150/156 convention, so rows compare line for line)
    gated ON  -> DPR: localise@300 + crop@300      gated OFF -> uniform@600 (the bar)
    scored against uniform@600 on every item; tokens matched by construction.

THRESHOLD: the training-fold MEDIAN, out-of-fold, no free parameter. §20D is the reason -- a τ fitted
to maximise training-fold accuracy overfit at n=191 and routed 71%, while §18E's untuned median
cleared on both models.

ARMS
    R0  always DPR                                  (Table 1's pooled row)
    R_disp   crop iff dispersion >= fold median     (§18E reproduction, the incumbent gate)
    R_vrh    crop iff VRH mass in window >= median  (THIS)
    R_cov    crop iff the window really covers      (CEILING -- uses the label, not deployable)
PRE-REGISTERED: R_vrh - bar, CI clear of zero on BOTH models, AND R_vrh >= R_disp. Anything less and
the better verifier does not convert, which is itself the result.
CPU only.
"""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W = 0.25
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
OUTC = {"qwen3": f"{D}/phase78_w_sweep.jsonl", "qwen2": f"{D}/phase97m_merged_qwen2vl.jsonl"}
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
rng = np.random.default_rng(182)


def ci(d, B=8000):
    n = len(d); b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean()*100, float(np.percentile(b, 2.5))*100, float(np.percentile(b, 97.5))*100


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, H, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    out = {r["question_id_full"]: r for r in (json.loads(l) for l in open(OUTC[which]))}
    idx = [e for e in idx if e["question_id_full"] in props and e["question_id_full"] in out]
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)

    inwin, inref, cov, cat, bar, dpr = [], [], [], [], [], []
    for e in idx:
        q = e["question_id_full"]; p = props[q]; r = out[q]
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        cx, cy = p["head"]
        x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
        inwin.append((fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W))
        g = p["gt_box_frac"]
        inref.append((fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3]))
        cov.append(p["head_cov"] >= 0.5); cat.append(r["category"])
        lab = r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
        bar.append(int(np.argmax(r["probs"]["uniform@600"]) == lab))
        dpr.append(int(np.argmax(r["probs"]["head@0.25"]) == lab))
    cov = np.array(cov); cat = np.array(cat)
    bar = np.array(bar, float); dpr = np.array(dpr, float)

    k = max(1, int(round(0.25*H)))
    ref = np.zeros((N, NL, H), np.float32)
    for i, e in enumerate(idx):
        A = imap(e); ref[i] = A[:, :, inref[i]].sum(-1) if inref[i].any() else 0.0
    V = np.zeros(N); Dsp = np.zeros(N)
    G = np.arange(N)
    folds = list(GroupKFold(5).split(G.reshape(-1, 1), G, G))
    for tr, te in folds:
        sc = ref[tr].mean(0); m = np.zeros((NL, H), bool)
        for l in range(NL): m[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            A = imap(idx[i])
            sel = (A * m[:, :, None]).sum(1) / np.maximum(m.sum(1)[:, None], 1)
            g_ = sel[GATE[which]].mean(0); V[i] = g_[inwin[i]].sum() / max(g_.sum(), 1e-12)
            a_ = A.mean(1)[GATE[which]].max(0); Dsp[i] = a_.max() / max(a_.sum(), 1e-12)

    def gate_oof(score):
        r = np.zeros(N, bool)
        for tr, te in folds: r[te] = score[te] >= np.median(score[tr])
        return r

    routers = {"R0 always DPR": np.ones(N, bool),
               "R_disp (§18E incumbent)": gate_oof(Dsp),
               "R_vrh (verifier)": gate_oof(V),
               "R_cov (CEILING, uses label)": cov}
    print(f"\n{'='*94}\n{which}  n={N}   threshold = training-fold median, out-of-fold, no free parameter")
    print(f"  {'router':>28} {'crops':>6} {'pooled':>7} {'bar':>6}   {'pooled - bar':>22}  "
          f"{'single-obj':>18} {'relational':>18}")
    for nm, r in routers.items():
        acc = np.where(r, dpr, bar)
        m, lo, hi = ci(acc-bar)
        s = cat == "direct_attributes"; rel = cat == "relative_position"
        ms = ci((acc-bar)[s])[0]; mr = ci((acc-bar)[rel])[0]
        print(f"  {nm:>28} {100*r.mean():5.1f}% {100*acc.mean():6.1f}% {100*bar.mean():5.1f}%   "
              f"{m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{' CLEARS' if lo > 0 else '       '}  "
              f"{ms:+17.1f} {mr:+17.1f}")
    a_v = np.where(routers["R_vrh (verifier)"], dpr, bar)
    a_d = np.where(routers["R_disp (§18E incumbent)"], dpr, bar)
    m, lo, hi = ci(a_v - a_d)
    print(f"  R_vrh - R_disp  {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]"
          f"{'  better verifier converts' if lo > 0 else '  not distinguishable'}")
