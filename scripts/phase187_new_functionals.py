"""
Phase 187: THREE NEW VERIFIER FUNCTIONALS — not reweightings of mass-in-window.

§20L/§20M closed the reweighting families: head selection is rank-invariant, monotone transforms of V
are provably inert under rank-based evaluation, and mass+concentration fusion is bounded by their
0.86-0.89 correlation. A winning score must be a DIFFERENT FUNCTIONAL. Three, all training-free, all
using the same fold-honest original-VRH head set:

  A  SIGNED DEPTH-CONTRAST      V_delta = m_{gate band}(R) - m_{L26-27}(R)
     The paper's own mechanism applied to the verifier. §19 / METHOD.md §2: the re-ranker earns its
     gain by SUBTRACTING layers the block mean adds -- the final layer is anti-correlated with the
     target (gt_pct 0.529 < 0.500 chance). The current verifier pools the gate band with a MEAN,
     exactly the operation shown to destroy that signal. A null here is still informative: it would
     say the subtraction mechanism is specific to cell RANKING and does not transfer to region
     VERIFICATION.

  B  HEAD CONSENSUS             V_vote = fraction of selected heads whose ARGMAX cell lies in R
     Currently heads are pooled and then mass is measured. This measures whether the retrieval heads
     CORROBORATE EACH OTHER -- an agreement statistic, not a mass statistic.

  C  ITEM-LEVEL MARGIN          V_margin = m_(1) - m_(2) over the top-3 NMS regions
     Decisiveness: does the map have one answer or several? Tested in §20L only as a HEAD score.

MULTIPLICITY. Three tests against one gate result. Both 95% and Bonferroni-corrected 98.33% intervals
are reported; promotion requires clearing on BOTH models at the CORRECTED level and exceeding the
~0.02 AUROC noise seen in §20M. The gate is not re-run unless that holds. CPU only.
"""
import json
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, EPS, K_REG, NMS = 0.25, 1e-12, 3, 3
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
NEG = [26, 27]
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
OUTC = {"qwen3": f"{D}/phase78_w_sweep.jsonl", "qwen2": f"{D}/phase97m_merged_qwen2vl.jsonl"}
rng = np.random.default_rng(187)


def auroc(s, p):
    p = np.asarray(p, bool)
    if p.all() or not p.any(): return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = p.sum(), (~p).sum()
    return (r[p].sum() - n1*(n1+1)/2) / (n1*n0)


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, NH, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    out = {r["question_id_full"]: r for r in (json.loads(l) for l in open(OUTC[which]))}
    idx = [e for e in idx if e["question_id_full"] in props and e["question_id_full"] in out]
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, NH, -1)

    WIN, REF, GRID = [], [], []
    y_cov, y_acc = np.zeros(N, bool), np.zeros(N, bool)
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
        GRID.append((gh, gw, n, fx, fy))
        y_cov[i] = p["head_cov"] >= 0.5
        lab = r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
        y_acc[i] = int(np.argmax(r["probs"]["head@0.25"]) == lab)

    ref_mass = np.zeros((N, NL, NH), np.float32)
    for i, e in enumerate(idx):
        A = imap(e); ref_mass[i] = A[:, :, REF[i]].sum(-1) if REF[i].any() else 0.0
    k = max(1, int(round(0.25*NH)))
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))

    V = np.zeros(N); Vd = np.zeros(N); Vv = np.zeros(N); Vm = np.zeros(N)
    for tr, te in folds:
        sc = ref_mass[tr].mean(0)
        msk = np.zeros((NL, NH), bool)
        for l in range(NL): msk[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            A = imap(idx[i]); gh, gw, n, fx, fy = GRID[i]
            sel = (A*msk[:, :, None]).sum(1)/np.maximum(msk.sum(1)[:, None], 1)   # (NL, n)
            sel = sel/np.maximum(sel.sum(1, keepdims=True), EPS)
            gm = sel[GATE[which]].mean(0); gm = gm/max(gm.sum(), EPS)
            V[i] = gm[WIN[i]].sum()
            # A: signed depth contrast
            Vd[i] = sel[GATE[which]].mean(0)[WIN[i]].sum() - sel[NEG].mean(0)[WIN[i]].sum()
            # B: head consensus vote over selected heads in the gate band
            hits = tot = 0
            for l in GATE[which]:
                for h in np.where(msk[l])[0]:
                    tot += 1
                    if WIN[i][int(np.argmax(A[l, h]))]: hits += 1
            Vv[i] = hits/max(tot, 1)
            # C: item-level margin over top-3 NMS regions of the pooled gated map
            order = np.argsort(-gm); picked = []
            for c in order:
                if len(picked) >= K_REG: break
                cy_, cx_ = divmod(int(c), gw)
                if all(abs(cy_-a)+abs(cx_-b) >= NMS for a, b in picked): picked.append((cy_, cx_))
            ms = []
            for (a, b) in picked:
                rx, ry = (b+.5)/gw, (a+.5)/gh
                rx0 = min(max(0, rx-W/2), 1-W); ry0 = min(max(0, ry-W/2), 1-W)
                s_ = (fx >= rx0) & (fx <= rx0+W) & (fy >= ry0) & (fy <= ry0+W)
                ms.append(gm[s_].sum())
            ms = sorted(ms, reverse=True)
            Vm[i] = (ms[0]-ms[1]) if len(ms) >= 2 else ms[0]

    arms = {"V (incumbent)": V, "A signed depth-contrast": Vd,
            "B head consensus vote": Vv, "C item-level margin": Vm}
    print(f"\n{'='*88}\n{which}  n={N}   head set = original VRH, fold-honest; 3 tests -> Bonferroni")
    for nm, s in arms.items():
        print(f"    {nm:26} AUROC y_cov {auroc(s, y_cov):.3f} | y_acc {auroc(s, y_acc):.3f}")
    for nm in list(arms)[1:]:
        d = auroc(arms[nm], y_cov) - auroc(V, y_cov)
        b = []
        for _ in range(6000):
            i = rng.integers(0, N, N)
            x = auroc(arms[nm][i], y_cov[i]) - auroc(V[i], y_cov[i])
            if not np.isnan(x): b.append(x)
        lo95, hi95 = np.percentile(b, [2.5, 97.5])
        loB, hiB = np.percentile(b, [100*0.05/6, 100*(1-0.05/6)])
        print(f"    {nm:26} Δ {d:+.3f}  95% [{lo95:+.3f},{hi95:+.3f}]"
              f"{' ✔' if lo95 > 0 else '  '}  Bonf [{loB:+.3f},{hiB:+.3f}]"
              f"{' PROMOTE' if loB > 0 and d > 0.02 else ''}")
    print(f"    corr with V:  A {np.corrcoef(V,Vd)[0,1]:+.3f}  B {np.corrcoef(V,Vv)[0,1]:+.3f}  "
          f"C {np.corrcoef(V,Vm)[0,1]:+.3f}")
