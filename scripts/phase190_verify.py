"""
Phase 190: VERIFY THE VRH IMPLEMENTATION. Are the §20 negatives real, or artefacts?

CHECKS
 C1 POSITIVE CONTROL  the verifier scored on the GT-centred window must be much higher than on the
                      DPR window, which must in turn beat a random window. If not, the mass-in-region
                      computation is wrong.
 C2 NEGATIVE CONTROL  verifier score on a RANDOM window must predict coverage at ~0.500 AUROC.
 C3 LABEL SHUFFLE     selecting heads on SHUFFLED referent masks must collapse the verifier to chance.
 C4 STRICT VRH        the paper's actual prescription: keys = cells overlapping the GT BOX (not my
                      fatter ">=50% window coverage" mask), selection = GLOBAL top-20 (l,h) pairs
                      (~2.6% of heads), not per-layer top-25%. Does the §20H result survive?
 C5 SPARSITY PROFILE  where do the selected heads sit? VRH reports a sparse, concentrated set.
 C6 REPRODUCE §20H    VRH vs naive all-head pooling, the +0.045/+0.038 paired result.
"""
import json
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, EPS = 0.25, 1e-12
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
PROP = {"qwen3": f"{D}/phase71a_head_proposals.json", "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}
rng = np.random.default_rng(190)


def auroc(s, p):
    p = np.asarray(p, bool)
    if p.all() or not p.any(): return float("nan")
    r = np.argsort(np.argsort(s))+1.0
    n1, n0 = p.sum(), (~p).sum()
    return (r[p].sum()-n1*(n1+1)/2)/(n1*n0)


for which in ["qwen3", "qwen2"]:
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NL, NH, idx = meta["n_layers"], meta["n_heads"], meta["items"]
    props = json.load(open(PROP[which]))
    idx = [e for e in idx if e["question_id_full"] in props]
    N = len(idx)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, NH, -1)

    winD, winO, winR, REF, cov = [], [], [], [], np.zeros(N, bool)
    for i, e in enumerate(idx):
        q = e["question_id_full"]; p = props[q]
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        def box(cx, cy):
            x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
            return (fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W)
        g = p["gt_box_frac"]
        winD.append(box(*p["head"])); winO.append(box((g[0]+g[2])/2, (g[1]+g[3])/2))
        winR.append(box(float(rng.uniform(W/2, 1-W/2)), float(rng.uniform(W/2, 1-W/2))))
        REF.append((fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3]))
        cov[i] = p["head_cov"] >= 0.5

    ref_strict = np.zeros((N, NL, NH), np.float32)   # C4: keys = GT-box cells (the paper's K_s)
    for i, e in enumerate(idx):
        A = imap(e)
        ref_strict[i] = A[:, :, REF[i]].sum(-1) if REF[i].any() else 0.0
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))

    def verifier(mask_fn, window, shuffle=False):
        V = np.zeros(N)
        for tr, te in folds:
            src = ref_strict[tr]
            if shuffle:
                src = ref_strict[rng.permutation(N)[:len(tr)]]
            m = mask_fn(src.mean(0))
            for i in te:
                A = imap(idx[i])
                sel = (A*m[:, :, None]).sum(1)/np.maximum(m.sum(1)[:, None], 1)
                gm = sel[GATE[which]].mean(0); gm = gm/max(gm.sum(), EPS)
                V[i] = gm[window[i]].sum()
        return V

    k = max(1, int(round(0.25*NH)))
    def perlayer(sc):
        m = np.zeros((NL, NH), bool)
        for l in range(NL): m[l, np.argsort(-sc[l])[:k]] = True
        return m
    def top20(sc):
        m = np.zeros((NL, NH), bool); m.ravel()[np.argsort(-sc.ravel())[:20]] = True
        return m
    def allheads(sc):
        return np.ones((NL, NH), bool)

    Vd = verifier(perlayer, winD); Vo = verifier(perlayer, winO); Vr = verifier(perlayer, winR)
    Va = verifier(allheads, winD); V20 = verifier(top20, winD)
    Vsh = verifier(perlayer, winD, shuffle=True)

    print(f"\n{'='*88}\n{which}  n={N}  ({NL} layers x {NH} heads; top-{k}/layer = "
          f"{100*k*NL/(NL*NH):.0f}% of heads, global top-20 = {100*20/(NL*NH):.1f}%)")
    print(f"  C1 POSITIVE mean verifier mass:  GT window {Vo.mean():.3f} > DPR window {Vd.mean():.3f} "
          f"> random window {Vr.mean():.3f}   -> {'PASS' if Vo.mean() > Vd.mean() > Vr.mean() else 'FAIL'}")
    a_r = auroc(Vr, cov)
    print(f"  C2 NEGATIVE random-window AUROC vs coverage {a_r:.3f}  "
          f"-> {'PASS (chance)' if abs(a_r-0.5) < 0.08 else 'FAIL'}")
    a_sh = auroc(Vsh, cov)
    print(f"  C3 SHUFFLE  heads picked on shuffled referents: AUROC {a_sh:.3f} vs real {auroc(Vd, cov):.3f}")
    print(f"  C4 STRICT VRH (global top-20, GT-box keys) AUROC {auroc(V20, cov):.3f} | "
          f"per-layer 25% {auroc(Vd, cov):.3f} | all-heads {auroc(Va, cov):.3f}")
    sc = ref_strict.mean(0); m20 = top20(sc)
    ls = np.where(m20.any(1))[0]
    print(f"  C5 SPARSITY top-20 heads sit in layers {list(ls)}")
    d = auroc(Vd, cov)-auroc(Va, cov)
    b = [auroc(Vd[i], cov[i])-auroc(Va[i], cov[i]) for i in
         (rng.integers(0, N, N) for _ in range(3000))]
    b = [x for x in b if not np.isnan(x)]
    print(f"  C6 REPRODUCE §20H  VRH - all-heads {d:+.3f} "
          f"[{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]")
