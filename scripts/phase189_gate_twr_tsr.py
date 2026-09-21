"""
Phase 189: THE VRH GATED VERIFIER ON THE TWO FINALISED POLICIES.

§20I showed a VRH verifier gate clears pooled on both models when the localiser is the GBT head
(+8.4 / +6.3 over uniform@600). The paper's two finalised policies are TWR (Transport-Weighted
Read-out, "ridge" in code; user's name: DWA / depth-weighted attention) and TSR (Transport-Scheduled
Resolution; user's name: AVR / adaptive spatial resolution). This tests the gate on both.

⚠ NAME MAPPING ASSUMED: DWA -> TWR/ridge, AVR -> TSR. Neither DWA nor AVR appears in the repo.

LEG A -- TWR/DWA.  TWR proposes a cell, so the gate transfers directly: verify TWR's W=0.25 window,
crop if verified, else spend uniform@600. Ridge placements are recomputed here with phase 182's exact
spec (log-attention + within-layer ranks + 7 geometry terms, standardised, ridge alpha=1 with an
unpenalised intercept, OOF GroupKFold(5) x 3 seeds); phase 182 computes them inline and never saves
them. Outcomes come from phase182_ridge_*.jsonl (`ridge` arm vs `uniform@600`).

LEG B -- TSR/AVR.  TSR proposes NO region: it prunes 90% at the transport boundary and re-encodes at
900. So a region verifier has nothing of its own to verify, and the gate becomes a POLICY SELECTOR,
not a verifier. The signal used is the verifier score on the localiser's window -- i.e. "is there a
concentrated target here?" -- deciding tsr900 vs uniform@600. This is a weaker, differently-shaped
claim and is labelled as such. Outcomes from phase185_tsr_*.jsonl.

Threshold in both legs: the training-fold MEDIAN, out-of-fold, no free parameter (§20D).
PRE-REGISTERED: gated − bar, CI clear of zero on BOTH models, per leg. Always-on and the
label-using ceiling are reported alongside. CPU only.
"""
import json, sys
import numpy as np
from sklearn.model_selection import GroupKFold

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
W, EPS = 0.25, 1e-12
GATE = {"qwen3": list(range(17, 21)), "qwen2": list(range(19, 23))}
BLK = {"qwen3": (16, 27), "qwen2": (15, 27)}
rng = np.random.default_rng(189)


def ci(d, B=8000):
    n = len(d); b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean()*100, float(np.percentile(b, 2.5))*100, float(np.percentile(b, 97.5))*100


def ridge_cells(which, rows, NL):
    """phase 182's ridge placement, verbatim spec."""
    import phase70_rerank_head as P70
    P70.W = W
    Xr, Yr, Gr = [], [], []
    for gi, q in enumerate(rows):
        gh, gw = q["grid"]; n = q["n_img_tokens"]
        A = np.stack([np.asarray(q["attn"][f"L{i}"], float) for i in range(NL)])
        A = A/np.maximum(A.sum(1, keepdims=True), 1e-12)
        LA = np.log(A+1e-12).T
        R = (np.argsort(np.argsort(-A, axis=1), axis=1)/max(n-1, 1)).T
        dep = A[BLK[which][0]:BLK[which][1]].mean(0).reshape(gh, gw)
        pad = np.pad(dep, 1, mode="edge")
        nb = (sum(pad[i:i+gh, j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
        geo = np.c_[np.log(nb+1e-12), fx, fy, np.sqrt((fx-.5)**2+(fy-.5)**2),
                    np.minimum(np.minimum(fx, 1-fx), np.minimum(fy, 1-fy)),
                    (xx.ravel() == gw-1).astype(float), (yy.ravel() == gh-1).astype(float)]
        Xr.append(np.c_[LA, R, geo])
        Yr.append(np.array([P70.coverage(float(fx[i]), float(fy[i]), q["gt_box_frac"])
                            for i in range(n)]))
        Gr.append(np.full(n, gi))
    Xr = np.vstack(Xr); Yr = np.concatenate(Yr); Gr = np.concatenate(Gr)
    Pr = np.zeros(len(Yr)); gs = np.unique(Gr)
    for s in range(3):
        r_ = np.random.default_rng(700+s)
        perm = {g: i for i, g in enumerate(r_.permutation(gs))}
        Gp = np.vectorize(perm.get)(Gr)
        for tr, te in GroupKFold(5).split(Xr, Yr, Gp):
            mu, sd = Xr[tr].mean(0), Xr[tr].std(0)+1e-9
            Xt = np.c_[(Xr[tr]-mu)/sd, np.ones(len(tr))]
            A_ = Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
            w = np.linalg.solve(A_, Xt.T@Yr[tr])
            Pr[te] += np.c_[(Xr[te]-mu)/sd, np.ones(len(te))]@w
    Pr /= 3
    out = {}
    for gi, q in enumerate(rows):
        gh, gw = q["grid"]; m = Gr == gi
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        jr = int(np.argmax(np.where(rm.ravel(), Pr[m], -1e9)))
        out[q["question_id_full"]] = [float((jr % gw+.5)/gw), float((jr//gw+.5)/gh)]
    return out


sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
for which in ["qwen3", "qwen2"]:
    import phase70_rerank_head as P70
    src = {"qwen3": f"{D}/phase30c_attn_maps_all.jsonl",
           "qwen2": f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl"}[which]
    rows = [json.loads(l) for l in open(src)]
    NL = 28
    RC = ridge_cells(which, rows, NL)

    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NH, idx = meta["n_heads"], meta["items"]
    props = json.load(open({"qwen3": f"{D}/phase71a_head_proposals.json",
                            "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}[which]))
    rg = {r["question_id_full"]: r for r in
          (json.loads(l) for l in open(f"{D}/phase182_ridge_{which}.jsonl"))}
    ts = {r["question_id_full"]: r for r in
          (json.loads(l) for l in open(f"{D}/phase185_tsr_{which}.jsonl"))}
    ids = [e for e in idx if e["question_id_full"] in RC and e["question_id_full"] in props]
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, NH, -1)

    N = len(ids)
    winR, winH, REF = [], [], []
    for e in ids:
        q = e["question_id_full"]; gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        def box(cx, cy):
            x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
            return (fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W)
        winR.append(box(*RC[q])); winH.append(box(*props[q]["head"]))
        g = props[q]["gt_box_frac"]
        REF.append((fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3]))

    ref_mass = np.zeros((N, NL, NH), np.float32)
    for i, e in enumerate(ids):
        A = imap(e); ref_mass[i] = A[:, :, REF[i]].sum(-1) if REF[i].any() else 0.0
    k = max(1, int(round(0.25*NH)))
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))
    Vr = np.zeros(N); Vh = np.zeros(N)
    for tr, te in folds:
        sc = ref_mass[tr].mean(0)
        msk = np.zeros((NL, NH), bool)
        for l in range(NL): msk[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            A = imap(ids[i])
            sel = (A*msk[:, :, None]).sum(1)/np.maximum(msk.sum(1)[:, None], 1)
            gm = sel[GATE[which]].mean(0); gm = gm/max(gm.sum(), EPS)
            Vr[i] = gm[winR[i]].sum(); Vh[i] = gm[winH[i]].sum()

    def gate(score):
        r = np.zeros(N, bool)
        for tr, te in folds: r[te] = score[te] >= np.median(score[tr])
        return r

    print(f"\n{'='*90}\n{which}  n={N}")
    for leg, store, arm, sig, nm in [
            ("LEG A  TWR/DWA gate", rg, "ridge", Vr, "verifier on TWR's window"),
            ("LEG B  TSR/AVR selector", ts, "tsr900", Vh, "verifier on the localiser window (proxy)")]:
        keep = [i for i, e in enumerate(ids) if e["question_id_full"] in store]
        if not keep: print(f"  {leg}: no overlapping outcomes"); continue
        qs = [ids[i]["question_id_full"] for i in keep]
        lab = lambda r: r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
        bar = np.array([int(np.argmax(store[q]["probs"]["uniform@600"]) == lab(store[q])) for q in qs], float)
        pol = np.array([int(np.argmax(store[q]["probs"][arm]) == lab(store[q])) for q in qs], float)
        g = gate(sig)[keep]
        acc = np.where(g, pol, bar)
        m1, l1, h1 = ci(pol-bar); m2, l2, h2 = ci(acc-bar)
        print(f"  {leg}   n={len(qs)}   signal: {nm}")
        print(f"     always-on {arm:8} {100*pol.mean():5.1f}%  bar {100*bar.mean():5.1f}%   "
              f"{m1:+5.1f} [{l1:+5.1f},{h1:+5.1f}]{' CLEARS' if l1 > 0 else ''}")
        print(f"     GATED     ({100*g.mean():4.1f}% on) {100*acc.mean():5.1f}%              "
              f"{m2:+5.1f} [{l2:+5.1f},{h2:+5.1f}]{' CLEARS' if l2 > 0 else ''}")
