"""Phase 189b: WHY does the gate help the GBT head but hurt TWR? Decompose the gate's arithmetic.
gain(gate) = (losses avoided on skipped items) - (wins forfeited on skipped items). Measures, per
localiser: coverage rate, always-on delta, and the delta split by gate-ON vs gate-OFF items."""
import json, sys, re
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70

src = open("/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts/phase189_gate_twr_tsr.py").read()
body = src[src.index("D = "):src.index("sys.path.insert")]
exec(body)                                   # brings in ridge_cells, GATE, BLK, W, EPS, ci

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
for which in ["qwen3", "qwen2"]:
    rows = [json.loads(l) for l in open({"qwen3": f"{D}/phase30c_attn_maps_all.jsonl",
                                         "qwen2": f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl"}[which])]
    NL = 28
    RC = ridge_cells(which, rows, NL)
    buf = np.load(f"{D}/phase170_perhead_{which}.npy")
    meta = json.load(open(f"{D}/phase170_perhead_{which}_index.json"))
    NH, idx = meta["n_heads"], meta["items"]
    props = json.load(open({"qwen3": f"{D}/phase71a_head_proposals.json",
                            "qwen2": f"{D}/phase80a_qwen2vl_proposals.json"}[which]))
    rg = {r["question_id_full"]: r for r in
          (json.loads(l) for l in open(f"{D}/phase182_ridge_{which}.jsonl"))}
    ids = [e for e in idx if e["question_id_full"] in RC and e["question_id_full"] in props
           and e["question_id_full"] in rg]
    N = len(ids)
    imap = lambda e: buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, NH, -1)
    winR, winH, REF, covR, covH = [], [], [], [], []
    P70.W = W
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
        covR.append(P70.coverage(RC[q][0], RC[q][1], g) >= 0.5)
        covH.append(props[q]["head_cov"] >= 0.5)
    covR, covH = np.array(covR), np.array(covH)
    ref = np.zeros((N, NL, NH), np.float32)
    for i, e in enumerate(ids):
        A = imap(e); ref[i] = A[:, :, REF[i]].sum(-1) if REF[i].any() else 0.0
    k = max(1, int(round(0.25*NH)))
    folds = list(GroupKFold(5).split(np.arange(N).reshape(-1, 1), np.arange(N), np.arange(N)))
    Vr, Vh = np.zeros(N), np.zeros(N)
    for tr, te in folds:
        sc = ref[tr].mean(0); msk = np.zeros((NL, NH), bool)
        for l in range(NL): msk[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            A = imap(ids[i]); sel = (A*msk[:, :, None]).sum(1)/np.maximum(msk.sum(1)[:, None], 1)
            gm = sel[GATE[which]].mean(0); gm = gm/max(gm.sum(), EPS)
            Vr[i] = gm[winR[i]].sum(); Vh[i] = gm[winH[i]].sum()
    qs = [e["question_id_full"] for e in ids]
    lab = lambda r: r["label"] if isinstance(r["label"], int) else "ABCD".index(r["label"])
    bar = np.array([int(np.argmax(rg[q]["probs"]["uniform@600"]) == lab(rg[q])) for q in qs], float)

    def auroc(s, p):
        p = np.asarray(p, bool)
        if p.all() or not p.any(): return float("nan")
        r = np.argsort(np.argsort(s))+1.0
        n1, n0 = p.sum(), (~p).sum()
        return (r[p].sum()-n1*(n1+1)/2)/(n1*n0)

    print(f"\n{'='*94}\n{which}  n={N}   window covers the evidence: GBT head {100*covH.mean():.1f}%  "
          f"TWR {100*covR.mean():.1f}%")
    for nm, arm, sig, cov in [("GBT head", "head", Vh, covH), ("TWR/ridge", "ridge", Vr, covR)]:
        pol = np.array([int(np.argmax(rg[q]["probs"][arm]) == lab(rg[q])) for q in qs], float)
        on = np.zeros(N, bool)
        for tr, te in folds: on[te] = sig[te] >= np.median(sig[tr])
        d = pol-bar
        print(f"  {nm:10} always-on {100*d.mean():+5.1f}pp | ON {100*d[on].mean():+5.1f}pp "
              f"(n={on.sum():3d}) | OFF {100*d[~on].mean():+5.1f}pp (n={(~on).sum():3d}) | "
              f"gate Δ {100*(d[on].mean()*on.mean()) - 100*d.mean():+5.1f}pp | "
              f"verifier AUROC vs own coverage {auroc(sig, cov):.3f}")
        print(f"             on MISSED items (window off-target): {100*d[~cov].mean():+5.1f}pp "
              f"(n={(~cov).sum():3d})   on COVERED items: {100*d[cov].mean():+5.1f}pp")
