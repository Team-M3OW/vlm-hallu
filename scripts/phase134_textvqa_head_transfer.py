"""
Phase 134: does a head trained on TextVQA boxes (zero-shot) beat the 191-item V*Bench OOF head?
See phase 133 docstring for the pre-registration (P1 GBT, P2 readable head, P3 combined).
Metric: V*Bench top-1 coverage, ring-masked, W in {0.15, 0.25}. Also reports the deployed argmax
and the TextVQA head's OOF coverage on TextVQA itself (sanity: it must beat the argmax there).
"""
import json, sys, numpy as np, warnings
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
src = open("/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts/phase132_loglin_plus.py").read()
exec(src[:src.rindex("for W in [0.15, 0.25]:")])   # build, build_nb, oof, top1, ci, LogLinear, LogLinNB, GBT import
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
VS, TX = f"{D}/phase30c_attn_maps_all.jsonl", f"{D}/phase133_textvqa_attn_qwen3.jsonl"

def fit_all(X, Y, G, make, seed=0, neg_per_item=30):
    rng = np.random.default_rng(700+seed); pos = np.where(Y > 0)[0]; negpool = np.where(Y <= 0)[0]
    neg = rng.choice(negpool, size=min(len(negpool), neg_per_item*len(np.unique(G))), replace=False)
    sub = np.concatenate([pos, neg]); return make(seed).fit(X[sub], Y[sub])

GBT = lambda s: HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10, random_state=s)
for W in [0.15, 0.25]:
    Xv, Yv, Gv, Dv, Rv, NL = build(VS, 28, W);  Xvn, _, _, _, _, _ = build_nb(VS, 28, W)
    Xt, Yt, Gt, Dt, Rt, _  = build(TX, 28, W);  Xtn, _, _, _, _, _ = build_nb(TX, 28, W)
    nT = len(np.unique(Gt))
    dep = np.array([float(Yv[Gv==gi][int(np.argmax(np.where(Rv[gi], Dv[gi], -1e9)))] >= P70.COV_HIT) for gi in np.unique(Gv)])
    depT = np.array([float(Yt[Gt==gi][int(np.argmax(np.where(Rt[gi], Dt[gi], -1e9)))] >= P70.COV_HIT) for gi in np.unique(Gt)])
    print(f"\n================ W = {W}   (TextVQA train items: {nT}) ================", flush=True)
    print(f"  sanity on TextVQA itself (OOF): deployed argmax {depT.mean()*100:.1f}%  GBT {top1(oof(Xt, Yt, Gt, GBT), Yt, Gt, Rt).mean()*100:.1f}%", flush=True)
    res = {}
    res["V*Bench OOF GBT (incumbent)"] = top1(oof(Xv, Yv, Gv, GBT), Yv, Gv, Rv)
    res["V*Bench OOF readable"]        = top1(oof(Xvn, Yv, Gv, lambda s: LogLinNB(NL)), Yv, Gv, Rv)
    m = [fit_all(Xt, Yt, Gt, GBT, s) for s in range(3)];            res["TextVQA-only GBT -> V*Bench (zero-shot)"] = top1(np.mean([mm.predict(Xv) for mm in m], 0), Yv, Gv, Rv)
    m = [fit_all(Xtn, Yt, Gt, lambda s: LogLinNB(NL), s) for s in range(3)]; res["TextVQA-only readable -> V*Bench (zero-shot)"] = top1(np.mean([mm.predict(Xvn) for mm in m], 0), Yv, Gv, Rv)
    # P3 combined: TextVQA items added to every training fold of the V*Bench OOF
    P = np.zeros(len(Yv)); gs = np.unique(Gv)
    from sklearn.model_selection import GroupKFold
    for s in range(3):
        rng = np.random.default_rng(700+s); perm = {g: i for i, g in enumerate(rng.permutation(gs))}; Gp = np.vectorize(perm.get)(Gv)
        for tr, te in GroupKFold(5).split(Xv, Yv, Gp):
            ytr = Yv[tr]; pos = tr[ytr > 0]; negpool = tr[ytr <= 0]; neg = rng.choice(negpool, size=min(len(negpool), 30*len(np.unique(Gv[tr]))), replace=False)
            sub = np.concatenate([pos, neg])
            posT = np.where(Yt > 0)[0]; negT = rng.choice(np.where(Yt <= 0)[0], size=min(int((Yt <= 0).sum()), 30*nT), replace=False)
            Xc = np.vstack([Xv[sub], Xt[np.concatenate([posT, negT])]]); Yc = np.concatenate([Yv[sub], Yt[np.concatenate([posT, negT])]])
            P[te] += GBT(s).fit(Xc, Yc).predict(Xv[te])
    res["TextVQA + V*Bench(OOF) GBT"] = top1(P/3, Yv, Gv, Rv)
    inc = res["V*Bench OOF GBT (incumbent)"]
    print(f"  {'arm':>46} {'V*Bench cov':>12}   vs incumbent")
    print(f"  {'deployed argmax':>46} {dep.mean()*100:11.1f}%")
    for k, v in res.items():
        mm, lo, hi = ci(v-inc); print(f"  {k:>46} {v.mean()*100:11.1f}%   {mm:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ('WORSE' if hi<0 else '')}", flush=True)
