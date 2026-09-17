"""
Phase 166: DEPTH RE-RANKING OVER (cell, window) PAIRS -- adaptation inside the re-ranker.

The head regresses coverage@0.25 per cell from the depth profile and takes the argmax cell with a fixed
window. Coverage labels exist for every window size, so train the SAME head with W as an input feature
to predict coverage@W, and take the argmax over (cell, W). Nothing else changes: same 65 features,
same GBT, same OOF folds, no gate, no map heuristic. W in {0.15, 0.25, 0.35, 0.5}.

Distinct from the dead sizer (SS14U): that regressed W from six summary features; this is the per-cell
re-ranker itself, ranking proposals that differ in window as well as position.

PRE-REGISTERED (coverage, OOF, both models)
    metric  coverage of the CHOSEN (cell, W) window >= 0.5           (same COV_HIT)
    P1      joint >= fixed head@0.25 on RELATIONAL items, CI clear on both models  (the failing cell)
    GUARD   joint - fixed on SINGLE-object not significantly negative
    context fixed head@W for each W, and the oracle over W per item
End-task follows only if P1 and the guard both hold.
"""
import json, sys, numpy as np
sys.path.insert(0,"/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
WS=[0.15,0.25,0.35,0.5]
def cov_all(rows,W):
    P70.W=W; Y=[]
    for r in rows:
        gh,gw=r["grid"]; n=r["n_img_tokens"]; yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=(yy+.5)/gh,(xx+.5)/gw
        Y.append(np.array([P70.coverage(float(fx.flat[i]),float(fy.flat[i]),r["gt_box_frac"]) for i in range(n)]))
    return np.concatenate(Y)
for name,build in [("Qwen3-VL",P70.build),("Qwen2-VL",P80.build)]:
    P70.W=0.25; r=build(); X,_,G,DEP,rows=r[0],r[1],r[2],r[3],r[4]
    YW={W:cov_all(rows,W) for W in WS}; gs=np.unique(G); N=len(gs); cat=np.array([q["category"] for q in rows])
    ring={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; m=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: m[1:-1,1:-1]=True
        else: m[:]=True
        ring[gi]=m.ravel()
    # stacked training set: rows x windows, W appended as a feature
    Xs=np.vstack([np.c_[X,np.full(len(X),W)] for W in WS]); Ys=np.concatenate([YW[W] for W in WS]); Gs=np.tile(G,len(WS))
    Pj=np.zeros(len(Ys)); Pf=np.zeros(len(X))
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,YW[0.25],Gp):
            # joint model
            trs=np.concatenate([tr+k*len(X) for k in range(len(WS))]); tes=np.concatenate([te+k*len(X) for k in range(len(WS))])
            ytr=Ys[trs]; pos=trs[ytr>0]; negpool=trs[ytr<=0]; neg=rng.choice(negpool,size=min(len(negpool),30*len(WS)*len(np.unique(G[tr]))),replace=False)
            sub=np.concatenate([pos,neg]); mj=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(Xs[sub],Ys[sub]); Pj[tes]+=mj.predict(Xs[tes])
            # fixed-W model (incumbent), same folds
            y25=YW[0.25][tr]; pos=tr[y25>0]; negpool=tr[y25<=0]; neg=rng.choice(negpool,size=min(len(negpool),30*len(np.unique(G[tr]))),replace=False)
            sub=np.concatenate([pos,neg]); mf=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(X[sub],YW[0.25][sub]); Pf[te]+=mf.predict(X[te])
    Pj/=3; Pf/=3
    fixed=[]; joint=[]; chosenW=[]; orc=[]
    for gi in gs:
        m=np.where(G==gi)[0]; rm=ring[gi]
        j=int(np.argmax(np.where(rm,Pf[m],-1e9))); fixed.append(float(YW[0.25][m][j]>=P70.COV_HIT))
        best=(-1e9,None,None)
        for k,W in enumerate(WS):
            sc=np.where(rm,Pj[m+k*len(X)],-1e9); jj=int(np.argmax(sc))
            if sc[jj]>best[0]: best=(sc[jj],jj,W)
        joint.append(float(YW[best[2]][m][best[1]]>=P70.COV_HIT)); chosenW.append(best[2])
        orc.append(float(max(YW[W][m][int(np.argmax(np.where(rm,Pf[m],-1e9)))] for W in WS)>=P70.COV_HIT))
    fixed=np.array(fixed); joint=np.array(joint); chosenW=np.array(chosenW); orc=np.array(orc); single=cat=="direct_attributes"
    rng=np.random.default_rng(166)
    def ci(d):
        n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
    print(f"\n{name}  n={N}  OOF top-1 coverage (chosen window)")
    print(f"  {'':>12} {'fixed@0.25':>11} {'joint(cell,W)':>14}   joint − fixed          W chosen (median)")
    for tag,mk in [("single",single),("relational",~single),("ALL",np.ones(N,bool))]:
        a,lo,hi=ci((joint-fixed)[mk]); print(f"  {tag:>12} {fixed[mk].mean()*100:10.1f}% {joint[mk].mean()*100:13.1f}%   {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else (' ✗' if hi<0 else '  ')}   {np.median(chosenW[mk]):.2f}  (frac W>0.25: {np.mean(chosenW[mk]>0.25)*100:.0f}%)")
    print(f"  oracle-over-W at the fixed head's cell: {orc.mean()*100:.1f}%   |  fixed head@W for reference: " + "  ".join(f"W{W}:{np.mean([YW[W][G==gi][int(np.argmax(np.where(ring[gi],Pf[G==gi],-1e9)))]>=P70.COV_HIT for gi in gs])*100:.1f}" for W in WS))
