"""
Phase 174 -- WINDOW CHOSEN BY EXPECTED ACCURACY, not coverage (repairs the SS18G metric artefact).
SS18G: a (cell,W) re-ranker trained on coverage picks W=0.5 everywhere because coverage never charges for lost
magnification. The fix is the objective: choose W to maximise EXPECTED ACCURACY, where accuracy is a single
global curve f(coverage, W) fit from the end-task W sweeps (phase 78 Qwen3, phase 97m Qwen2; arms head@W,
oracle@W, uniform@300 as W=1). The curve has 4 parameters and no question information. Per item:
    W* = argmax_W  f( predicted coverage@W at the head's cell , W )
predicted coverage@W comes from the SS18G joint (cell,W) regressor, OOF; the curve is fit on the training fold.
Evaluation is an EXACT LOOKUP of the end-task result of arm head@W* (or uniform@300 for W*=1) -- no GPU.
PRE-REGISTERED
    P1     utility-W  -  always-0.25 (head@0.25), pooled, CI clear of zero on BOTH models
    GUARD  single-object not significantly negative
    report chosen-W histogram (a two-value collapse = a gate in disguise), relational, and the same rule fed
    ACTUAL coverage (upper bound for the mechanism given perfect coverage prediction).
"""
import json, sys, numpy as np
sys.path.insert(0,"/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
WS=[0.15,0.25,0.35,0.5,0.7]; WA=WS+[1.0]
CFG={"Qwen3-VL":(P70.build,"data/phase71a_head_proposals.json","data/phase78_w_sweep.jsonl"),
     "Qwen2-VL":(P80.build,"data/phase80a_qwen2vl_proposals.json","data/phase97m_merged_qwen2vl.jsonl")}
def covW(r,cell,W):
    if W>=1.0: return 1.0
    gh,gw=r["grid"]; P70.W=W; return P70.coverage((cell%gw+.5)/gw,(cell//gw+.5)/gh,r["gt_box_frac"])
def oracle_cov(r,W):
    if W>=1.0: return 1.0
    x0,y0,x1,y1=r["gt_box_frac"]; P70.W=W; return P70.coverage((x0+x1)/2,(y0+y1)/2,r["gt_box_frac"])
def feats(c,W): c=np.asarray(c,float); l=np.log(np.asarray(W,float)); return np.c_[c,l,c*l]
for name,(build,propf,sweepf) in CFG.items():
    P70.W=0.25; X,_,G,_,rows=build()[:5]; props=json.load(open(propf)); sweep={json.loads(l)["question_id_full"]:json.loads(l) for l in open(sweepf)}
    gs=np.unique(G); N=len(gs); qid=[r["question_id_full"] for r in rows]; cat=np.array([r["category"] for r in rows])
    cell=np.array([int(props[q]["head"]) if not isinstance(props[q]["head"],(list,tuple)) else int(props[q]["head"][0]*rows[i]["grid"][1]+props[q]["head"][1]) for i,q in enumerate(qid)])
    acc=lambda q,arm: float(int(np.argmax(sweep[q]["probs"][arm]))==sweep[q]["label"])
    ACC={W:np.array([acc(q,f"head@{W}" if W<1 else "uniform@300") for q in qid]) for W in WA}
    OACC={W:np.array([acc(q,f"oracle@{W}") for q in qid]) for W in WS}
    bar=np.array([acc(q,"uniform@600") for q in qid])
    CV={W:np.array([covW(rows[i],cell[i],W) for i in range(N)]) for W in WA}; OCV={W:np.array([oracle_cov(rows[i],W) for i in range(N)]) for W in WS}
    YW={}
    for W in WS:
        P70.W=W; YW[W]=np.concatenate([[P70.coverage(((j%r["grid"][1])+.5)/r["grid"][1],((j//r["grid"][1])+.5)/r["grid"][0],r["gt_box_frac"]) for j in range(r["n_img_tokens"])] for r in rows])
    Xs=np.vstack([np.c_[X,np.full(len(X),W)] for W in WS]); Ys=np.concatenate([YW[W] for W in WS])
    offs={gi:int(np.where(G==gi)[0][0]) for gi in gs}
    pick=np.zeros((3,N)); pick_act=np.zeros((3,N)); chosen=np.zeros((3,N))
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,YW[0.25],Gp):
            tri=np.unique(G[tr]); tei=np.unique(G[te])
            trs=np.concatenate([tr+k*len(X) for k in range(len(WS))]); ytr=Ys[trs]; pos=trs[ytr>0]; negpool=trs[ytr<=0]
            neg=rng.choice(negpool,size=min(len(negpool),30*len(WS)*len(tri)),replace=False); sub=np.concatenate([pos,neg])
            mj=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(Xs[sub],Ys[sub])
            # accuracy curve on training-fold items: head@W, oracle@W, uniform@300 observations
            fc,fw,fy=[],[],[]
            for W in WA: fc+=list(CV[W][tri]); fw+=[W]*len(tri); fy+=list(ACC[W][tri])
            for W in WS: fc+=list(OCV[W][tri]); fw+=[W]*len(tri); fy+=list(OACC[W][tri])
            lr=LogisticRegression(C=10.0,max_iter=2000).fit(feats(fc,fw),np.array(fy))
            for gi in tei:
                i=offs[gi]+cell[gi]; pc=[float(np.clip(mj.predict(np.r_[X[i],W][None])[0],0,1)) if W<1 else 1.0 for W in WA]
                u=lr.predict_proba(feats(pc,WA))[:,1]; k=int(np.argmax(u)); pick[s,gi]=ACC[WA[k]][gi]; chosen[s,gi]=WA[k]
                ua=lr.predict_proba(feats([CV[W][gi] for W in WA],WA))[:,1]; pick_act[s,gi]=ACC[WA[int(np.argmax(ua))]][gi]
    util=pick.mean(0); util_act=pick_act.mean(0); base=ACC[0.25]; single=cat=="direct_attributes"
    rng=np.random.default_rng(174)
    def ci(d):
        n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
    print(f"\n{name}  n={N}   arms: "+"  ".join(f"head@{W}:{ACC[W].mean()*100:.1f}" for W in WS)+f"  uniform@300:{ACC[1.0].mean()*100:.1f}  bar:{bar.mean()*100:.1f}")
    print(f"  {'':>11} {'always.25':>9} {'utility-W':>9} {'bar':>6}   utility − always.25      utility − bar          | utility(ACTUAL cov) − always.25")
    for tag,mk in [("single",single),("relational",~single),("ALL",np.ones(N,bool))]:
        a,lo,hi=ci((util-base)[mk]); b_,blo,bhi=ci((util-bar)[mk]); c,clo,chi=ci((util_act-base)[mk])
        print(f"  {tag:>11} {base[mk].mean()*100:8.1f}% {util[mk].mean()*100:8.1f}% {bar[mk].mean()*100:5.1f}%   {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else (' ✗' if hi<0 else '  ')}   {b_:+5.1f} [{blo:+5.1f},{bhi:+5.1f}]{'✔' if blo>0 else '  '}   |  {c:+5.1f} [{clo:+5.1f},{chi:+5.1f}]")
    ch=chosen.ravel(); print("  chosen-W histogram (3 seeds): "+"  ".join(f"W{W}:{np.mean(ch==W)*100:.0f}%" for W in WA)+f"   | single median {np.median(chosen[:,single]):.2f}  relational median {np.median(chosen[:,~single]):.2f}")
