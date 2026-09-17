"""
Phase 181 -- (a) IS THE METHOD JUST A LINEAR DEPTH FILTER?  (b) IS THE FILTER'S HIGH FREQUENCY REAL?

(a) Closed-form ridge vs the incumbent GBT on IDENTICAL folds, IDENTICAL rows, IDENTICAL features,
    3 seeds x GroupKFold(5) grouped by item. Phase 180 hinted ridge >= tree on all four architectures
    but compared across different fold seeds / negative subsampling, i.e. inside the 1-2.5pp floor.
(b) Phase 180 found the fitted w(l) alternates sign layer-to-layer and a low-order Legendre basis is
    SIGNIFICANTLY worse. Alternating signs on correlated regressors is the multicollinearity signature,
    so the honest test is a ROUGHNESS penalty: minimise ||y-Xw||^2 + lam*||w||^2 + mu*||D2 w||^2 with D2
    the second difference over DEPTH, sweeping mu from 0 (free) to large (smooth). If accuracy falls
    monotonically as mu rises, the high-frequency content is signal; if it is flat, it was collinearity.
Features (same for both models classes): log A_l (NL) | within-layer rank_l (NL) | log 3x3 nb | 6 geometry.
"""
import json, sys, numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70
P70.W=0.25
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
SRC={"Qwen3-VL":("data/phase30c_attn_maps_all.jsonl",28,(16,27)),
     "Qwen2-VL":("data/phase74_Qwen2_VL_7B_Instruct.jsonl",28,(15,27)),
     "LLaVA-NeXT":("data/phase82_llavanext.jsonl",32,(18,30)),
     "LLaVA-OneVision":("data/phase82_onevision.jsonl",28,(15,27))}
def prep(src,NL,blk):
    rows=[json.loads(l) for l in open(src)]; X=[];Y=[];G=[];RM=[];gi=0
    for r in rows:
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        if gh*gw!=n: continue
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
        dep=A[blk[0]:blk[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                  np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                  (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        X.append(np.c_[LA,R,geo]); Y.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)]))
        G.append(np.full(n,gi)); m=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: m[1:-1,1:-1]=True
        else: m[:]=True
        RM.append(m.ravel()); gi+=1
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),RM
def hits(P,Y,G,RM):
    return np.array([float(Y[G==gi][int(np.argmax(np.where(RM[gi],P[G==gi],-1e9)))]>=0.5) for gi in np.unique(G)])
def run(X,Y,G,RM,NL,kind,mu=0.0,lam=1.0,seeds=3):
    gs=np.unique(G); P=np.zeros(len(Y))
    D=np.zeros((NL-2,X.shape[1]))
    for i in range(NL-2): D[i,i]=1; D[i,i+1]=-2; D[i,i+2]=1          # roughness over the log-A block only
    for s in range(seeds):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            if kind=="tree":
                m=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(X[tr],Y[tr])
                P[te]+=m.predict(X[te])
            else:
                mu_,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu_)/sd,np.ones(len(tr))]
                Ds=np.c_[D/sd[None,:],np.zeros((D.shape[0],1))]
                A_=Xt.T@Xt+lam*np.eye(Xt.shape[1])+mu*(Ds.T@Ds); A_[-1,-1]-=lam
                w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu_)/sd,np.ones(len(te))]@w
    return hits(P/seeds,Y,G,RM)
rng=np.random.default_rng(181)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
for name,(src,NL,blk) in SRC.items():
    X,Y,G,RM=prep(src,NL,blk); t=run(X,Y,G,RM,NL,"tree"); r0=run(X,Y,G,RM,NL,"ridge",0.0)
    m,lo,hi=ci(r0-t)
    print(f"\n{name}  n={len(RM)}   identical folds/rows/features, 3 seeds")
    print(f"   GBT (incumbent)        {t.mean()*100:5.1f}%")
    print(f"   ridge, closed form     {r0.mean()*100:5.1f}%   vs tree {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'RIDGE WINS' if lo>0 else ('TREE WINS' if hi<0 else 'tie')}")
    prev=r0
    for mu in (1.0,10.0,100.0,1000.0):
        h=run(X,Y,G,RM,NL,"ridge",mu); a,l2,h2=ci(h-r0)
        print(f"   ridge + roughness mu={mu:<6g} {h.mean()*100:5.1f}%   vs free ridge {a:+5.1f} [{l2:+5.1f},{h2:+5.1f}] {'SMOOTHING HURTS' if h2<0 else ''}")
