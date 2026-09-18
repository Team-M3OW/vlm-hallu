"""Phase 188a analysis: does a higher-resolution localisation pass PLACE better? Ridge OOF coverage@0.25 (hit>=0.5), same
folds, on 300-token stored maps vs 450 unpruned vs 450 pruned(L16,k=.10). Also gate-max and block-mean on each map set.
Decision (pre-registered): 188b end-task only if ridge coverage on 450-pruned >= 300-stored on BOTH models (CI not sig. negative)."""
import json,sys,numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
from sklearn.model_selection import GroupKFold
tag=sys.argv[1]; NL=28
BLK={"qwen3":(16,27),"qwen2":(15,27)}[tag]; GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[tag]
SRC={"300 stored":{"qwen3":"data/phase30c_attn_maps_all.jsonl","qwen2":"data/phase74_Qwen2_VL_7B_Instruct.jsonl"}[tag],
     "450 unpruned":f"data/phase188a_loc450_unpruned_{tag}.jsonl","450 pruned L16":f"data/phase188a_loc450_pruned_{tag}.jsonl"}
def prep(rows):
    X,Y,G,RM,dep,gm=[],[],[],[],[],[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
        d=A[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(d,1,mode="edge"); nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),(xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        X.append(np.c_[LA,R,geo]); Y.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)])); G.append(np.full(n,gi))
        m=np.zeros((gh,gw),bool); m[1:-1,1:-1]=True; RM.append(m.ravel()); dep.append(d.ravel()); gm.append(A[GATE].max(0))
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),RM,dep,gm
def ridge_oof(X,Y,G,lam=1.0):
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]; A_=Xt.T@Xt+lam*np.eye(Xt.shape[1]); A_[-1,-1]-=lam
            w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
    return P/3
def hits(S,Y,G,RM): return np.array([float(Y[G==gi][int(np.argmax(np.where(RM[gi],S[gi] if isinstance(S,list) else S[G==gi],-1e9)))]>=0.5) for gi in np.unique(G)])
rng=np.random.default_rng(188)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(8000)]); return f"{d.mean()*100:+5.1f} [{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]"
res={}; ids={}
for name,f in SRC.items():
    rows=[json.loads(l) for l in open(f)]; ids[name]=[r["question_id_full"] for r in rows]
    X,Y,G,RM,dep,gm=prep(rows); res[name]={"ridge":hits(ridge_oof(X,Y,G),Y,G,RM),"gate max":hits(gm,Y,G,RM),"block mean":hits(dep,Y,G,RM),"cat":np.array([r["category"] for r in rows])}
common=sorted(set.intersection(*[set(v) for v in ids.values()]))
print(f"{tag}: common items {len(common)}")
print(f"  {'map set':>16} {'ridge':>7} {'gate max':>9} {'block':>7}   ridge vs 300-stored (ALL | single | relational)")
for name in SRC:
    idx=[ids[name].index(q) for q in common]; idx0=[ids["300 stored"].index(q) for q in common]
    r=res[name]; r0=res["300 stored"]; cat=r["cat"][idx]; s=cat=="direct_attributes"
    d=r["ridge"][idx]-r0["ridge"][idx0]
    print(f"  {name:>16} {r['ridge'][idx].mean()*100:6.1f}% {r['gate max'][idx].mean()*100:8.1f}% {r['block mean'][idx].mean()*100:6.1f}%   {ci(d)} | {ci(d[s])} | {ci(d[~s])}")
