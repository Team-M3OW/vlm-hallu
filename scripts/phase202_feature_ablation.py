"""
Phase 202 (CPU): WHICH features carry TWR's advantage? Leave-one-group-in / leave-one-group-out.

Feature groups (the deployed TWR uses all of them):
  A  log A_l            28  per-layer log attention        <- the "depth filter" the paper names
  R  rank_l             28  per-layer within-map rank      <- monotone-invariant version of A
  N  log_nb              1  3x3 neighbourhood mean of the block-mean map   } spatial
  C  r_center,dist_edge  2  centre prior / distance to edge                } priors
  P  fx,fy               2  raw position
  S  is_last_col/row     2  explicit serialisation-sink indicators

PRE-REGISTERED PREDICTION (written before running): if TWR's nuisance robustness comes from the SPATIAL
PRIORS rather than from the signed depth weighting, then dropping {N,C} costs most of the advantage while
dropping {S} costs ~nothing, and "A only" lands near the block-mean baseline.
Maps UNMASKED throughout, so the ring mask cannot do the work. n = items on the modal grid, both models.
"""
import json, numpy as np
from sklearn.model_selection import GroupKFold
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25
MAPS={"qwen3":("phase30c_attn_maps_all.jsonl",(16,27)),"qwen2":("phase74_Qwen2_VL_7B_Instruct.jsonl",(15,27))}
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
def oof(X,Y,G,seeds=(700,701,702)):
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in seeds:
        rng=np.random.default_rng(s); perm={g:i for i,g in enumerate(rng.permutation(gs))}
        Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
    return P/len(seeds)
for which,(fn,BLK) in MAPS.items():
    rows=[json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0=max({tuple(r["grid"]) for r in rows},key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]; gh,gw=g0; NL=28; n=len(sub); ncell=gh*gw
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    blocks={}; Xall=[]; Y=[]; G=[]
    for i,r in enumerate(sub):
        a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)])
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        LA=np.log(a+1e-12).T
        R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(ncell-1,1)).T
        dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
        N=np.log(nb+1e-12)[:,None]
        C=np.c_[np.sqrt((fx-.5)**2+(fy-.5)**2),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy))]
        P_=np.c_[fx,fy]; S=np.c_[(xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        Xall.append(np.c_[LA,R,N,C,P_,S])
        Y.append(np.array([cov(float(fx[c]),float(fy[c]),r["gt_box_frac"]) for c in range(ncell)]))
        G.append(np.full(ncell,i))
    X=np.vstack(Xall); Y=np.concatenate(Y); G=np.concatenate(G)
    idx={"A":list(range(0,28)),"R":list(range(28,56)),"N":[56],"C":[57,58],"P":[59,60],"S":[61,62]}
    covgrid=Y.reshape(n,ncell)
    def run(groups):
        cols=sum((idx[g] for g in groups),[])
        Pm=oof(X[:,cols],Y,G).reshape(n,ncell)
        c=np.array([covgrid[i,int(np.argmax(Pm[i]))] for i in range(n)])
        return c.mean(),100*np.mean(c>=.5)
    bm=[]
    for i,r in enumerate(sub):
        a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)])
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        bm.append(covgrid[i,int(np.argmax(a[BLK[0]:BLK[1]].mean(0)))])
    bm=np.array(bm)
    print(f"=== {which}  n={n} on the modal {gh}x{gw} grid, UNMASKED maps")
    print(f"   {'block-mean arg-max (baseline)':34s} mean-cov {bm.mean():.3f}   >=0.5 {100*np.mean(bm>=.5):4.1f}%")
    for nm,grp in [("A only (the depth filter)",["A"]),("R only (ranks)",["R"]),
                   ("N+C only (spatial priors)",["N","C"]),("A+R",["A","R"]),
                   ("A+R+S (no spatial priors)",["A","R","S"]),
                   ("ALL minus S",["A","R","N","C","P"]),("ALL (deployed TWR)",["A","R","N","C","P","S"])]:
        m,p=run(grp); print(f"   {nm:34s} mean-cov {m:.3f}   >=0.5 {p:4.1f}%   vs block {m-bm.mean():+.3f}")
    print()
