"""
Phase 201 (CPU): is the conventional read-out's failure CAUSED by the item-independent component?

Ablation, definition fixed in §49 and not changed: for each layer, remove the projection of that item's
map onto the item-MEAN map:   A'_l = A_l - <A_l, mbar_l>/<mbar_l, mbar_l> * mbar_l
Then recompute placement, on UNMASKED maps (the ring mask already removes most of the nuisance).

PRE-REGISTERED PREDICTIONS
  Q1  block-mean coverage on ablated maps >> on raw maps, and approaches its ring-masked value
      (raw 0.139/0.313 -> ring-masked 0.475/0.392). Predicted: ablation recovers most of that gap.
  Q2  TWR coverage on ablated maps ~= on raw maps (it is already 0.642/0.562 unmasked), i.e. TWR does
      not need the ablation. Predicted |delta| <= 0.05.
  Q3  If Q1 and Q2 both hold, the conventional read-out's failure IS the nuisance and TWR is already
      robust to it. If TWR's ADVANTAGE survives ablation, its advantage is NOT mainly nuisance-robustness.
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
def ridge_oof(Xr,Yr,Gr,seeds=(700,701,702)):
    Pr=np.zeros(len(Yr)); gs=np.unique(Gr)
    for s in seeds:
        rng=np.random.default_rng(s); perm={g:i for i,g in enumerate(rng.permutation(gs))}
        Gp=np.vectorize(perm.get)(Gr)
        for tr,te in GroupKFold(5).split(Xr,Yr,Gp):
            mu,sd=Xr[tr].mean(0),Xr[tr].std(0)+1e-9; Xt=np.c_[(Xr[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            w=np.linalg.solve(A_,Xt.T@Yr[tr]); Pr[te]+=np.c_[(Xr[te]-mu)/sd,np.ones(len(te))]@w
    return Pr/len(seeds)
for which,(fn,BLK) in MAPS.items():
    rows=[json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0=max({tuple(r["grid"]) for r in rows},key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]; gh,gw=g0; NL=28; n=len(sub)
    A=np.stack([np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)]) for r in sub])
    A=A/np.maximum(A.sum(2,keepdims=True),1e-12)
    mbar=A.mean(0)                                        # (NL, cells)
    den=np.maximum((mbar*mbar).sum(1),1e-12)              # (NL,)
    coef=(A*mbar[None]).sum(2)/den[None]                  # (n, NL)
    Aab=A-coef[:,:,None]*mbar[None]                       # ablated
    Aab=Aab-Aab.min(axis=2,keepdims=True)+1e-9            # keep positive for log features
    Aab=Aab/np.maximum(Aab.sum(2,keepdims=True),1e-12)
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    gts=[r["gt_box_frac"] for r in sub]
    covgrid=np.array([[cov(float(fx[c]),float(fy[c]),g) for c in range(gh*gw)] for g in gts])
    def blockcov(M):
        out=[]
        for i in range(n):
            dep=M[i,BLK[0]:BLK[1]].mean(0); j=int(np.argmax(dep)); out.append(covgrid[i,j])
        return np.array(out)
    def feats(M):
        Xs=[]
        for i in range(n):
            a=M[i]; LA=np.log(a+1e-12).T
            R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(gh*gw-1,1)).T
            dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
            nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
            geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                      np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                      (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
            Xs.append(np.c_[LA,R,geo])
        return np.vstack(Xs)
    Yr=covgrid.ravel(); Gr=np.repeat(np.arange(n),gh*gw)
    def ridgecov(M):
        P=ridge_oof(feats(M),Yr,Gr).reshape(n,gh*gw)
        return np.array([covgrid[i,int(np.argmax(P[i]))] for i in range(n)])
    res={}
    for nm,M in (("raw",A),("nuisance-ablated",Aab)):
        b=blockcov(M); r=ridgecov(M)
        res[nm]=(b,r)
        print(f"  {which} {nm:17s}: block-mean mean-cov {b.mean():.3f} (>=0.5 {100*np.mean(b>=.5):4.1f}%)   "
              f"TWR mean-cov {r.mean():.3f} (>=0.5 {100*np.mean(r>=.5):4.1f}%)   TWR-block {r.mean()-b.mean():+.3f}")
    db=res["nuisance-ablated"][0].mean()-res["raw"][0].mean()
    dr=res["nuisance-ablated"][1].mean()-res["raw"][1].mean()
    print(f"  => Q1 block-mean delta {db:+.3f}   Q2 TWR delta {dr:+.3f} (predicted |.|<=0.05)"
          f"   Q3 TWR advantage raw {res['raw'][1].mean()-res['raw'][0].mean():+.3f} -> "
          f"ablated {res['nuisance-ablated'][1].mean()-res['nuisance-ablated'][0].mean():+.3f}\n")
