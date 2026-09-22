"""Fit the 63-feature DWA ridge on V* for any family, from a phase214 map dump, and save the
weight vector for cross-benchmark transfer (phase224).

Same estimator as fig_ridge_scores.py: OOF GroupKFold(5) x 3 group permutations, lambda=1,
unpenalised intercept, features standardised per fold. We save the fold-mean weights; phase224
re-standardises on the target benchmark (label-free), so only the coefficients transfer.

Usage: fit_ridge_weights.py <tag>     # tag = llava_ov | llava_next | gemma3_4b
"""
import json,sys,os,numpy as np
from sklearn.model_selection import GroupKFold
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG=sys.argv[1]; W=0.25
SRC=f"{D}/data/phase214_dwa_{TAG}.jsonl"
rows=[json.loads(l) for l in open(SRC) if l.strip()]
rows=[r for r in rows if "attn" in r and "grid" in r and r.get("gt")]
NL=rows[0]["nl"]; BLK=(round(0.57*NL),NL-1)
print(f"{TAG}: {len(rows)} items, NL={NL}, depth band L{BLK[0]}-{BLK[1]-1}")
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
X=[];Y=[];G=[]
for gi,q in enumerate(rows):
    gh,gw=q["grid"]; nc=gh*gw
    a=np.stack([np.asarray(q["attn"][f"L{l}"],float) for l in range(NL)])
    if a.shape[1]!=nc: continue
    a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    LA=np.log(a+1e-12).T; R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(nc-1,1)).T
    dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
    nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
    geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
              np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
              (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
    X.append(np.c_[LA,R,geo]); G.append(np.full(nc,gi))
    Y.append(np.array([cov(float(fx[i]),float(fy[i]),q["gt"]) for i in range(nc)]))
X=np.vstack(X);Y=np.concatenate(Y);G=np.concatenate(G)
print(f"  {X.shape[0]} cells, {X.shape[1]} features ({2*NL}+7)")
Wsum=np.zeros(X.shape[1]+1); nfit=0; P=np.zeros(len(Y)); gs=np.unique(G)
for s in range(3):
    rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}
    Gp=np.vectorize(perm.get)(G)
    for tr,te in GroupKFold(5).split(X,Y,Gp):
        mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]
        A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
        w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
        Wsum+=w; nfit+=1
P/=3
# OOF sanity: does the fitted map beat the plain depth-block arg-max on coverage?
off=0; cr=[]; cb=[]
for q in rows:
    gh,gw=q["grid"]; nc=gh*gw
    a=np.stack([np.asarray(q["attn"][f"L{l}"],float) for l in range(NL)])
    if a.shape[1]!=nc: continue
    a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
    sc=P[off:off+nc].reshape(gh,gw); off+=nc
    dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw)
    rm=np.zeros((gh,gw),bool)
    if gh>2 and gw>2: rm[1:-1,1:-1]=True
    else: rm[:]=True
    for M,acc in ((sc,cr),(dep,cb)):
        iy,ix=np.unravel_index(np.argmax(np.where(rm,M,-1e9)),M.shape)
        acc.append(cov((ix+.5)/gw,(iy+.5)/gh,q["gt"]))
print(f"  OOF coverage: ridge {np.mean(cr):.3f}  block {np.mean(cb):.3f}  (ridge-block {np.mean(cr)-np.mean(cb):+.3f})")
out=f"{D}/data/fig_ridge_w_{TAG}.npy"; np.save(out,Wsum/nfit)
print(f"  -> {out}  dim={len(Wsum)}")
