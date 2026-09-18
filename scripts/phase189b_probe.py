"""
Phase 189b: LEARNED PRUNING HEAD probe. For each depth l in {4,8,12,16}: logistic ridge on
  x = [h_l(token) ; h_l(token) * h_l(last prompt token)]  (fp16 -> fp32, standardised)   ->   y = token centre inside GT box
GroupKFold(5) by item x 1 seed (the design matrix is 191*~300 = 57k rows). Metrics, OOF, per item:
  AUC(token in box) ; keep-25% retention = fraction of in-box tokens kept when keeping the top 25% by score ;
  box-hit@25% = at least 50% of in-box tokens retained (the pruning analogue of coverage).
References at the same depth: attention ranking (last-token attention at l), random. Reference across depth: attention at L16.
Reading: a learned head "pays" at depth l if its box-hit@25% approaches attention@L16 (the free-prune reference) while the
same-depth attention ranking does not. usage: phase189b_probe.py qwen3|qwen2
"""
import json,sys,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
tag=sys.argv[1]; Z=np.load(f"data/phase189a_hidden_{tag}.npz"); meta=json.load(open(f"data/phase189a_hidden_{tag}_meta.json")); N=len(meta)
def inbox(m):
    gh,gw=m["grid"]; yy,xx=np.mgrid[0:gh,0:gw]; fx,fy=((xx+.5)/gw).ravel(),((yy+.5)/gh).ravel(); x0,y0,x1,y1=m["gt_box_frac"]
    return ((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(int)
Y=[inbox(m) for m in meta]; G=np.concatenate([np.full(len(y),i) for i,y in enumerate(Y)]); Yc=np.concatenate(Y)
rng=np.random.default_rng(189)
def per_item(score,keep=0.25):
    ret=[];hit=[]
    for i in range(N):
        s=score[G==i]; y=Y[i]; k=max(1,int(round(keep*len(y)))); kept=np.zeros(len(y),bool); kept[np.argsort(-s)[:k]]=True
        if y.sum()==0: continue
        r=(kept&(y==1)).sum()/y.sum(); ret.append(r); hit.append(float(r>=0.5))
    return np.mean(ret),np.mean(hit),np.array(hit)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return f"[{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]"
print(f"{tag}: {N} items, {len(Yc)} tokens, in-box rate {Yc.mean()*100:.1f}%")
refA16=np.concatenate([Z[f"a_L16_{i}"] for i in range(N)]); _,h16,H16=per_item(refA16)
print(f"  reference: attention@L16 ranking  box-hit@25% {h16*100:.1f}%   | random {per_item(rng.random(len(Yc)))[1]*100:.1f}%")
print(f"  {'depth':>6} {'probe AUC':>9} {'probe retain':>12} {'probe box-hit@25%':>17} {'attn box-hit@25%':>16}   probe - attn@L16")
for l in (4,8,12,16):
    Hs=[Z[f"h_L{l}_{i}"].astype(np.float32) for i in range(N)]; Qs=[Z[f"q_L{l}_{i}"].astype(np.float32) for i in range(N)]
    X=np.vstack([np.c_[h, h*q[None,:]] for h,q in zip(Hs,Qs)]); mu,sd=X.mean(0),X.std(0)+1e-6; X=(X-mu)/sd
    P=np.zeros(len(Yc))
    for tr,te in GroupKFold(5).split(X,Yc,G):
        clf=LogisticRegression(C=0.05,max_iter=300,class_weight="balanced").fit(X[tr],Yc[tr]); P[te]=clf.decision_function(X[te])
    auc=roc_auc_score(Yc,P); pr_,ph,PH=per_item(P); A=np.concatenate([Z[f"a_L{l}_{i}"] for i in range(N)]); _,ah,_=per_item(A)
    print(f"  {'L'+str(l):>6} {auc:9.3f} {pr_*100:11.1f}% {ph*100:16.1f}% {ah*100:15.1f}%   {(ph-h16)*100:+5.1f} {ci(PH-H16)}")
