"""Phase 189c: controls for the hidden-state pruning probe + top-1 placement.
Variants at each depth: full [h ; h*q] | token-only [h] | geometry-only [fx,fy,dist,edge,minedge] | attention (same depth).
Metrics: box-hit@25% (pruning) and TOP-1 COVERAGE@0.25 ring-masked (placement; ridge reference 63.9 / 55.0, gate-max 56.0 / 51.8)."""
import json,sys,numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
tag=sys.argv[1]; Z=np.load(f"data/phase189a_hidden_{tag}.npz"); meta=json.load(open(f"data/phase189a_hidden_{tag}_meta.json")); N=len(meta)
RIDGE={"qwen3":63.9,"qwen2":55.0}[tag]; GM={"qwen3":56.0,"qwen2":51.8}[tag]
geo=[];Y=[];COV=[];RM=[]
for m in meta:
    gh,gw=m["grid"]; yy,xx=np.mgrid[0:gh,0:gw]; fx,fy=((xx+.5)/gw).ravel(),((yy+.5)/gh).ravel(); x0,y0,x1,y1=m["gt_box_frac"]
    Y.append(((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(int))
    geo.append(np.c_[fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),(xx.ravel()==gw-1),(yy.ravel()==gh-1),(xx.ravel()==0),(yy.ravel()==0)].astype(np.float32))
    COV.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),m["gt_box_frac"]) for i in range(len(fx))])); r=np.zeros((gh,gw),bool); r[1:-1,1:-1]=True; RM.append(r.ravel())
G=np.concatenate([np.full(len(y),i) for i,y in enumerate(Y)]); Yc=np.concatenate(Y); rng=np.random.default_rng(1893)
def metrics(score):
    hit=[];cov=[]
    for i in range(N):
        s=score[G==i]; y=Y[i]; k=max(1,int(round(.25*len(y)))); kept=np.zeros(len(y),bool); kept[np.argsort(-s)[:k]]=True
        if y.sum()>0: hit.append(float((kept&(y==1)).sum()/y.sum()>=0.5))
        cov.append(float(COV[i][int(np.argmax(np.where(RM[i],s,-1e9)))]>=0.5))
    return np.mean(hit),np.array(cov)
def oof(X):
    mu,sd=X.mean(0),X.std(0)+1e-6; X=(X-mu)/sd; P=np.zeros(len(Yc))
    for tr,te in GroupKFold(5).split(X,Yc,G):
        P[te]=LogisticRegression(C=0.05,max_iter=300,class_weight="balanced").fit(X[tr],Yc[tr]).decision_function(X[te])
    return P
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return f"[{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]"
print(f"{tag}: references  ridge top-1 coverage {RIDGE}%   gate-max {GM}%")
Xg=np.vstack(geo); h,cv=metrics(oof(Xg)); print(f"  {'geometry-only':>22} {'-':>5}  box-hit@25% {h*100:5.1f}%   top-1 coverage {cv.mean()*100:5.1f}%")
print(f"  {'variant':>22} {'depth':>5}  {'box-hit@25%':>11}   {'top-1 cov':>9}   cov - ridge")
for l in (4,8,12,16):
    Hs=[Z[f"h_L{l}_{i}"].astype(np.float32) for i in range(N)]; Qs=[Z[f"q_L{l}_{i}"].astype(np.float32) for i in range(N)]
    A=np.concatenate([Z[f"a_L{l}_{i}"] for i in range(N)]); ha,ca=metrics(A)
    print(f"  {'attention':>22} L{l:<4} {ha*100:11.1f}%   {ca.mean()*100:8.1f}%   {(ca.mean()*100-RIDGE):+5.1f}")
    for name,X in (("token-only [h]",np.vstack(Hs)),("full [h ; h*q]",np.vstack([np.c_[h_,h_*q[None,:]] for h_,q in zip(Hs,Qs)]))):
        hp,cp=metrics(oof(X)); print(f"  {name:>22} L{l:<4} {hp*100:11.1f}%   {cp.mean()*100:8.1f}%   {(cp.mean()*100-RIDGE):+5.1f}")
