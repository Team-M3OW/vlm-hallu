"""Phase 189e: does the hidden-state probe TRANSFER across encoding resolution? Train on the 300-token dump (all items),
apply to the 600/900-token dumps of the same items (no item overlap issue: evaluation is cross-resolution, and we also
report within-resolution OOF). Metrics at each depth: box-hit@25% and top-1 coverage@0.25 (ring-masked).
Also prints the compute accounting for 'prune at L_p keeping 25%' at each encoding budget vs the 16,800 bar."""
import json,sys,numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
tag=sys.argv[1]; NL=28
def load(E):
    suf="" if E==300 else f"_{E}"; Z=np.load(f"data/phase189a_hidden_{tag}{suf}.npz"); meta=json.load(open(f"data/phase189a_hidden_{tag}{suf}_meta.json")); return Z,meta
def geom(meta):
    Y=[];COV=[];RM=[]
    for m in meta:
        gh,gw=m["grid"]; yy,xx=np.mgrid[0:gh,0:gw]; fx,fy=((xx+.5)/gw).ravel(),((yy+.5)/gh).ravel(); x0,y0,x1,y1=m["gt_box_frac"]
        Y.append(((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(int)); COV.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),m["gt_box_frac"]) for i in range(len(fx))]))
        r=np.zeros((gh,gw),bool); r[1:-1,1:-1]=True; RM.append(r.ravel())
    return Y,COV,RM
def feats(Z,N,l): return np.vstack([np.c_[Z[f"h_L{l}_{i}"].astype(np.float32),Z[f"h_L{l}_{i}"].astype(np.float32)*Z[f"q_L{l}_{i}"].astype(np.float32)[None,:]] for i in range(N)])
def metrics(score,G,Y,COV,RM):
    hit=[];cov=[]
    for i in range(len(Y)):
        s=score[G==i]; y=Y[i]; k=max(1,int(round(.25*len(y)))); kept=np.zeros(len(y),bool); kept[np.argsort(-s)[:k]]=True
        if y.sum()>0: hit.append(float((kept&(y==1)).sum()/y.sum()>=0.5))
        cov.append(float(COV[i][int(np.argmax(np.where(RM[i],s,-1e9)))]>=0.5))
    return np.mean(hit)*100,np.mean(cov)*100
Z3,m3=load(300); N3=len(m3); Y3,C3,R3=geom(m3); G3=np.concatenate([np.full(len(y),i) for i,y in enumerate(Y3)]); Yc3=np.concatenate(Y3)
print(f"{tag}: bar = 16,800 token-layers.  Compute for encode@E, prune to 25% at L_p:  E*(p+1) + 0.25*E*(27-p)")
for E in (600,900):
    try: Z,m=load(E)
    except FileNotFoundError: print(f"  E={E}: dump not found"); continue
    N=len(m); Y,C,R=geom(m); G=np.concatenate([np.full(len(y),i) for i,y in enumerate(Y)]); Yc=np.concatenate(Y)
    print(f"  ===== encode@{E} (n={N}) =====")
    for l in (4,8,12,16):
        X3=feats(Z3,N3,l); mu,sd=X3.mean(0),X3.std(0)+1e-6
        clf=LogisticRegression(C=0.05,max_iter=300,class_weight="balanced").fit((X3-mu)/sd,Yc3)
        X=feats(Z,N,l); Pt=clf.decision_function((X-mu)/sd); ht,ct=metrics(Pt,G,Y,C,R)
        P=np.zeros(len(Yc)); mu2,sd2=X.mean(0),X.std(0)+1e-6; Xs=(X-mu2)/sd2
        for tr,te in GroupKFold(5).split(Xs,Yc,G): P[te]=LogisticRegression(C=0.05,max_iter=300,class_weight="balanced").fit(Xs[tr],Yc[tr]).decision_function(Xs[te])
        hw,cw=metrics(P,G,Y,C,R); A=np.concatenate([Z[f"a_L{l}_{i}"] for i in range(N)]); ha,ca=metrics(A,G,Y,C,R)
        TL=E*(l+1)+0.25*E*(27-l)
        print(f"    L{l:<3} probe trained@300 -> box-hit {ht:5.1f}%  top-1 cov {ct:5.1f}%  |  within-{E} OOF: box-hit {hw:5.1f}%  cov {cw:5.1f}%  |  attention@L{l}: box-hit {ha:5.1f}%  cov {ca:5.1f}%  |  prune@L{l} TL {TL:,.0f} ({TL/16800*100:.0f}% of bar)")
