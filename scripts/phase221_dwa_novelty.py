"""
Phase 221 (CPU): two ways to make DWA a RULE rather than a per-model fit.

A) CROSS-MODEL TRANSFER. Fit the depth weights on one checkpoint, apply to the other with NO refitting.
   Features are standardised using the TARGET model's own statistics, which needs no labels. If this
   holds, the depth profile is a property of the architecture class, not of the checkpoint.
B) ZERO-SUPERVISION DEPTH RULE. §53's non-negative fit selects [4,5,17,19] (Qwen3) and [19,21] (Qwen2)
   -- both inside the post-boundary read-out band. Test a FIXED rule with no fitting at all: average the
   k layers at given depth FRACTIONS. If a fixed fraction matches the fitted weights, DWA needs no boxes.
Baselines: each model's own fitted ridge, and the block mean.
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
def build(which):
    fn,BLK=MAPS[which]
    rows=[json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    X=[];Y=[];G=[];meta=[];A_all=[]
    for i,r in enumerate(rows):
        gh,gw=r["grid"]; n=gh*gw; NL=28
        a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)])
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        LA=np.log(a+1e-12).T; R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(n-1,1)).T
        dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                  np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                  (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        X.append(np.c_[LA,R,geo]); A_all.append(a)
        Y.append(np.array([cov(float(fx[j]),float(fy[j]),r["gt_box_frac"]) for j in range(n)]))
        G.append(np.full(n,i)); meta.append((gh,gw,r["gt_box_frac"],r["category"]))
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),meta,A_all
def fit_w(X,Y,G):
    ws=[]
    for s in (700,701,702):
        rng=np.random.default_rng(s); gs=np.unique(G); perm={g:i for i,g in enumerate(rng.permutation(gs))}
        Gp=np.vectorize(perm.get)(G)
        for tr,_ in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            ws.append(np.linalg.solve(A_,Xt.T@Y[tr]))
    return np.mean(ws,0)
def evaluate(X,meta,w):
    mu,sd=X.mean(0),X.std(0)+1e-9              # TARGET model's own stats: no labels needed
    P=np.c_[(X-mu)/sd,np.ones(len(X))]@w
    off=0; out=[]
    for gh,gw,gt,cat in meta:
        n=gh*gw; s=P[off:off+n]; off+=n
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        j=int(np.argmax(np.where(rm.ravel(),s,-1e9)))
        out.append((cov((j%gw+.5)/gw,(j//gw+.5)/gh,gt),cat))
    return out
DAT={k:build(k) for k in MAPS}
Wt={k:fit_w(DAT[k][0],DAT[k][1],DAT[k][2]) for k in MAPS}
def summ(res):
    a=np.mean([c for c,_ in res]); s=np.mean([c for c,k in res if k=="direct_attributes"])
    x=np.mean([c for c,k in res if k!="direct_attributes"]); return a,s,x
print("A) CROSS-MODEL TRANSFER of the fitted depth weights (coverage; no refit on the target)")
for tgt in MAPS:
    X,Y,G,meta,A_all=DAT[tgt]
    own=summ(evaluate(X,meta,Wt[tgt]))
    src=[k for k in MAPS if k!=tgt][0]
    tr=summ(evaluate(X,meta,Wt[src]))
    fn,BLK=MAPS[tgt]; bm=[]
    for a,(gh,gw,gt,cat) in zip(A_all,meta):
        dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw)
        rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True
        j=int(np.argmax(np.where(rm.ravel(),dep.ravel(),-1e9)))
        bm.append((cov((j%gw+.5)/gw,(j//gw+.5)/gh,gt),cat))
    bms=summ(bm)
    print(f"   target {tgt}:  own fit {own[0]:.3f}   weights from {src} {tr[0]:.3f}"
          f"   block-mean {bms[0]:.3f}   transfer retains {100*(tr[0]-bms[0])/max(own[0]-bms[0],1e-9):.0f}% of the gain")
print("\nB) ZERO-SUPERVISION fixed-depth rule (mean of k layers at given depth fractions, NO fitting)")
for tgt in MAPS:
    X,Y,G,meta,A_all=DAT[tgt]; fn,BLK=MAPS[tgt]
    own=summ(evaluate(X,meta,Wt[tgt]))[0]
    for fr in ([0.61],[0.68],[0.61,0.68],[0.61,0.64,0.68],[0.61,0.68,0.75]):
        Ls=sorted({min(27,int(round(f*28))) for f in fr}); res=[]
        for a,(gh,gw,gt,cat) in zip(A_all,meta):
            dep=a[Ls].mean(0).reshape(gh,gw)
            rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True
            j=int(np.argmax(np.where(rm.ravel(),dep.ravel(),-1e9)))
            res.append((cov((j%gw+.5)/gw,(j//gw+.5)/gh,gt),cat))
        print(f"   {tgt}: layers {Ls} (fractions {fr})  cov {summ(res)[0]:.3f}   vs own fit {own:.3f}")
