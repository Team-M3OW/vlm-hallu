"""
Phase 169 -- EARLY-EXIT LOCALISER: does the head need the late layers?
If the read-out can be taken at the gate layers (L17-20 Qwen3, L19-22 Qwen2, ~0.57 depth LLaVA), pass 1
stops there (~70% of a forward) and the method's total compute falls below the uniform@600 bar.
SS19 says the head SUBTRACTS late layers (L20-26), so truncation may cost coverage. Test on disk:
build the SS14 head from layers 0..Lmax only (depth profile, ranks, neighbourhood + block mean over
the AVAILABLE block layers, geometry), OOF GroupKFold(5) x 3 seeds, W=0.25, hit = coverage>=0.5.
Cutoffs: full, gate end, gate start, half depth.  Bootstrap CI vs full-depth head (paired).
"""
import json, sys, os, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","6")
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70
P70.W=0.25
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
SRC={"Qwen3-VL":("data/phase30c_attn_maps_all.jsonl",28,(16,27),[27,20,17,14]),
     "Qwen2-VL":("data/phase74_Qwen2_VL_7B_Instruct.jsonl",28,(15,27),[27,22,19,14]),
     "LLaVA-NeXT":("data/phase82_llavanext.jsonl",32,(18,30),[31,22,18,16])}
def build(rows,NL,blk,Lmax):
    X,Y,G,DEP,RM=[],[],[],[],[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        if gh*gw!=n: continue
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(Lmax+1)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        nL=A.shape[0]; M=A.reshape(nL,gh,gw)
        b0=min(blk[0],Lmax); b1=min(blk[1],Lmax+1); dep=M[b0:b1].mean(0)
        R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).reshape(nL,gh,gw)
        pad=np.pad(dep,1,mode="edge"); nb=sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=(yy+.5)/gh,(xx+.5)/gw
        feats=np.concatenate([M.reshape(nL,-1).T,R.reshape(nL,-1).T,nb.reshape(-1,1),dep.reshape(-1,1),
            fx.reshape(-1,1),fy.reshape(-1,1),np.sqrt((fx-.5)**2+(fy-.5)**2).reshape(-1,1),
            np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)).reshape(-1,1),
            (xx==gw-1).astype(float).reshape(-1,1),(yy==gh-1).astype(float).reshape(-1,1),(xx==0).astype(float).reshape(-1,1)],1)
        cov=np.array([P70.coverage(float(fx.flat[i]),float(fy.flat[i]),r["gt_box_frac"]) for i in range(n)])
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        X.append(feats);Y.append(cov);G.append(np.full(n,len(DEP)));DEP.append(dep.ravel());RM.append(rm.ravel())
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),DEP,RM
def oof(X,Y,G,seeds=3):
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in range(seeds):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            pos=tr[Y[tr]>0]; negpool=tr[Y[tr]<=0]; neg=rng.choice(negpool,size=min(len(negpool),30*len(np.unique(G[tr]))),replace=False)
            sub=np.concatenate([pos,neg]); m=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s)
            m.fit(X[sub],Y[sub]); P[te]+=m.predict(X[te])
    return P/seeds
def hits(P,Y,G,RM):
    out=[]
    for gi in np.unique(G):
        m=G==gi; s=np.where(RM[gi],P[m],-1e9); out.append(float(Y[m][int(np.argmax(s))]>=0.5))
    return np.array(out)
rng=np.random.default_rng(169)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
for name,(src,NL,blk,cuts) in SRC.items():
    rows=[json.loads(l) for l in open(src)]; ref=None
    print(f"\n{name}  block {blk}")
    for Lmax in cuts:
        X,Y,G,DEP,RM=build(rows,NL,blk,Lmax); h=hits(oof(X,Y,G),Y,G,RM)
        dep=hits(np.concatenate(DEP),Y,G,RM)
        if ref is None: ref=h; print(f"  L<= {Lmax:2d} (full)   head {h.mean()*100:5.1f}%   deployed block mean {dep.mean()*100:5.1f}%   n={len(h)}")
        else:
            m,lo,hi=ci(h-ref); frac=(Lmax+1)/NL
            print(f"  L<= {Lmax:2d} ({frac:.2f} of depth)  head {h.mean()*100:5.1f}%   vs full {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'WORSE' if hi<0 else ''}   (block mean over avail. {dep.mean()*100:5.1f}%)")
