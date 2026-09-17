"""Reproducibility of read-out rules across attention-map extractions.
usage: phase171_repro_check.py NEW.jsonl OLD.jsonl qwen3|qwen2 [newkey=q] [oldkey=None]
Per-layer cosine / argmax agreement between the two extractions; block mean, gate max, max-all, head OOF on both."""
import json, numpy as np, sys, os
os.environ.setdefault("OMP_NUM_THREADS","6")
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
from sklearn.ensemble import HistGradientBoostingRegressor; from sklearn.model_selection import GroupKFold
newf,oldf,tag=sys.argv[1:4]; nk=sys.argv[4] if len(sys.argv)>4 else "q"; ok=sys.argv[5] if len(sys.argv)>5 else None
NL=28; GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[tag]; BLK={"qwen3":(16,27),"qwen2":(15,27)}[tag]
old={json.loads(l)["question_id_full"]:json.loads(l) for l in open(oldf)}; new=[json.loads(l) for l in open(newf)]
def A_of(r,key):
    d=r["attn"] if key is None else r["attn"][key]
    A=np.stack([np.asarray(d[f"L{i}"],float) for i in range(NL)]); return A/np.maximum(A.sum(1,keepdims=True),1e-12)
same=[r for r in new if r["question_id_full"] in old and old[r["question_id_full"]]["grid"]==r["grid"]]
print(f"{os.path.basename(newf)} vs {os.path.basename(oldf)}: items in both with identical grid {len(same)}/{len(new)}")
cos=np.zeros((len(same),NL)); ag=np.zeros((len(same),NL))
for i,r in enumerate(same):
    a=A_of(r,nk); b=A_of(old[r["question_id_full"]],ok)
    for L in range(NL): cos[i,L]=a[L]@b[L]/(np.linalg.norm(a[L])*np.linalg.norm(b[L])+1e-12); ag[i,L]=float(np.argmax(a[L])==np.argmax(b[L]))
print("  cosine per layer :", " ".join(f"L{L}:{cos[:,L].mean():.3f}" for L in range(0,NL,3)), f"| gate {cos[:,GATE].mean():.3f} block {cos[:,BLK[0]:BLK[1]].mean():.3f}")
print("  argmax agreement :", " ".join(f"L{L}:{ag[:,L].mean():.2f}" for L in range(0,NL,3)))
def build(rows,key):
    X,Y,G,DEP,RM=[],[],[],[],[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; n=gh*gw; A=A_of(r,key); M=A.reshape(NL,gh,gw); dep=M[BLK[0]:BLK[1]].mean(0)
        R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).reshape(NL,gh,gw)
        pad=np.pad(dep,1,mode="edge"); nb=sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=(yy+.5)/gh,(xx+.5)/gw
        feats=np.concatenate([M.reshape(NL,-1).T,R.reshape(NL,-1).T,nb.reshape(-1,1),dep.reshape(-1,1),fx.reshape(-1,1),fy.reshape(-1,1),
            np.sqrt((fx-.5)**2+(fy-.5)**2).reshape(-1,1),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)).reshape(-1,1),
            (xx==gw-1).astype(float).reshape(-1,1),(yy==gh-1).astype(float).reshape(-1,1),(xx==0).astype(float).reshape(-1,1)],1)
        cov=np.array([P70.coverage(float(fx.flat[i]),float(fy.flat[i]),r["gt_box_frac"]) for i in range(n)])
        rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True
        X.append(feats);Y.append(cov);G.append(np.full(n,gi));DEP.append(A);RM.append(rm.ravel())
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),DEP,RM
def oof(X,Y,G):
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            pos=tr[Y[tr]>0]; negpool=tr[Y[tr]<=0]; neg=rng.choice(negpool,size=min(len(negpool),30*len(np.unique(G[tr]))),replace=False); sub=np.concatenate([pos,neg])
            P[te]+=HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=s).fit(X[sub],Y[sub]).predict(X[te])
    return P/3
def hits(fn,Y,G,DEP,RM,P=None):
    return np.array([float(Y[G==gi][int(np.argmax(np.where(RM[gi],(fn(DEP[gi]) if P is None else P[G==gi]),-1e9)))]>=0.5) for gi in np.unique(G)])
out={}
for label,rows,key in [("OLD",[old[r["question_id_full"]] for r in same],ok),("NEW",same,nk)]:
    X,Y,G,DEP,RM=build(rows,key); h=hits(None,Y,G,DEP,RM,oof(X,Y,G))
    bm=hits(lambda A:A[BLK[0]:BLK[1]].mean(0),Y,G,DEP,RM); gm=hits(lambda A:A[GATE].max(0),Y,G,DEP,RM); am=hits(lambda A:A.max(0),Y,G,DEP,RM)
    out[label]=(bm,gm,am,h); print(f"  {label}: block mean {bm.mean()*100:.1f}  gate max {gm.mean()*100:.1f}  max all layers {am.mean()*100:.1f}  head OOF {h.mean()*100:.1f}")
rng=np.random.default_rng(1)
for name,i in [("block mean",0),("gate max",1),("max all",2),("head",3)]:
    d=out["NEW"][i]-out["OLD"][i]; b=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(6000)])
    print(f"  NEW-OLD {name:>10}: {d.mean()*100:+5.1f} [{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]   per-item agreement {np.mean(out['NEW'][i]==out['OLD'][i])*100:.0f}%")
