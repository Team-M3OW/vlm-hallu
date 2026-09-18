"""Phase 188c (on disk): EARLY-EXIT RIDGE. Ridge features restricted to layers <= Lmax (log A, ranks) + geometry; coverage@0.25 OOF.
Map sets: 300 stored vs 450 unpruned. If ridge(L<=16) on 450 >= ridge(all 28) on 300, the localiser can run at 450 tokens
and EXIT at L16: 450*17 + 300*28 = 16,050 token-layers <= 16,800. usage: phase188c_earlyexit_ridge.py qwen3|qwen2"""
import json,sys,numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
from sklearn.model_selection import GroupKFold
tag=sys.argv[1]; NL=28; BLK={"qwen3":(16,27),"qwen2":(15,27)}[tag]
SRC={"300 stored":{"qwen3":"data/phase30c_attn_maps_all.jsonl","qwen2":"data/phase74_Qwen2_VL_7B_Instruct.jsonl"}[tag],"450 unpruned":f"data/phase188a_loc450_unpruned_{tag}.jsonl"}
def prep(rows,Lmax):
    X,Y,G,RM=[],[],[],[]
    for gi,r in enumerate(rows):
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(Lmax+1)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
        b0,b1=min(BLK[0],Lmax),min(BLK[1],Lmax+1); d=A[b0:b1].mean(0).reshape(gh,gw); pad=np.pad(d,1,mode="edge"); nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),(xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        X.append(np.c_[LA,R,geo]); Y.append(np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)])); G.append(np.full(n,gi))
        m=np.zeros((gh,gw),bool); m[1:-1,1:-1]=True; RM.append(m.ravel())
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),RM
def ridge_hits(X,Y,G,RM,lam=1.0):
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in range(3):
        rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]; A_=Xt.T@Xt+lam*np.eye(Xt.shape[1]); A_[-1,-1]-=lam
            w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
    P/=3; return np.array([float(Y[G==gi][int(np.argmax(np.where(RM[gi],P[G==gi],-1e9)))]>=0.5) for gi in gs])
rng=np.random.default_rng(1889)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(8000)]); return f"{d.mean()*100:+5.1f} [{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]"
rows={k:[json.loads(l) for l in open(f)] for k,f in SRC.items()}
common=sorted(set(r["question_id_full"] for r in rows["300 stored"])&set(r["question_id_full"] for r in rows["450 unpruned"]))
rows={k:sorted([r for r in v if r["question_id_full"] in common],key=lambda r:r["question_id_full"]) for k,v in rows.items()}
cat=np.array([r["category"] for r in rows["300 stored"]]); s=cat=="direct_attributes"
ref=ridge_hits(*prep(rows["300 stored"],27))
print(f"{tag}: n={len(common)}   reference = ridge, all 28 layers, 300 tokens: {ref.mean()*100:.1f}%")
print(f"  {'map set':>13} {'Lmax':>5} {'loc TL':>7} {'total TL':>9} {'ridge':>7}   vs reference (ALL | single | relational)")
for name in SRC:
    for Lmax in (27,20,16,12):
        h=ridge_hits(*prep(rows[name],Lmax)); E=300 if name.startswith("300") else 450; loc=E*(Lmax+1); tot=loc+300*28
        flag="  <= bar" if tot<=16800 else "  OVER"
        print(f"  {name:>13} {Lmax:5d} {loc:7d} {tot:9d}{flag:>8} {h.mean()*100:6.1f}%   {ci(h-ref)} | {ci((h-ref)[s])} | {ci((h-ref)[~s])}")
