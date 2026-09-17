"""
Phase 180 -- THE HEAD AS A SMOOTH SIGNED DEPTH FILTER (structure instead of capacity).

WHY NOT A BIGGER MODEL: phases 45/102/112/130 -- CNN, 9-seed GBT, flip aug, ensembles, pairwise and
listwise objectives, four architectures -- capacity is not the limit at n=191 (CNN is -8.9/-9.4 on Qwen2).
THIS GOES THE OTHER WAY. The head's 28 free layer weights are replaced by a SMOOTH low-order kernel
    w(l) = sum_k c_k * B_k(l/(NL-1)),   B_k = Legendre polynomials, K in {3,4,6,8}
so the depth correction is K parameters, not 28. Score per cell (ridge, closed form, same OOF folds):
    s = sum_l w(l) * log A_l  +  sum_l v(l) * rank_l  +  a*log(3x3 neighbourhood) + geometry(7)
with v(l) kernelised the same way. Arms: tree (incumbent), free-28 linear, kernel-K.
CLAIM UNDER TEST: if kernel-K ties the free-28 head and the tree, the method IS a smooth signed depth
filter -- fewer parameters, one visualisable curve, and its negative lobe is §19's subtraction.
Also reports the fitted w(l) (sign per depth) and the spectrum of the free-28 weights (is it low-frequency?).
"""
import json, sys, numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70
P70.W=0.25
from numpy.polynomial import legendre as LEG
from sklearn.model_selection import GroupKFold
SRC={"Qwen3-VL":("data/phase30c_attn_maps_all.jsonl",28,(16,27)),
     "Qwen2-VL":("data/phase74_Qwen2_VL_7B_Instruct.jsonl",28,(15,27)),
     "LLaVA-NeXT":("data/phase82_llavanext.jsonl",32,(18,30)),
     "LLaVA-OneVision":("data/phase82_onevision.jsonl",28,(15,27))}
TREE={"Qwen3-VL":63.4,"Qwen2-VL":54.5,"LLaVA-NeXT":28.3,"LLaVA-OneVision":37.2}
def basis(NL,K):
    x=np.linspace(-1,1,NL); return np.stack([LEG.legval(x,[0]*k+[1]) for k in range(K)],1)   # NL x K
def prep(src,NL,blk):
    rows=[json.loads(l) for l in open(src)]; items=[]
    for r in rows:
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        if gh*gw!=n: continue
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        LA=np.log(A+1e-12).T                                        # n x NL
        R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T   # n x NL
        dep=A[blk[0]:blk[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                  np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                  (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        cov=np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)])
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        items.append((LA,R,geo,cov,rm.ravel()))
    return items
def fit_eval(items,NL,B,lam=1.0):
    """B = NL x K basis (or identity for free weights). Ridge on standardised design, OOF by item."""
    N=len(items); X=[];Y=[];G=[]
    for gi,(LA,R,geo,cov,rm) in enumerate(items):
        X.append(np.c_[LA@B,R@B,geo]); Y.append(cov); G.append(np.full(len(cov),gi))
    X=np.vstack(X); Y=np.concatenate(Y); G=np.concatenate(G)
    mu,sd=X.mean(0),X.std(0)+1e-9; Xs=(X-mu)/sd
    P=np.zeros(len(Y)); W=[]
    gs=np.unique(G); rng=np.random.default_rng(700); perm={g:i for i,g in enumerate(rng.permutation(gs))}; Gp=np.vectorize(perm.get)(G)
    for tr,te in GroupKFold(5).split(Xs,Y,Gp):
        Xt=np.c_[Xs[tr],np.ones(len(tr))]; yt=Y[tr]
        A_=Xt.T@Xt+lam*np.eye(Xt.shape[1]); A_[-1,-1]-=lam
        w=np.linalg.solve(A_,Xt.T@yt); W.append(w)
        P[te]=np.c_[Xs[te],np.ones(len(te))]@w
    hits=np.array([float(items[gi][3][np.argmax(np.where(items[gi][4],P[G==gi],-1e9))]>=0.5) for gi in gs])
    K=B.shape[1]; wm=np.mean(W,0)
    wl=B@(wm[:K]/sd[:K]); vl=B@(wm[K:2*K]/sd[K:2*K])         # depth kernels in layer space
    return hits,wl,vl
rng=np.random.default_rng(180)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
for name,(src,NL,blk) in SRC.items():
    items=prep(src,NL,blk); N=len(items)
    free,wfree,_=fit_eval(items,NL,np.eye(NL))
    print(f"\n{name}  n={N}   tree(GBT) reference {TREE[name]}%")
    print(f"  {'free 28 layer weights':>26} {free.mean()*100:5.1f}%   (vs tree {free.mean()*100-TREE[name]:+5.1f})")
    for K in (3,4,6,8):
        h,wl,vl=fit_eval(items,NL,basis(NL,K)); m,lo,hi=ci(h-free)
        sgn="".join("+" if x>0 else "-" for x in wl)
        print(f"  {'smooth kernel K='+str(K):>26} {h.mean()*100:5.1f}%   vs free-28 {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]   w(l) sign by depth L0..L{NL-1}: {sgn}")
    sp=np.abs(np.fft.rfft(wfree-wfree.mean()))
    print(f"  free-28 w(l) spectrum (|FFT|, low->high): "+" ".join(f"{v:.2f}" for v in sp[:8])+f"   | energy in first 4 modes: {100*sp[:4].sum()/max(sp.sum(),1e-9):.0f}%")
    print(f"  free-28 w(l) sign by depth: "+"".join("+" if x>0 else "-" for x in wfree))
