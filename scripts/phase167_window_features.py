"""
Phase 167: give the head features that describe the WINDOW, not the point.

DEFECT. The head's target is coverage of the W-window centred at a cell, but every feature it gets
describes the cell itself (28 layer values, 28 ranks, a 3x3 neighbourhood mean). At a ~15x20 grid a
W=0.25 window is ~4x5 cells -- strictly larger than the 3x3 neighbourhood. So no feature tells the
head what the crop would actually contain. For a SINGLE-object question that hardly matters: the best
window is centred on the peak. For a RELATIONAL question the best window is centred BETWEEN two
attended regions, on a cell with little attention of its own -- invisible to the current features.

EVIDENCE THE HEADROOM IS THERE: an oracle window on relational questions beats the equal-compute bar
(SS36: +19.7pp; phase 163 Qwen3: 75.0% vs 65.8%), so a correctly placed single window does cover a
two-object evidence set. The method fails to find it, not because it cannot exist.

FIX (no gate, no question text, same head, same target): add per-cell WINDOWED SUMS -- the total
attention inside the W-window that would actually be cropped if this cell were chosen -- computed with
the same border clamping as coverage(). Arms:
    base      the deployed 65 features
    +win      base + 28 per-layer windowed sums + windowed sum of the block mean + windowed/own ratio
    +win_only base + windowed sum of the block mean + ratio (a 2-feature version, to see if the
              28 per-layer windowed sums are needed or the block-mean window suffices)
PRE-REGISTERED (OOF coverage, both models, same folds as phase 70)
    P1  +win - base on RELATIONAL items: CI clear of zero on BOTH models   <- the failing cell
    GUARD +win - base on SINGLE-object not significantly negative
End-task only if P1 and the guard hold.
"""
import json, sys, numpy as np, warnings
sys.path.insert(0,"/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
warnings.filterwarnings("ignore")
W=0.25; P70.W=W

def win_bounds(cx,cy,w):
    x0,x1,y0,y1=cx-w/2,cx+w/2,cy-w/2,cy+w/2
    if x0<0: x0,x1=0.0,w
    if y0<0: y0,y1=0.0,w
    if x1>1: x0,x1=1-w,1.0
    if y1>1: y0,y1=1-w,1.0
    return x0,y0,x1,y1

def windowed(maps,gh,gw,w=W):
    """maps: (K, gh*gw). Returns (K, gh*gw) sums inside the clamped W-window centred at each cell."""
    K=maps.shape[0]; M=maps.reshape(K,gh,gw)
    I=np.zeros((K,gh+1,gw+1)); I[:,1:,1:]=M.cumsum(1).cumsum(2)
    out=np.zeros((K,gh*gw))
    for c in range(gh*gw):
        cy=((c//gw)+.5)/gh; cx=((c%gw)+.5)/gw
        x0,y0,x1,y1=win_bounds(cx,cy,w)
        i0=int(np.floor(y0*gh)); i1=int(np.ceil(y1*gh)); j0=int(np.floor(x0*gw)); j1=int(np.ceil(x1*gw))
        i0=max(0,i0); j0=max(0,j0); i1=min(gh,max(i1,i0+1)); j1=min(gw,max(j1,j0+1))
        out[:,c]=I[:,i1,j1]-I[:,i0,j1]-I[:,i1,j0]+I[:,i0,j0]
    return out

for name,build,src,NL in [("Qwen3-VL",P70.build,"data/phase30c_attn_maps_all.jsonl",28),
                          ("Qwen2-VL",P80.build,"data/phase74_Qwen2_VL_7B_Instruct.jsonl",28)]:
    r=build(); X,Y,G,DEP,rows=r[0],r[1],r[2],r[3],r[4]
    cat=np.array([q["category"] for q in rows]); single=cat=="direct_attributes"
    WIN=[]; WINB=[]
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; n=q["n_img_tokens"]
        A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        ws=windowed(A,gh,gw)                     # (NL, n) windowed sums per layer
        dep=DEP[gi][None,:]; wdep=windowed(dep,gh,gw)
        WIN.append(np.c_[ws.T, wdep.T, (wdep.T/np.maximum(dep.T,1e-9))])
        WINB.append(np.c_[wdep.T, (wdep.T/np.maximum(dep.T,1e-9))])
    WIN=np.vstack(WIN); WINB=np.vstack(WINB)
    ring={}
    for gi,q in enumerate(rows):
        gh,gw=q["grid"]; m=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: m[1:-1,1:-1]=True
        else: m[:]=True
        ring[gi]=m.ravel()
    def top1(P):
        return np.array([float(Y[G==gi][int(np.argmax(np.where(ring[gi],P[G==gi],-1e9)))]>=P70.COV_HIT) for gi in np.unique(G)])
    res={}
    for tag,XX in [("base (65 feats)",X),("+win (28 layer-window sums + 2)",np.c_[X,WIN]),("+win_only (block-window + ratio)",np.c_[X,WINB])]:
        res[tag]=top1(P70.oof(XX,Y,G,seeds=3))
    dep_hit=np.array([float(Y[G==gi][int(np.argmax(np.where(ring[gi],DEP[gi],-1e9)))]>=P70.COV_HIT) for gi in np.unique(G)])
    base=res["base (65 feats)"]; rng=np.random.default_rng(167)
    def ci(d):
        n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
    print(f"\n{name}  OOF top-1 coverage, W={W}   (deployed argmax {dep_hit.mean()*100:.1f}%)")
    print(f"  {'arm':>34} {'single':>8} {'relational':>11} {'ALL':>7}   relational vs base        single vs base")
    for tag,v in res.items():
        a,lo,hi=ci((v-base)[~single]); s,ls,hs=ci((v-base)[single])
        print(f"  {tag:>34} {v[single].mean()*100:7.1f}% {v[~single].mean()*100:10.1f}% {v.mean()*100:6.1f}%   "
              f"{a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ' '}   {s:+5.1f} [{ls:+5.1f},{hs:+5.1f}]{' ✗' if hs<0 else '  '}")
