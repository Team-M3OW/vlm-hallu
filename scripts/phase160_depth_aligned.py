"""
Phase 160 (architecture fix 1): DEPTH-ALIGNED features -- make the head task-agnostic.

SS16E: a head trained on 3,000 TextVQA boxes transfers to V*Bench BELOW the deployed argmax. SS16E
addendum: the mechanism is that question-conditioned localisation switches on at L11-13 for text
reading and at L16 for small-object attributes, so a TextVQA head trusts layers that are still
question-blind on V*Bench.

FIX: index the depth profile relative to the task's switch-on layer s, i.e. feature j = attention at
layer (s + j), j = -6..+11 (18 layers), instead of absolute layer numbers. The head then reads
"depth since localisation began". s is a per-(model,task) constant.

TWO WAYS TO SET s -- both reported:
    s_gt   from the gt_pct profile (uses boxes; the hypothesis test)          V*Bench 16, TextVQA 12
    s_free from a label-free proxy: the first layer whose ring-masked max-cell attention share exceeds
           2x the median over layers 0-8 (attention "sharpening") -- computed per task from maps only.
PRE-REGISTERED
    P1  aligned TextVQA-only head -> V*Bench zero-shot, vs the unaligned transfer (38.2 / 37.2) and vs
        the deployed argmax (39.3 / 46.6). CI clear over the argmax on BOTH windows -> the head is
        task-agnostic under alignment; parity with the V*Bench OOF head (53.4 / 64.4) is the strong form.
    P2  aligned V*Bench OOF head must not lose to the unaligned one (a sanity cost check).
"""
import json, sys, numpy as np, warnings
sys.path.insert(0,"/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
D="/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
VS,TX=f"{D}/phase30c_attn_maps_all.jsonl",f"{D}/phase133_textvqa_attn_qwen3.jsonl"
NL=28; LO,HI=-6,12   # relative window s-6 .. s+11 (18 layers)

def load(path):
    rows=[json.loads(l) for l in open(path)]
    A=[]; meta=[]
    for r in rows:
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        a=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        A.append(a); meta.append((gh,gw,n,r["gt_box_frac"]))
    return A,meta

def s_free(A,meta):
    """label-free switch-on: first layer where the ring-masked max share > 2x the median of L0-8."""
    share=np.zeros(NL); cnt=0
    for a,(gh,gw,n,_) in zip(A,meta):
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rm=rm.ravel()
        share+=np.array([a[L][rm].max() for L in range(NL)]); cnt+=1
    share/=cnt; base=np.median(share[:9])
    s=int(np.argmax(share>2*base)) if (share>2*base).any() else 16
    return s, share

def feats(A,meta,s,W):
    P70.W=W; X,Y,G,DEP,RING=[],[],[],[],[]
    idx=[min(max(s+j,0),NL-1) for j in range(LO,HI)]
    b0,b1=int(.57*NL),int(.93*NL)+1
    for gi,(a,(gh,gw,n,gt)) in enumerate(zip(A,meta)):
        M=a[idx].reshape(len(idx),gh,gw); dep=a[b0:b1].mean(0).reshape(gh,gw)
        R=(np.argsort(np.argsort(-a[idx],axis=1),axis=1)/max(n-1,1)).reshape(len(idx),gh,gw)
        pad=np.pad(dep,1,mode="edge"); nb=sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=(yy+.5)/gh,(xx+.5)/gw
        X.append(np.concatenate([M.reshape(len(idx),-1).T,R.reshape(len(idx),-1).T,nb.reshape(-1,1),dep.reshape(-1,1),fx.reshape(-1,1),fy.reshape(-1,1),
            np.sqrt((fx-.5)**2+(fy-.5)**2).reshape(-1,1),np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)).reshape(-1,1),
            (xx==gw-1).astype(float).reshape(-1,1),(yy==gh-1).astype(float).reshape(-1,1),(xx==0).astype(float).reshape(-1,1)],1))
        Y.append(np.array([P70.coverage(float(fx.flat[i]),float(fy.flat[i]),gt) for i in range(n)])); G.append(np.full(n,gi)); DEP.append(dep.ravel())
        m=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: m[1:-1,1:-1]=True
        else: m[:]=True
        RING.append(m.ravel())
    return np.vstack(X),np.concatenate(Y),np.concatenate(G),DEP,RING

def top1(P,Y,G,RING): return np.array([float(Y[G==gi][int(np.argmax(np.where(RING[gi],P[G==gi],-1e9)))]>=P70.COV_HIT) for gi in np.unique(G)])
def fit_all(X,Y,G,seed):
    rng=np.random.default_rng(700+seed); pos=np.where(Y>0)[0]; negpool=np.where(Y<=0)[0]
    neg=rng.choice(negpool,size=min(len(negpool),30*len(np.unique(G))),replace=False); sub=np.concatenate([pos,neg])
    return HistGradientBoostingRegressor(max_depth=4,max_iter=150,learning_rate=0.10,random_state=seed).fit(X[sub],Y[sub])
rng=np.random.default_rng(160)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100

Av,mv=load(VS); At,mt=load(TX)
sv,shv=s_free(Av,mv); st,sht=s_free(At,mt)
print(f"label-free switch-on: V*Bench s={sv}  TextVQA s={st}   (gt-based: 16 / 12)")
print("  V*Bench max-share by layer:", np.round(shv,3)[8:20]); print("  TextVQA max-share by layer:", np.round(sht,3)[8:20])
for W in [0.15,0.25]:
    print(f"\n=== W={W} ===")
    Xv,Yv,Gv,Dv,Rv=feats(Av,mv,16,W)          # absolute-equivalent for V*Bench when s=16 fixed for both
    dep=np.array([float(Yv[Gv==gi][int(np.argmax(np.where(Rv[gi],Dv[gi],-1e9)))]>=P70.COV_HIT) for gi in np.unique(Gv)])
    print(f"  deployed argmax on V*Bench: {dep.mean()*100:.1f}%")
    for tag,(s_v,s_t) in {"UNALIGNED (same absolute layers, s=16 both)":(16,16),
                          "ALIGNED, s from gt profile (16 / 12)":(16,12),
                          f"ALIGNED, s label-free ({sv} / {st})":(sv,st)}.items():
        Xv,Yv,Gv,Dv,Rv=feats(Av,mv,s_v,W); Xt,Yt,Gt,_,_=feats(At,mt,s_t,W)
        ms=[fit_all(Xt,Yt,Gt,s) for s in range(3)]; tr=top1(np.mean([m.predict(Xv) for m in ms],0),Yv,Gv,Rv)
        oof=top1(P70.oof(Xv,Yv,Gv,seeds=1),Yv,Gv,Rv)
        a,lo,hi=ci(tr-dep)
        print(f"  {tag:>44}: TextVQA->V*Bench zero-shot {tr.mean()*100:5.1f}%  vs argmax {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ''} | V*Bench OOF head {oof.mean()*100:5.1f}%")
