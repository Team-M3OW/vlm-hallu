"""
Phase 172 -- ILVAD depth-difference rule (Finding the Correct Visual Evidence Without Forgetting, ICML'26,
arXiv 2605.20965) ported as a label-free cell scorer for depth re-ranking.
    S = sum_l ReLU( B^(l+1) - B^(l) ),   B^(l) = 1[ A^(l) > tau * mean(A^(l)) ]        (paper: binarised, tau ~ 1)
i.e. count how many times a cell becomes NEWLY activated as depth increases. Also a soft variant
    S_soft = sum_l ReLU( A^(l+1) - A^(l) )   (no threshold)
and the same restricted to the question-conditioned layers (l >= gate start). Read-out token: ours (answer
position, last prompt token) -- ILVAD reads the first 10 generated tokens; we only store the prompt-final map.
Coverage at W=0.25 (hit = >=0.5 of box), ring mask, vs deployed block mean; references gate max and learned head.
"""
import json, sys, numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70
P70.W=0.25
SRC={"Qwen3-VL":("data/phase30c_attn_maps_all.jsonl",28,list(range(17,21)),(16,27),63.4),
     "Qwen2-VL":("data/phase74_Qwen2_VL_7B_Instruct.jsonl",28,list(range(19,23)),(15,27),54.5),
     "LLaVA-NeXT":("data/phase82_llavanext.jsonl",32,list(range(18,22)),(18,30),28.3),
     "LLaVA-OneVision":("data/phase82_onevision.jsonl",28,list(range(16,20)),(15,27),37.2)}
rng=np.random.default_rng(170)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
for name,(src,NL,gate,(b0,b1),head) in SRC.items():
    rows=[json.loads(l) for l in open(src)]; res={}
    def add(k,v): res.setdefault(k,[]).append(v)
    for r in rows:
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        if gh*gw!=n: continue
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        cov=np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)])
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rm=rm.ravel()
        hit=lambda s: float(cov[int(np.argmax(np.where(rm,s,-1e9)))]>=0.5)
        g0=gate[0]
        add("block mean (deployed)",hit(A[b0:b1].mean(0)))
        add("gate max (§16D)",hit(A[gate].max(0)))
        for tau in (1.0,2.0):
            B=(A>tau*A.mean(1,keepdims=True)).astype(float)
            add(f"ILVAD binarised tau={tau}",hit(np.maximum(B[1:]-B[:-1],0).sum(0)))
            add(f"ILVAD binarised tau={tau}, l>=gate",hit(np.maximum(B[g0+1:]-B[g0:-1],0).sum(0)))
        add("ILVAD soft (all layers)",hit(np.maximum(A[1:]-A[:-1],0).sum(0)))
        add("ILVAD soft, l>=gate",hit(np.maximum(A[g0+1:]-A[g0:-1],0).sum(0)))
        add("ILVAD soft, l<=gate end (early rise only)",hit(np.maximum(A[1:gate[-1]+1]-A[:gate[-1]],0).sum(0)))
    base=np.array(res["block mean (deployed)"]); N=len(base)
    print(f"\n{name}  n={N}   learned head {head}%")
    for k,v in res.items():
        v=np.array(v); m,lo,hi=ci(v-base); print(f"  {k:>42} {v.mean()*100:5.1f}%  vs block {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ('WORSE' if hi<0 else '')}")
