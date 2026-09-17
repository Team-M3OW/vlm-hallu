"""
Phase 175 -- LLM "layer disagreement" techniques ported to the attention read-out, label-free, on stored maps.
  CLD   Confident Layer Decoding (2606.21906): backward scan from the gate end, stop at the first entropy valley
        (min-entropy layer in the window); read that layer.                        [per-sample layer]
  AttJSD Attention-guided layer selection (2607.23067): premature layer = argmax JSD(A_l, A_final) in the gate
        window; read A_l ('Attention-JSD'), and 'Attention-Entropy-Min' = min-entropy layer over all l>=gate start.
  ASL   Adaptive layer selection by rank stability (2601.07667): per layer, Spearman(rank_l, rank_{l+1}); pick the
        layer where ranking first stabilises (max stability in window); also stability-weighted rank sum.
  EMA   InertiaKV-style order-preserving aggregation (2609.03515): exponential moving average over DEPTH of the
        within-layer ranks (later layers weigh more), read the EMA at the gate end.
  SLED  Self-evolution (2411.02433): S = A_gate + alpha * (A_gate - A_early), early = mean of layers < gate start;
        alpha in {0.5, 1, 2}.  (DoLa-style contrast of mature vs premature map.)
References: block mean (deployed), gate max (§16D), learned head.  Coverage@0.25, ring mask, 4 architectures.
"""
import json, sys, numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
from scipy.stats import spearmanr
SRC={"Qwen3-VL":("data/phase30c_attn_maps_all.jsonl",28,list(range(17,21)),(16,27),63.4),
     "Qwen2-VL":("data/phase74_Qwen2_VL_7B_Instruct.jsonl",28,list(range(19,23)),(15,27),54.5),
     "LLaVA-NeXT":("data/phase82_llavanext.jsonl",32,list(range(18,22)),(18,30),28.3),
     "LLaVA-OneVision":("data/phase82_onevision.jsonl",28,list(range(16,20)),(15,27),37.2)}
rng=np.random.default_rng(175)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
def H(p): p=p/np.maximum(p.sum(),1e-12); return float(-(p*np.log(p+1e-12)).sum())
def jsd(p,q): p=p/np.maximum(p.sum(),1e-12); q=q/np.maximum(q.sum(),1e-12); m=(p+q)/2; return 0.5*float((p*np.log((p+1e-12)/(m+1e-12))).sum()+(q*np.log((q+1e-12)/(m+1e-12))).sum())
for name,(src,NL,gate,(b0,b1),head) in SRC.items():
    rows=[json.loads(l) for l in open(src)]; res={}; picks={}
    def add(k,v): res.setdefault(k,[]).append(v)
    g0,g1=gate[0],gate[-1]
    for r in rows:
        gh,gw=r["grid"]; n=r["n_img_tokens"]
        if gh*gw!=n: continue
        A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
        cov=np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)])
        rm=np.zeros((gh,gw),bool)
        if gh>2 and gw>2: rm[1:-1,1:-1]=True
        else: rm[:]=True
        rm=rm.ravel(); hit=lambda s: float(cov[int(np.argmax(np.where(rm,s,-1e9)))]>=0.5)
        Ar=np.where(rm[None,:],A,0)                                  # ring-masked for statistics
        add("block mean (deployed)",hit(A[b0:b1].mean(0))); add("gate max (§16D)",hit(A[gate].max(0)))
        ent=np.array([H(Ar[l]) for l in range(NL)])
        # CLD: backward scan from gate end within window [g0-2, g1]: first local minimum of entropy
        lo=max(g0-2,1); l=g1
        while l>lo and ent[l-1]<ent[l]: l-=1
        add("CLD entropy-valley layer (backward scan)",hit(A[l])); picks.setdefault("CLD",[]).append(l)
        lmin=int(lo+np.argmin(ent[lo:g1+1])); add("Att-Entropy-Min layer (window)",hit(A[lmin]))
        lmin2=int(g0+np.argmin(ent[g0:])); add("Att-Entropy-Min layer (l>=gate start)",hit(A[lmin2])); picks.setdefault("EntMin",[]).append(lmin2)
        js=np.array([jsd(Ar[l],Ar[NL-1]) for l in range(NL)]); lj=int(lo+np.argmax(js[lo:g1+1])); add("Att-JSD layer vs final (window)",hit(A[lj])); picks.setdefault("JSD",[]).append(lj)
        R=np.argsort(np.argsort(-A,axis=1),axis=1).astype(float)      # within-layer ranks (0 = strongest)
        stab=np.array([spearmanr(R[l][rm],R[l+1][rm]).correlation for l in range(NL-1)]+[0.0]); stab=np.nan_to_num(stab)
        ls=int(lo+np.argmax(stab[lo:g1+1])); add("ASL rank-stability layer (window)",hit(A[ls])); picks.setdefault("ASL",[]).append(ls)
        w=np.clip(stab,0,None)[g0:]; add("ASL stability-weighted mean (l>=gate)",hit((w[:,None]*A[g0:]).sum(0)/max(w.sum(),1e-9)))
        for beta in (0.5,0.8):
            e=-R[0]
            for l in range(1,g1+1): e=beta*e+(1-beta)*(-R[l])
            add(f"depth-EMA of ranks beta={beta} (to gate end)",hit(e))
        early=A[:g0].mean(0); Ag=A[gate].mean(0)
        for a in (0.5,1.0,2.0): add(f"SLED extrapolation alpha={a}",hit(Ag+a*(Ag-early)))
        add("SLED extrapolation on max (alpha=1)",hit(A[gate].max(0)+1.0*(A[gate].max(0)-early)))
    base=np.array(res["block mean (deployed)"]); N=len(base); gm=np.array(res["gate max (§16D)"])
    print(f"\n{name}  n={N}   learned head {head}%   gate {gate}")
    for k,v in res.items():
        v=np.array(v); m,lo_,hi=ci(v-base); m2,lo2,hi2=ci(v-gm)
        print(f"  {k:>46} {v.mean()*100:5.1f}%  vs block {m:+5.1f} [{lo_:+5.1f},{hi:+5.1f}] {'CLEARS' if lo_>0 else ('WORSE' if hi<0 else '      ')}   vs gate max {m2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]")
    print("  picked-layer medians: "+", ".join(f"{k} L{int(np.median(v))}" for k,v in picks.items()))
