"""Paired contrasts on phase-173 maps: query-contrast rules vs gate max (fixed) and vs block mean. usage: phase173_paired.py qwen3|qwen2 [suffix]"""
import json, numpy as np, sys
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
tag=sys.argv[1]; suf=sys.argv[2] if len(sys.argv)>2 else ""; NL=28; rng=np.random.default_rng(5)
GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[tag]; BLK={"qwen3":(16,27),"qwen2":(15,27)}[tag]
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return f"{d.mean()*100:+5.1f} [{np.percentile(b,2.5)*100:+5.1f},{np.percentile(b,97.5)*100:+5.1f}]"
rows=[json.loads(l) for l in open(f"data/phase173_laser_{tag}{suf}.jsonl")]; R={}
for r in rows:
    gh,gw=r["grid"]; n=gh*gw; yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    cov=np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)]); rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; rm=rm.ravel()
    hit=lambda s: float(cov[int(np.argmax(np.where(rm,s,-1e9)))]>=0.5)
    A=lambda k: (lambda M: M/np.maximum(M.sum(1,keepdims=True),1e-12))(np.stack([np.asarray(r["attn"][k][f"L{i}"],float) for i in range(NL)]))
    Aq=A("q"); C=np.maximum(Aq-A("noq"),0); Cb=np.maximum(Aq-A("bare"),0)
    for k,v in {"block":Aq[BLK[0]:BLK[1]].mean(0),"gate max":Aq[GATE].max(0),"max all":Aq.max(0),
                "contrast(noq) max gate":C[GATE].max(0),"contrast(noq) max all":C.max(0),"contrast(noq) sum all":C.sum(0),
                "contrast(bare) max gate":Cb[GATE].max(0),"contrast(bare) max all":Cb.max(0)}.items(): R.setdefault(k,[]).append(hit(v))
R={k:np.array(v) for k,v in R.items()}
print(f"{tag}{suf} n={len(rows)}: "+"  ".join(f"{k} {v.mean()*100:.1f}" for k,v in R.items()))
for k in R:
    if k in ("block","gate max"): continue
    print(f"  {k:>26} - gate max {ci(R[k]-R['gate max'])}    - block {ci(R[k]-R['block'])}")
