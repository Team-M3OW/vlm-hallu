"""
Phase 173 analysis -- LASER-style per-sample layer selection vs our fixed gate / head. Coverage at W=0.25.
Rules (all label-free, no router):
  block mean (deployed)                        gate max (§16D, fixed layers)
  LASER: l*=argmax_l VAQ_l (head-mean contrast), score = ReLU(Aq-Anoq)[l*]
  LASER topK heads: VAQ_l = mean of top-8 heads' ||contrast||, score = contrast at l* averaged over those heads
  raw@l*: raw with-query map at the LASER layer            contrast max over gate layers
  contrast sum over all layers                             per-sample argmax layer of raw by peakiness (no ablation)
Two ablation baselines: noq (instruction kept) and bare (image only).
"""
import json, sys, numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70
P70.W=0.25
tag=sys.argv[1]; NL=28
GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[tag]; BLK={"qwen3":(16,27),"qwen2":(15,27)}[tag]
HEAD={"qwen3":63.4,"qwen2":54.5}[tag]
rows=[json.loads(l) for l in open(f"data/phase173_laser_{tag}.jsonl")]
hz=np.load(f"data/phase173_laser_{tag}_heads.npz")
rng=np.random.default_rng(171)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
res={}; picked={}
def add(k,v): res.setdefault(k,[]).append(v)
for r in rows:
    gh,gw=r["grid"]; n=r["n_img_tokens"]
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    cov=np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)])
    rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; rm=rm.ravel()
    hit=lambda s: float(cov[int(np.argmax(np.where(rm,s,-1e9)))]>=0.5)
    Aq=np.stack([np.asarray(r["attn"]["q"][f"L{i}"],float) for i in range(NL)])
    add("block mean (deployed)",hit(Aq[BLK[0]:BLK[1]].mean(0))); add("gate max (§16D)",hit(Aq[GATE].max(0)))
    for abl in ("noq","bare"):
        An=np.stack([np.asarray(r["attn"][abl][f"L{i}"],float) for i in range(NL)])
        C=np.maximum(Aq-An,0); vaq=np.linalg.norm(C,axis=1); ls=int(np.argmax(vaq)); picked.setdefault(abl,[]).append(ls)
        add(f"[{abl}] LASER contrast @ l*",hit(C[ls])); add(f"[{abl}] raw with-query @ l*",hit(Aq[ls]))
        add(f"[{abl}] contrast sum all layers",hit(C.sum(0))); add(f"[{abl}] contrast max over gate",hit(C[GATE].max(0)))
        add(f"[{abl}] contrast max over all layers",hit(C.max(0)))
        Hq=hz[f"{r['question_id_full']}|q"].astype(float); Hn=hz[f"{r['question_id_full']}|{abl}"].astype(float)   # raw per-head
        Hq=Hq/np.maximum(Hq.sum(-1,keepdims=True),1e-12); Hn=Hn/np.maximum(Hn.sum(-1,keepdims=True),1e-12)          # LASER normalises per head
        Ch=np.maximum(Hq-Hn,0); vh=np.linalg.norm(Ch,axis=2)             # L x heads
        K=8; top=np.argsort(-vh,axis=1)[:,:K]; vl=np.take_along_axis(vh,top,1).mean(1); lh=int(np.argmax(vl))
        add(f"[{abl}] LASER top-{K} heads @ l*",hit(Ch[lh][top[lh]].mean(0)))
    peak=(Aq.max(1)/np.maximum(Aq.mean(1),1e-12)); lp=int(np.argmax(np.where(np.arange(NL)>=GATE[0],peak,-1)))
    add("per-sample peakiest layer >= gate (no ablation)",hit(Aq[lp]))
base=np.array(res["block mean (deployed)"]); N=len(base)
print(f"\n{tag}  n={N}   learned head {HEAD}%")
for k,v in res.items():
    v=np.array(v); m,lo,hi=ci(v-base); print(f"  {k:>48} {v.mean()*100:5.1f}%  vs block {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ('WORSE' if hi<0 else '')}")
for abl,ls in picked.items():
    h=np.bincount(ls,minlength=NL); print(f"  l* histogram [{abl}]: "+" ".join(f"L{i}:{c}" for i,c in enumerate(h) if c))
