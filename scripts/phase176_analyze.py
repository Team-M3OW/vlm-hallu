"""Phase 176 analysis: per-layer read-out sensitivity profile -> label-free layer weights -> read-out rules on stored maps."""
import json, sys, numpy as np
sys.path.insert(0,"scripts"); import phase70_rerank_head as P70; P70.W=0.25
tag=sys.argv[1]; NL=28
GATE={"qwen3":list(range(17,21)),"qwen2":list(range(19,23))}[tag]; BLK={"qwen3":(16,27),"qwen2":(15,27)}[tag]
SRC={"qwen3":"data/phase30c_attn_maps_all.jsonl","qwen2":"data/phase74_Qwen2_VL_7B_Instruct.jsonl"}[tag]
S=json.load(open(f"data/phase176_sens_{tag}.json")); K=np.array([r["kl"] for r in S]); KL_=np.array([r["kl_letters"] for r in S]); F=np.array([r["flip"] for r in S])
cal={r["question_id_full"] for r in S}
print(f"{tag}: calibration n={len(S)}  kl_all mean {np.mean([r['kl_all'] for r in S]):.4f}  flip_all {np.mean([r['flip_all'] for r in S]):.2f}  | per-layer sum of KL {K.sum(1).mean():.4f}")
print("  per-layer KL  :"," ".join(f"L{l}:{K[:,l].mean():.4f}" for l in range(NL)))
print("  per-layer flip:"," ".join(f"L{l}:{F[:,l].mean():.2f}" for l in range(NL)))
top=np.argsort(-K.mean(0))[:6]; print("  most sensitive layers:",[f"L{l}" for l in top], "| gate", GATE)
w=K.mean(0); w=w/w.sum(); wl=KL_.mean(0); wl=wl/wl.sum()
rows=[json.loads(l) for l in open(SRC)]; res={}; rng=np.random.default_rng(176)
def ci(d):
    n=len(d); b=np.array([d[rng.integers(0,n,n)].mean() for _ in range(6000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
def add(k,v): res.setdefault(k,[]).append(v)
for r in rows:
    if r["question_id_full"] in cal: continue                      # evaluate on the non-calibration items
    gh,gw=r["grid"]; n=r["n_img_tokens"]
    if gh*gw!=n: continue
    A=np.stack([np.asarray(r["attn"][f"L{i}"],float) for i in range(NL)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    cov=np.array([P70.coverage(float(fx[i]),float(fy[i]),r["gt_box_frac"]) for i in range(n)]); rm=np.zeros((gh,gw),bool); rm[1:-1,1:-1]=True; rm=rm.ravel()
    hit=lambda s: float(cov[int(np.argmax(np.where(rm,s,-1e9)))]>=0.5)
    add("block mean (deployed)",hit(A[BLK[0]:BLK[1]].mean(0))); add("gate max (§16D)",hit(A[GATE].max(0)))
    add("sensitivity-weighted mean (all layers)",hit((w[:,None]*A).sum(0))); add("sensitivity-weighted mean (letters KL)",hit((wl[:,None]*A).sum(0)))
    for k in (2,4,6): add(f"max over top-{k} sensitive layers",hit(A[np.argsort(-w)[:k]].max(0)))
    add("max over layers with above-median sensitivity",hit(A[w>np.median(w)].max(0)))
    add("signed: gate max − mean of top-4 sensitive (if disjoint)",hit(A[GATE].max(0)-A[[l for l in np.argsort(-w)[:4] if l not in GATE] or GATE].mean(0)))
base=np.array(res["block mean (deployed)"]); gm=np.array(res["gate max (§16D)"]); print(f"  evaluation n={len(base)} (non-calibration items)")
for k,v in res.items():
    v=np.array(v); m,lo,hi=ci(v-base); m2,lo2,hi2=ci(v-gm); print(f"  {k:>52} {v.mean()*100:5.1f}%  vs block {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]   vs gate max {m2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]")
