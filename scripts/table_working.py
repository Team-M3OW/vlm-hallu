"""4x4 AUDIT RESTRICTED TO CONDITIONS WHERE CROPPING IS THE RIGHT OPERATION.

The full grid mixes benchmarks whose questions need the WHOLE scene (CV-Bench Count/Depth/
Distance/Relation, RealworldQA's spatial MCQ) with ones about a local property of one region.
Cropping discards 93.75% of the image, so it cannot help on the former -- measured directly:
on CV-Bench Count the published read-out under-counts by 1.29 objects (0.68 predicted vs 1.97 gold).

This table asks the narrower, fairer question: WHERE CROPPING IS APPROPRIATE, does it pay, and
does the depth-weighted read-out beat the published one?

Strata used (all single-object / local-property by the benchmark's own labels):
  vstar/direct_attributes   attribute of one named object
  hr4k/single               single-instance, 4032px images
  realworldqa/open          free-form attribute questions
"""
import json,os,sys,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(0)
MODELS=[("qwen3_2b","Qwen3-VL-2B"),("qwen2_7b","Qwen2-VL-7B"),
        ("internvl3_8b","InternVL3-8B"),("llava_ov","LLaVA-OV-7B")]
CONDS=[("vstar","direct_attributes","V*/attribute"),
       ("hr4k","single","HR-4k/single"),
       ("realworldqa","open","RWQA/open")]
def load(m,b):
    f=f"{D}/data/phase225_{m}_{b}.jsonl"
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
def acc(rs,a):
    v=[B.item_correct(r,a) for r in rs if a in r.get("probs",{}) or a in r.get("preds",{})]
    return 100*np.mean(v) if v else float("nan")
def ci(rs,a,b):
    ok=[r for r in rs if (a in r.get("probs",{}) or a in r.get("preds",{})) and (b in r.get("probs",{}) or b in r.get("preds",{}))]
    if len(ok)<8: return "        n/a       "
    d=np.array([B.item_correct(r,a)-B.item_correct(r,b) for r in ok])
    m=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    lo,hi=np.percentile(m,2.5)*100,np.percentile(m,97.5)*100
    return f"{d.mean()*100:+5.1f}[{lo:+5.1f},{hi:+5.1f}]"+("*" if (lo>0 or hi<0) else " ")
for title,(a,b) in (("Is cropping worth it here?  dwa_t - bar",("dwa_t","uniform@lo")),
                    ("Does our read-out beat the published one?  dwa_t - block",("dwa_t","block")),
                    ("Does the PUBLISHED rule work here?  block - bar",("block","uniform@lo"))):
    print("="*104); print(title); print("="*104)
    print(f"{'model':14s}"+"".join(f"{c[2]:>22s}" for c in CONDS))
    for mk,nm in MODELS:
        row=f"{nm:14s}"
        for bench,strat,_ in CONDS:
            rs=[r for r in load(mk,bench) if r.get("stratum")==strat]
            row+=f"{ci(rs,a,b):>22s}"
        print(row)
    print()
print("* = 95% CI excludes zero.  n per cell: V*/attribute 115, HR-4k/single 400, RWQA/open 337.")
