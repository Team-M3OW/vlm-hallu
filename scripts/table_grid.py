"""The 4x4 grid: AVR and DWA, 4 models x 4 benchmarks, from phase225 (one harness, identical items).

Scoring is per-item adaptive (see benchmarks.item_correct): mcq<N> by letter logits, `open` by
normalised exact match on a short greedy generation. Token counts are reported per arm because
`dwa_t - bar` is only equal-compute where localise+crop == bar; where it is not, the block-relative
column is the honest one and is marked.
"""
import json,os,sys,glob,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(0)
MODELS=["qwen3_2b","qwen2_7b","internvl3_8b","llava_ov"]
BENCH=["vstar","hr4k","cvbench","realworldqa","textvqa","gqa","docvqa"]
NAMES={"qwen3_2b":"Qwen3-VL-2B","qwen2_7b":"Qwen2-VL-7B","internvl3_8b":"InternVL3-8B","llava_ov":"LLaVA-OV-7B"}
def load(m,b):
    f=f"{D}/data/phase225_{m}_{b}.jsonl"
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
def acc(rows,a):
    v=[B.item_correct(r,a) for r in rows if a in r.get("probs",{}) or a in r.get("preds",{})]
    return 100*float(np.mean(v)) if v else float("nan")
def delta(rows,a,b,Bn=10000):
    d=[B.item_correct(r,a)-B.item_correct(r,b) for r in rows
       if (a in r.get("probs",{}) or a in r.get("preds",{})) and (b in r.get("probs",{}) or b in r.get("preds",{}))]
    if len(d)<8: return "      n/a      "
    d=np.array(d); m=d[rng.integers(0,len(d),(Bn,len(d)))].mean(1)
    lo,hi=np.percentile(m,2.5)*100,np.percentile(m,97.5)*100
    return f"{d.mean()*100:+5.1f}[{lo:+5.1f},{hi:+5.1f}]"+("*" if (lo>0 or hi<0) else " ")
def toks(rows,a):
    v=[r["tokens"][a] for r in rows if a in r.get("tokens",{})]
    return int(np.mean(v)) if v else 0

for title,arms in (("AVR  (uniform@lo = equal-compute bar; uniform@hi = headroom diagnostic)",
                    [("avr","uniform@lo"),("uniform@hi","uniform@lo"),("avr","uniform@hi")]),
                   ("DWA  (block = published read-out; dwa_t = transferred ridge)",
                    [("dwa_t","block"),("dwa_t","uniform@lo"),("block","uniform@lo")])):
    print("="*132); print(title); print("="*132)
    print(f"{'model':14s}{'bench':12s}{'n':>6s}"+"".join(f"{a+'-'+b:>22s}" for a,b in arms)+"   tokens")
    for m in MODELS:
        for b in BENCH:
            rows=load(m,b)
            if not rows: print(f"{NAMES[m]:14s}{b:12s}{'-':>6s}"); continue
            tk=f"lo{toks(rows,'uniform@lo')} hi{toks(rows,'uniform@hi')}"
            if "dwa_t" in rows[0].get("tokens",{}):
                tk+=f" dwa{toks(rows,'dwa_t')}+loc{toks(rows,'localise')}"
            print(f"{NAMES[m]:14s}{b:12s}{len(rows):6d}"+"".join(f"{delta(rows,a,bb):>22s}" for a,bb in arms)+f"   {tk}")
    print()
print("="*132); print("PER-STRATUM (dwa_t - block), the cost-fair placement comparison"); print("="*132)
for m in MODELS:
    for b in BENCH:
        rows=load(m,b)
        if not rows: continue
        for st in sorted({r["stratum"] for r in rows}):
            sub=[r for r in rows if r["stratum"]==st]
            if len(sub)>=25:
                print(f"{NAMES[m]:14s}{b:12s}{st:20s}n={len(sub):5d}  {delta(sub,'dwa_t','block')}")
print("\n* = 95% CI excludes zero.")
print("\nIN-SAMPLE WARNING: on the vstar column dwa_t applies weights FITTED ON V*, so it is")
print("in-sample, not out-of-fold. The out-of-fold V* numbers are phase223 (Qwen) and phase215b")
print("(others) and are what the paper must quote. dwa_t is genuine transfer on hr4k/cvbench/")
print("realworldqa only. Reported here for harness validation (it reproduces the OOF 72.8 on")
print("Qwen3 exactly), NOT as an independent result.")
