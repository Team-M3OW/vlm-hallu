"""usage: phase190_analyze.py <file> [strata key]. Works for newmodel (category) and HR-Bench (category single/cross) files."""
import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); rng=np.random.default_rng(190)
cat=np.array([str(r["category"]) for r in rows]); arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
tok={k:np.mean([r["realized_tokens"][k] for r in rows])+(0 if k.startswith("uniform") else 300) for k in arms}
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
ridge=[k for k in arms if k.startswith("ridge")][0]; head=[k for k in arms if k.startswith("head")][0]
print(f"\n{f.split('/')[-1]} n={n}   budgets: "+"  ".join(f"{k} {tok[k]:.0f}" for k in arms))
strata=[(s,cat==s) for s in sorted(set(cat))]+[("ALL",np.ones(n,bool))]
print(f"  {'stratum':>18} "+" ".join(f"{k:>12}" for k in arms))
for t,m in strata: print(f"  {t:>18} "+" ".join(f"{acc[k][m].mean()*100:11.1f}%" for k in arms))
for t,m in strata:
    out=[]
    for nm,a,b in [("P1 ridge-bar",acc[ridge],acc["uniform@600"]),("P2 ridge-head",acc[ridge],acc[head]),("head-bar",acc[head],acc["uniform@600"])]:
        d,lo,hi=ci((a-b)[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>18}  "+"   ".join(out))
