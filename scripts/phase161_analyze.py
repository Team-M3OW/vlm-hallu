"""Analyzer for the question-agnostic window arms (161 span / 163 mass). usage: phase161_analyze.py <file> <arm>"""
import json, sys, numpy as np
path, arm = sys.argv[1], sys.argv[2]
rows=[json.loads(l) for l in open(path)]; n=len(rows); cat=np.array([r["category"] for r in rows])
A=lambda k: np.array([int(np.argmax(r["probs"][k])==r["label"]) for r in rows],float)
T=lambda k: np.array([r["realized_tokens"][k] for r in rows],float)
rng=np.random.default_rng(161)
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
bar=A("uniform@600"); k=np.array([r.get("k",1) for r in rows]); area=np.array([r.get(f"{arm}_area", r.get("span_area",0.0625)) for r in rows])
print(f"{path.split('/')[-1]}  n={n}  budget bar {T('uniform@600').mean():.0f} vs method {T('uniform@300').mean()+T(arm).mean():.0f}")
for s in ["direct_attributes","relative_position"]:
    m=cat==s; print(f"  {s}: k=2 on {100*np.mean(k[m]==2):.0f}% of items; median {arm} window area {np.median(area[m]):.3f} (W=0.25 -> 0.0625)")
print(f"  {'stratum':>18} {'bar':>6} {'head@.25':>9} {arm:>9} {'oracle':>7} |  head-bar            {arm}-bar             {arm}-head")
for s in ["direct_attributes","relative_position","ALL"]:
    m=np.ones(n,bool) if s=="ALL" else cat==s
    h,lo,hi=ci((A("head@0.25")-bar)[m]); w,lo2,hi2=ci((A(arm)-bar)[m]); d,lo3,hi3=ci((A(arm)-A("head@0.25"))[m])
    print(f"  {s:>18} {bar[m].mean()*100:5.1f}% {A('head@0.25')[m].mean()*100:8.1f}% {A(arm)[m].mean()*100:8.1f}% {A('oracle@0.25')[m].mean()*100:6.1f}% | "
          f"{h:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ' '}  {w:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]{'✔' if lo2>0 else ' '}  {d:+5.1f} [{lo3:+5.1f},{hi3:+5.1f}]")
