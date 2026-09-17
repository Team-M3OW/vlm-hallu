import json, sys, numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows])
A=lambda k: np.array([int(np.argmax(r["probs"][k])==r["label"]) for r in rows],float)
T=lambda k: np.array([r["realized_tokens"][k] for r in rows],float)
K=np.array([r["k"] for r in rows]); rng=np.random.default_rng(168)
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
bar=A("uniform@600")
print(f"{f.split('/')[-1]}  n={n}   budget: bar {T('uniform@600').mean():.0f} | prop {T('uniform@300').mean()+T('prop').mean():.0f} | head {T('uniform@300').mean()+T('head@0.25').mean():.0f}")
for s in ["direct_attributes","relative_position"]:
    m=cat==s; print(f"  {s}: k=1 on {100*np.mean(K[m]==1):.0f}%, k=2 on {100*np.mean(K[m]==2):.0f}%, k=3 on {100*np.mean(K[m]==3):.0f}%  (degenerate => DPR)")
print(f"  {'stratum':>18} {'bar':>6} {'head':>7} {'prop':>7} {'propW':>7} {'oracle':>7} |  head-bar             prop-bar              prop-head")
for s in ["direct_attributes","relative_position","ALL"]:
    m=np.ones(n,bool) if s=="ALL" else cat==s
    h,l1,u1=ci((A("head@0.25")-bar)[m]); p,l2,u2=ci((A("prop")-bar)[m]); d,l3,u3=ci((A("prop")-A("head@0.25"))[m])
    print(f"  {s:>18} {bar[m].mean()*100:5.1f}% {A('head@0.25')[m].mean()*100:6.1f}% {A('prop')[m].mean()*100:6.1f}% {A('prop_fixedW')[m].mean()*100:6.1f}% {A('oracle@0.25')[m].mean()*100:6.1f}% | "
          f"{h:+5.1f} [{l1:+5.1f},{u1:+5.1f}]{'✔' if l1>0 else ' '}  {p:+5.1f} [{l2:+5.1f},{u2:+5.1f}]{'✔' if l2>0 else ' '}  {d:+5.1f} [{l3:+5.1f},{u3:+5.1f}]{' ✗' if u3<0 else '  '}")
