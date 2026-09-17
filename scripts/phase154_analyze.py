import json, sys, numpy as np
D="/home/kavinder/ARNABI_ARSH/vlm-hallu/data"; TAG=sys.argv[1]; W=0.25
rows=[json.loads(l) for l in open(f"{D}/phase154_{TAG}_endtask.jsonl")]; n=len(rows); cat=np.array([r["category"] for r in rows])
A=lambda k: np.array([int(np.argmax(r["probs"][k])==r["label"]) for r in rows],float)
T=lambda k: np.array([r["realized_tokens"][k] for r in rows],float)
rng=np.random.default_rng(154)
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
bar=A("uniform@600"); print(f"{TAG}: n={n}  budget bar {T('uniform@600').mean():.0f} vs method {T('uniform@300').mean()+T(f'head@{W}').mean():.0f}")
print(f"  {'arm':>12}  {'pooled':>7} {'single':>7} {'relational':>10}")
for k in ["uniform@300","uniform@600",f"argmax@{W}",f"head@{W}",f"rand@{W}",f"oracle@{W}"]:
    a=A(k); print(f"  {k:>12}  {a.mean()*100:6.1f}% {a[cat=='direct_attributes'].mean()*100:6.1f}% {a[cat=='relative_position'].mean()*100:9.1f}%")
for s in ["direct_attributes","relative_position","ALL"]:
    m=np.ones(n,bool) if s=="ALL" else cat==s
    for k in [f"head@{W}",f"argmax@{W}"]:
        v,lo,hi=ci((A(k)-bar)[m]); print(f"  {s:>18} {k:>12} - bar: {v:+6.1f} [{lo:+5.1f},{hi:+5.1f}] {'CLEARS' if lo>0 else ''}")
