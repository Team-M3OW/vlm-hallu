"""Phase 179 analysis. P1: head - best published baseline (max of vicrop_block/vicrop_L14/laser), pooled, both models.
P2: head - gatemax (what supervision buys over the best label-free rule)."""
import json, sys, numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows])
A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows])
arms=list(rows[0]["probs"].keys()); acc={k:A(k) for k in arms}
tok={k:np.mean([r["realized_tokens"][k] for r in rows])+(0 if k.startswith("uniform") else 300) for k in arms}
rng=np.random.default_rng(179)
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]}  n={n}")
print("  budget (incl. localiser): "+"  ".join(f"{k} {tok[k]:.0f}" for k in arms))
drift=[k for k in arms if k!="uniform@300" and abs(tok[k]-tok["uniform@600"])/tok["uniform@600"]>0.10]
if drift: print(f"  ** BUDGET DRIFT >10%: {drift} -- VOID **")
pub=[k for k in arms if k.startswith("vicrop") or k=="laser"]
best=np.maximum.reduce([acc[k] for k in pub]); bestname=max(pub,key=lambda k:acc[k].mean())
print(f"  {'stratum':>12} "+" ".join(f"{k:>13}" for k in arms))
for tag,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    print(f"  {tag:>12} "+" ".join(f"{acc[k][m].mean()*100:12.1f}%" for k in arms))
print(f"\n  best published baseline by mean = {bestname} ({acc[bestname].mean()*100:.1f}%)")
for tag,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    a,lo,hi=ci((acc["head"]-acc[bestname])[m]); b_,l2,h2=ci((acc["head"]-acc["gatemax"])[m]); c,l3,h3=ci((acc["head"]-acc["uniform@600"])[m])
    print(f"  {tag:>12}  P1 head-{bestname} {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else (' ✗' if hi<0 else '  ')}   P2 head-gatemax {b_:+5.1f} [{l2:+5.1f},{h2:+5.1f}]{'✔' if l2>0 else '  '}   head-bar {c:+5.1f} [{l3:+5.1f},{h3:+5.1f}]{'✔' if l3>0 else '  '}")
