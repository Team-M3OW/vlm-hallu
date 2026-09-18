import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows]); rng=np.random.default_rng(1888)
arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
TL={k:np.mean([r["token_layers"][k] for r in rows]) for k in arms}
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]} n={n}   token-layers % of bar: "+"  ".join(f"{k} {TL[k]/TL['uniform@600']*100:.0f}%" for k in arms))
print(f"  {'stratum':>11} "+" ".join(f"{k:>13}" for k in arms))
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    print(f"  {t:>11} "+" ".join(f"{acc[k][m].mean()*100:12.1f}%" for k in arms))
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    out=[]
    for nm,a,b in [("P1 400x20-300",acc["ridge400x20"],acc["ridge300"]),("P2 400x20-bar",acc["ridge400x20"],acc["uniform@600"]),("diag full-x20",acc["ridge400full"],acc["ridge400x20"])]:
        d,lo,hi=ci((a-b)[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>11}  "+"   ".join(out))
