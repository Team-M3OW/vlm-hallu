"""Phase 185 analysis. P1 tsr900-bar pooled both models; GUARD relational not sig. negative; S1 tsr900-fastv900; S2 tsr600-bar."""
import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows]); rng=np.random.default_rng(185)
arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
TL={k:np.mean([r["token_layers"][k] for r in rows]) for k in arms}; tok={k:np.mean([r["realized_tokens"][k] for r in rows]) for k in arms}
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]}  n={n}"); print("  arm            tokens  token-layers  (bar=uniform@600)")
for k in arms: print(f"  {k:>12} {tok[k]:7.0f} {TL[k]:12.0f}   {TL[k]/TL['uniform@600']*100:5.0f}% of bar")
print(f"\n  {'stratum':>11} "+" ".join(f"{k:>12}" for k in arms))
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    print(f"  {t:>11} "+" ".join(f"{acc[k][m].mean()*100:11.1f}%" for k in arms))
print()
for t,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    out=[]
    for nm,a,b in [("P1 tsr900-bar",acc["tsr900"],acc["uniform@600"]),("S1 tsr900-fastv",acc["tsr900"],acc["fastv900"]),
                   ("tsr900-rand",acc["tsr900"],acc["tsr900_rand"]),("S2 tsr600-bar",acc["tsr600"],acc["uniform@600"])]:
        d,lo,hi=ci((a-b)[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>11}  "+"   ".join(out))
