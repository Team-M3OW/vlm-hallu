import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows]); rng=np.random.default_rng(196)
arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
TL={k:np.mean([r["token_layers"][k] for r in rows]) for k in arms}
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]} n={n}  token-layers % of bar: "+"  ".join(f"{k} {TL[k]/TL['uniform@600']*100:.0f}%" for k in arms))
S=[("single",cat=="direct_attributes"),("cross",cat=="relative_position"),("ALL",np.ones(n,bool))]
print(f"  {'stratum':>9} "+" ".join(f"{k:>12}" for k in arms))
for t,m in S: print(f"  {t:>9} "+" ".join(f"{acc[k][m].mean()*100:11.1f}%" for k in arms))
for t,m in S:
    out=[]
    for nm,a,b in [("P1 merge-bar",acc["merge"],acc["uniform@600"]),("P2 merge-tsr",acc["merge"],acc["tsr900"]),
                   ("merge-ridge1",acc["merge"],acc["ridge1@300"]),("S1 merge-attn",acc["merge"],acc["merge_attn"]),("S2 merge-L16",acc["merge"],acc["merge_L16"])]:
        d,lo,hi=ci((a-b)[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>9}  "+"  ".join(out))
