import json,sys,numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows]); rng=np.random.default_rng(189)
arms=list(rows[0]["probs"]); A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows]); acc={k:A(k) for k in arms}
TL={k:np.mean([r["token_layers"][k] for r in rows]) for k in arms}
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
print(f"\n{f.split('/')[-1]} n={n}  token-layers % of bar: "+"  ".join(f"{k} {TL[k]/TL['uniform@600']*100:.0f}%" for k in arms))
S=[("single",cat=="direct_attributes"),("cross",cat=="relative_position"),("ALL",np.ones(n,bool))]
print(f"  {'stratum':>9} "+" ".join(f"{k:>14}" for k in arms))
for t,m in S: print(f"  {t:>9} "+" ".join(f"{acc[k][m].mean()*100:13.1f}%" for k in arms))
for t,m in S:
    out=[]
    pairs=[("S0 obj600-bar","objprune600","uniform@600"),("P1 obj1500-bar","objprune1500","uniform@600"),
           ("S1 obj-rand","objprune1500","random1500"),("obj-attn","objprune1500","attnprune1500")]
    for nm,a,b in pairs:
        if a in acc and b in acc:
            d,lo,hi=ci((acc[a]-acc[b])[m]); out.append(f"{nm} {d:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{'✔' if lo>0 else ('✗' if hi<0 else ' ')}")
    print(f"  {t:>9}  "+"  ".join(out))
