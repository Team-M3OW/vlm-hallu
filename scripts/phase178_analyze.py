"""Phase 178 analysis. PRIMARY = glocal_150_150. P1: glocal - bar on RELATIONAL, CI clear on both models.
GUARD: glocal - head@0.25 on SINGLE not significantly negative. Context arms are printed but not eligible."""
import json, sys, numpy as np
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); cat=np.array([r["category"] for r in rows])
A=lambda k: np.array([float(int(np.argmax(r["probs"][k]))==r["label"]) for r in rows])
T=lambda k: np.array([r["realized_tokens"][k] for r in rows],float)
rng=np.random.default_rng(178)
def ci(d):
    m=len(d); b=np.array([d[rng.integers(0,m,m)].mean() for _ in range(8000)]); return d.mean()*100,np.percentile(b,2.5)*100,np.percentile(b,97.5)*100
arms=list(rows[0]["probs"].keys()); acc={k:A(k) for k in arms}
tok={k:T(k).mean()+(300 if k not in ("uniform@300","uniform@600") else 0) for k in arms}   # +300 for the localiser pass
print(f"\n{f.split('/')[-1]}  n={n}")
print("  budget (incl. localiser pass): "+"  ".join(f"{k} {tok[k]:.0f}" for k in arms))
drift=[k for k in arms if k not in ("uniform@300",) and abs(tok[k]-tok["uniform@600"])/tok["uniform@600"]>0.10]
if drift: print(f"  ** BUDGET DRIFT >10% in {drift} -- those contrasts are VOID **")
bar=acc["uniform@600"]; head=acc["head@0.25"]; g=acc["glocal_150_150"]
print(f"  {'stratum':>12} {'bar':>6} {'head':>6} {'glocal':>7} {'orcGL':>6} {'orc.25':>6} |  P1 glocal-bar            GUARD glocal-head")
for tag,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
    p1,l1,h1=ci((g-bar)[m]); g2,l2,h2=ci((g-head)[m])
    print(f"  {tag:>12} {bar[m].mean()*100:5.1f}% {head[m].mean()*100:5.1f}% {g[m].mean()*100:6.1f}% {acc['oracle_glocal_150_150'][m].mean()*100:5.1f}% {acc['oracle@0.25'][m].mean()*100:5.1f}% | "
          f"{p1:+5.1f} [{l1:+5.1f},{h1:+5.1f}]{'✔' if l1>0 else (' ✗' if h1<0 else '  ')}   {g2:+5.1f} [{l2:+5.1f},{h2:+5.1f}]{'✔' if l2>0 else (' ✗' if h2<0 else '  ')}")
print("  context arms (NOT eligible as the method):")
for k in ("glocal_100_200","glocal_200_100","uniform@300"):
    for tag,m in [("single",cat=="direct_attributes"),("relational",cat=="relative_position"),("ALL",np.ones(n,bool))]:
        a,lo,hi=ci((acc[k]-bar)[m]); print(f"    {k:>16} {tag:>11} {acc[k][m].mean()*100:5.1f}%   vs bar {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]")
