"""phase215b: DWA end task per family, WITH the cost of each arm.
`bar` is the incumbent's single pass; `bar_matched` is a single pass at the DWA TOTAL
(localise + crop), which is the honest equal-compute comparison. Bootstrap over items."""
import json,glob,os,numpy as np
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(0)
def corr(r,a): return 1.0 if int(np.argmax(r["probs"][a]))==r["label"] else 0.0
def ci(rows,a,b,B=10000):
    d=np.array([corr(r,a)-corr(r,b) for r in rows]); n=len(d)
    m=d[rng.integers(0,n,(B,n))].mean(1)
    lo,hi=np.percentile(m,2.5)*100,np.percentile(m,97.5)*100
    return f"{d.mean()*100:+5.1f}[{lo:+5.1f},{hi:+5.1f}]"+("  *" if (lo>0 or hi<0) else "   ")
for f in sorted(glob.glob(f"{D}/data/phase215b_dwaeval_*.jsonl")):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    tag=os.path.basename(f)[:-6].replace("phase215b_dwaeval_","")
    arms=[a for a in ("bar","bar_matched","block","dwa") if a in rows[0]["probs"]]
    print(f"=== {tag}  n={len(rows)}")
    for a in arms: print(f"    {a:12s} {100*np.mean([corr(r,a) for r in rows]):5.1f}")
    if "tokens" in rows[0]:
        tk={k:int(np.mean([r['tokens'][k] for r in rows])) for k in rows[0]['tokens']}
        print(f"    mean visual tokens: {tk}")
        print(f"    DWA cost vs bar: {tk.get('dwa_total',0)/max(tk.get('bar',1),1):.2f}x")
    for a,b in [("dwa","bar"),("dwa","block"),("dwa","bar_matched"),("block","bar")]:
        if a in arms and b in arms: print(f"    {a:5s} - {b:12s} {ci(rows,a,b)}")
    for cat in sorted({r["category"] for r in rows}):
        sub=[r for r in rows if r["category"]==cat]
        if len(sub)>=15 and "bar_matched" in arms:
            print(f"      [{cat:18s} n={len(sub):3d}]  dwa-bar {ci(sub,'dwa','bar')}  dwa-block {ci(sub,'dwa','block')}  dwa-barM {ci(sub,'dwa','bar_matched')}")
