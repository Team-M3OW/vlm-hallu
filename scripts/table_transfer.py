"""DWA cross-benchmark transfer results, with CLUSTER bootstrap by question group.

HR-Bench is 200 unique questions x 4 cyclic option permutations. The 4 items in a cycle share an
image and a question, so they are NOT independent: a per-item bootstrap would report CIs that are
roughly 2x too narrow. We resample GROUPS (index//4) with replacement instead.

PRE-REGISTERED (phase224 docstring, fixed before any HR-Bench number was seen):
  adopt transfer iff (dwa_t - block) excludes zero on >=2 of the 4 Qwen cells.
  Beating `bar` alone is NOT sufficient -- that shows cropping helps, not that DWA's placement does.
"""
import json, glob, os, numpy as np
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(0)
def corr(r,a): return 1.0 if int(np.argmax(r["probs"][a]))==r["label"] else 0.0
def clusterci(rows,a,b,B=10000):
    g={}
    for r in rows: g.setdefault(r.get("group",r["qid"]),[]).append(corr(r,a)-corr(r,b))
    keys=list(g); per=np.array([np.mean(g[k]) for k in keys]); n=len(keys)
    if n<2: return per.mean()*100,float("nan"),float("nan"),n
    m=per[rng.integers(0,n,(B,n))].mean(1)
    return per.mean()*100,np.percentile(m,2.5)*100,np.percentile(m,97.5)*100,n
def acc(rows,a): return 100*np.mean([corr(r,a) for r in rows])
def show(rows,tag):
    out=[]
    for a,b in [("dwa_t","bar"),("block","bar"),("dwa_t","block")]:
        m,lo,hi,n=clusterci(rows,a,b)
        star="  *" if (lo==lo and (lo>0 or hi<0)) else ""
        out.append(f"{a}-{b}: {m:+5.1f}[{lo:+5.1f},{hi:+5.1f}]{star}")
    print(f"  {tag:22s} n={len(rows):4d} grp={clusterci(rows,'dwa_t','bar')[3]:3d} "
          f"bar {acc(rows,'bar'):5.1f}  block {acc(rows,'block'):5.1f}  dwa_t {acc(rows,'dwa_t'):5.1f}  | "
          +"   ".join(out))
    return out
files=sorted(glob.glob(f"{D}/data/phase224_dwat_*.jsonl"))
if not files: raise SystemExit("no phase224 results yet")
decisive=0; total=0; extra=[]
for f in files:
    rows=[json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    tag=os.path.basename(f)[:-6].replace("phase224_dwat_","")
    print(f"=== {tag}")
    o=show(rows,"ALL")
    m,lo,hi,_=clusterci(rows,"dwa_t","block")
    # The rule was pre-registered over the FOUR QWEN cells. Families added later are out-of-sample
    # evidence and must not grow the denominator -- that would silently weaken a fixed threshold.
    if tag.startswith("qwen"):
        total+=1; decisive+= 1 if (lo==lo and (lo>0 or hi<0)) else 0
    else:
        extra.append((tag, f"{m:+5.1f}[{lo:+5.1f},{hi:+5.1f}]", "sig" if (lo==lo and (lo>0 or hi<0)) else "n.s."))
    for cat in sorted({r["category"] for r in rows}):
        sub=[r for r in rows if r["category"]==cat]
        if len(sub)>=8: show(sub,f"  [{cat}]")
print(f"\nPRE-REGISTERED RULE (4 Qwen cells only): (dwa_t - block) excludes zero on {decisive} of {total}; "
      f"adopt iff >=2.  -> {'ADOPT' if decisive>=2 else 'REJECT'}")
if extra:
    print("OUT-OF-SAMPLE families (not part of the registered rule):")
    for t,v,k in extra: print(f"   {t:20s} dwa_t-block {v}  {k}")
