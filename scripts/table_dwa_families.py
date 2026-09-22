"""The DWA cross-family table, at EQUAL COMPUTE, from every source in one place.

Compute matching differs by architecture and the table says which applies:
  Qwen   (phase223): continuous resolution axis -> localise@300 + crop@300 vs bar@600. Matched by
                     construction.
  others (phase215b): anyres/fixed tilers cannot hit half the bar, so the crop runs at its natural
                     cost and `bar_matched` is a SINGLE pass at the DWA total. That is the honest
                     comparison; `dwa - bar` is reported too because it is what the literature
                     quotes, and the gap between the two columns is the point.
"""
import json,os,glob,numpy as np
D=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng=np.random.default_rng(0)
def corr(r,a): return 1.0 if int(np.argmax(r["probs"][a]))==r["label"] else 0.0
def ci(rows,a,b,B=10000):
    d=np.array([corr(r,a)-corr(r,b) for r in rows]); n=len(d)
    m=d[rng.integers(0,n,(B,n))].mean(1); lo,hi=np.percentile(m,2.5)*100,np.percentile(m,97.5)*100
    return d.mean()*100,lo,hi
def fmt(t):
    m,lo,hi=t; return f"{m:+5.1f}[{lo:+5.1f},{hi:+5.1f}]"+("*" if (lo>0 or hi<0) else " ")
print(f"{'model':16s} {'n':>4s} {'bar':>6s} {'block':>6s} {'DWA':>6s} {'barM':>6s} {'cost':>5s} | "
      f"{'DWA-block':>20s} {'DWA-bar(lit)':>20s} {'DWA-barMatched':>20s}")
NAMES={"qwen3":"Qwen3-VL-2B","qwen2":"Qwen2-VL-7B","gemma3_4b":"Gemma-3-4B",
       "llava_ov":"LLaVA-OV-7B","internvl3_8b":"InternVL3-8B"}
# Qwen: matched by construction in phase223 (no block arm there; block comes from phase184)
blk={}
for w in ("qwen3","qwen2"):
    f=f"{D}/data/phase184_allarms_{w}.jsonl"
    if os.path.exists(f):
        r=[json.loads(l) for l in open(f) if l.strip()]
        r=[x for x in r if "vicrop_block" in x["probs"]]
        blk[w]=100*np.mean([corr(x,"vicrop_block") for x in r])
for w in ("qwen3","qwen2"):
    f=f"{D}/data/phase223_logonly_{w}.jsonl"
    if not os.path.exists(f): continue
    r=[json.loads(l) for l in open(f) if l.strip()]
    bar=100*np.mean([corr(x,"bar") for x in r]); dwa=100*np.mean([corr(x,"dwa63") for x in r])
    print(f"{NAMES[w]:16s} {len(r):4d} {bar:6.1f} {blk.get(w,float('nan')):6.1f} {dwa:6.1f} {'--':>6s} {'1.00x':>5s} | "
          f"{'(see phase184)':>20s} {fmt(ci(r,'dwa63','bar')):>20s} {'= DWA-bar':>20s}")
for f in sorted(glob.glob(f"{D}/data/phase215b_dwaeval_*.jsonl")):
    r=[json.loads(l) for l in open(f) if l.strip()]
    if not r: continue
    tag=os.path.basename(f)[:-6].replace("phase215b_dwaeval_","")
    tk={k:np.mean([x["tokens"][k] for x in r]) for k in r[0]["tokens"]}
    cost=tk["dwa_total"]/max(tk["bar"],1)
    print(f"{NAMES.get(tag,tag):16s} {len(r):4d} "
          f"{100*np.mean([corr(x,'bar') for x in r]):6.1f} "
          f"{100*np.mean([corr(x,'block') for x in r]):6.1f} "
          f"{100*np.mean([corr(x,'dwa') for x in r]):6.1f} "
          f"{100*np.mean([corr(x,'bar_matched') for x in r]):6.1f} {cost:4.2f}x | "
          f"{fmt(ci(r,'dwa','block')):>20s} {fmt(ci(r,'dwa','bar')):>20s} {fmt(ci(r,'dwa','bar_matched')):>20s}")
print("\n* = 95% CI excludes zero.  'DWA-bar(lit)' is the comparison the literature makes;")
print("'DWA-barMatched' gives the bar the SAME total budget. Where they disagree, the gain was compute.")
