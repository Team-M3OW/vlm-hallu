"""Analysis for phase 198: does TSR stack on top of existing placement rules?"""
import json, sys, random, numpy as np
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; random.seed(198); B=8000; W=0.25
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
def boot(d):
    if not d: return None  # empty subset must never print as a zero effect
    m=sum(d)/len(d); s=sorted(sum(random.choice(d) for _ in range(len(d)))/len(d) for _ in range(B))
    return m*100,s[int(.025*B)]*100,s[int(.975*B)]*100
def mark(lo,hi): return " OK" if lo>0 else (" NEG" if hi<0 else "")
def fmt(res):
    v,n=res
    if v is None or n==0: return f"{'n/a (n=0)':>22s}"
    return f"{v[0]:+6.1f} [{v[1]:+5.1f},{v[2]:+5.1f}]{mark(v[1],v[2]):4s}" 
RULES=["vicrop_block","laser","ridge","oracle"]
for which in sys.argv[1:]:
    f=f"{D}/data/phase198_stack_{which}.jsonl"
    rows=[json.loads(l) for l in open(f)]
    print(f"\n{'='*104}\n=== {which}   n={len(rows)}")
    tl={k:int(np.mean([r['token_layers'][k] for r in rows if k in r['token_layers']])) for k in rows[0]['token_layers']}
    bar=600*28
    print("budgets (% of the 16,800 token-layer bar): " +
          "  ".join(f"{k}={100*v/bar:.0f}%" for k,v in sorted(tl.items()) if k.endswith(('plain','tsr','460')) or k=='uniform@600'))
    def acc(k,sub=None):
        rs=[r for r in rows if k in r['probs'] and (sub is None or r['category']==sub)]
        return 100*np.mean([np.argmax(r['probs'][k])==r['label'] for r in rs]) if rs else float('nan')
    def paired(a,b,sub=None):
        rs=[r for r in rows if a in r['probs'] and b in r['probs'] and (sub is None or r['category']==sub)]
        return boot([int(np.argmax(r["probs"][a])==r["label"])-int(np.argmax(r["probs"][b])==r["label"]) for r in rs]), len(rs)
    print(f"\n  bar uniform@600 = {acc('uniform@600'):.1f}")
    print(f"\n  {'rule':14s} {'cover':>6s} {'plain':>7s} {'TSR':>7s} {'460':>7s} | {'P1: TSR-plain':>24s} {'PRED: 460-plain':>22s} {'GUARD: TSR-460':>22s}")
    out={}
    for r in RULES:
        if f"{r}_plain" not in rows[0]['probs']: continue
        cv=100*np.mean([cov(*x['cells'][r],x['gt'])>=.5 for x in rows if r in x['cells']])
        p1=paired(f"{r}_tsr",f"{r}_plain"); pr=paired(f"{r}_460",f"{r}_plain"); gd=paired(f"{r}_tsr",f"{r}_460")
        out[r]=(cv,p1[0],pr[0],gd[0])
        print(f"  {r:14s} {cv:5.1f}% {acc(f'{r}_plain'):7.1f} {acc(f'{r}_tsr'):7.1f} {acc(f'{r}_460'):7.1f} | "
              f"{fmt(p1)} {fmt(pr)} {fmt(gd)}")
    print(f"\n  per stratum (TSR - plain):  {'rule':14s} {'single':>22s} {'cross':>22s}")
    for r in RULES:
        if f"{r}_plain" not in rows[0]['probs']: continue
        s=paired(f"{r}_tsr",f"{r}_plain","direct_attributes"); c=paired(f"{r}_tsr",f"{r}_plain","relative_position")
        print(f"  {'':28s}{r:14s} {fmt(s)} {fmt(c)}")
    if len(out)>2:
        cvs=[v[0] for v in out.values() if v[1]]; gns=[v[1][0] for v in out.values() if v[1]]
        print(f"\n  S1 gain vs placement quality: r = {np.corrcoef(cvs,gns)[0,1]:+.3f}  (coverage {['%.0f'%c for c in cvs]} -> gain {['%+.1f'%g for g in gns]})")
    print(f"\n  vs the bar:")
    for r in RULES:
        if f"{r}_plain" not in rows[0]['probs']: continue
        a=paired(f"{r}_plain","uniform@600"); b=paired(f"{r}_tsr","uniform@600")
        print(f"    {r:14s} plain {fmt(a)}   +TSR {fmt(b)}")
