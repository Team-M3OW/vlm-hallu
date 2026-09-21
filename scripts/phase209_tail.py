"""
Phase 209 (CPU): the block mean's failure is a TAIL/ARG-MAX problem, not an averaging-efficiency problem.

Why the previous two accounts are dead: signed cancellation (§49/§53, refuted) and matched filtering
(§208: 1 of 2 on both its key predictions, and the block mean already retains 77% of attainable SNR on
BOTH models while its coverage is 7x worse -- a mean-based quantity cannot explain an arg-max failure).

CANDIDATE: the decision is arg max over cells, which depends only on the upper TAIL of the score map.
A few cells carry large item-INDEPENDENT magnitude (§6A's raster sink). Equal-weight positive averaging
preserves their dominance, because they are large in every layer. The fix does not need supervision or
subtraction: it needs the score to be measured RELATIVE to what that cell scores on a typical item.

PRE-REGISTERED PREDICTIONS
  R1  on items the block mean gets wrong, its arg-max cell sits far higher in the ITEM-MEAN map than the
      target does. Threshold: median item-mean percentile of the chosen cell > 90 on both models.
  R2  an UNSUPERVISED lift read-out -- divide each cell by its item-mean value, then arg max -- recovers
      most of the supervised gap on BOTH models. Threshold: lift coverage >= 0.70 x DWA coverage.
  R3  lift needs no ring mask: its last-column rate is near the 5% chance rate on both models.
If R2 fails, this account is rejected too and nothing gets written.
"""
import json, numpy as np
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25
MAPS={"qwen3":("phase30c_attn_maps_all.jsonl",(16,27),0.578),"qwen2":("phase74_Qwen2_VL_7B_Instruct.jsonl",(15,27),0.588)}
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
for which,(fn,BLK,dwa) in MAPS.items():
    rows=[json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0=max({tuple(r["grid"]) for r in rows},key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]; gh,gw=g0; NL=28; n=len(sub)
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    A=np.stack([np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)]) for r in sub])
    A=A/np.maximum(A.sum(2,keepdims=True),1e-12)
    dep=A[:,BLK[0]:BLK[1]].mean(1)                     # (n, cells) the deployed block-mean map
    mbar=dep.mean(0)                                   # item-independent component of that map
    pct=100*(np.argsort(np.argsort(mbar))/(len(mbar)-1))
    gts=[r["gt_box_frac"] for r in sub]
    def covs(sc):
        out=[]
        for i in range(n):
            j=int(np.argmax(sc[i])); out.append(cov((j%gw+.5)/gw,(j//gw+.5)/gh,gts[i]))
        return np.array(out)
    cb=covs(dep)
    lift=dep/np.maximum(mbar[None],1e-12)              # UNSUPERVISED: relative to a typical item
    cl=covs(lift)
    bad=[i for i in range(n) if cb[i]<0.5]
    mp=np.median([pct[int(np.argmax(dep[i]))] for i in bad])
    def lastcol(sc):
        return 100*np.mean([ (int(np.argmax(sc[i]))%gw)==gw-1 for i in range(n)])
    print(f"=== {which}  n={n}")
    print(f"   R1 median item-mean percentile of the block mean's chosen cell, on its failures: {mp:.1f}"
          f"   (>90?) {'PASS' if mp>90 else 'FAIL'}   [{len(bad)} failures]")
    print(f"   R2 coverage  block mean {cb.mean():.3f}   ->  LIFT (unsupervised) {cl.mean():.3f}"
          f"   DWA (supervised) {dwa:.3f}   lift/DWA {cl.mean()/dwa:.2f}   (>=0.70?) {'PASS' if cl.mean()/dwa>=0.70 else 'FAIL'}")
    print(f"   R3 last-column rate  block mean {lastcol(dep):5.1f}%   lift {lastcol(lift):5.1f}%   (chance {100/gw:.1f}%)")
