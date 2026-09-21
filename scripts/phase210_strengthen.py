"""
Phase 210 (CPU): strengthen §61 from an explanation into a PREDICTION.

The tail-dominance proposition says the arg-max is decided by
    margin(c) = [m(c) - m(c*)]  +  [u(c) - u(c*)],
m item-independent, u item-specific. Three consequences, each now stated as a testable number.

S1 (RATE).  Under approximate independence across depth, the deterministic lead over a band B grows ~|B|
    (the m_l leads are sign-aligned) while sd(sum_B u_l) grows ~sqrt(|B|). So Delta_B / sigma_B should grow like
    |B|^0.5. PRE-REGISTERED: log-log slope in [0.3, 0.7] on BOTH models.
S2 (DERIVED ESTIMATOR). Dividing by m zeroes the deterministic lead exactly. But the item-specific part is
    heteroscedastic across cells, so the theory says the RIGHT unsupervised statistic is the studentised one,
    z(c) = (x(c) - m(c)) / s(c), with m, s the per-cell mean and sd ACROSS ITEMS. PRE-REGISTERED: z >= lift on
    BOTH models. This is a prediction: nothing measured so far says studentising should help.
S3 (HOW MUCH). PRE-REGISTERED: z closes >= 85% of the block-mean -> DWA gap on BOTH models.
A failure of S2 or S3 means the theory explains but does not predict, and must be reported as such.
"""
import json, numpy as np
W=0.25
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
for which,fn,C,DWA in (("qwen3","phase30c_attn_maps_all.jsonl",16,0.578),
                       ("qwen2","phase74_Qwen2_VL_7B_Instruct.jsonl",15,0.588)):
    rows=[json.loads(l) for l in open(f"data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0=max({tuple(r["grid"]) for r in rows},key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]; gh,gw=g0; NL=28; n=len(sub)
    A=np.stack([np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)]) for r in sub])
    A=A/np.maximum(A.sum(2,keepdims=True),1e-12)
    gts=[r["gt_box_frac"] for r in sub]
    def covs(sc):
        return np.array([cov((int(np.argmax(sc[i]))%gw+.5)/gw,(int(np.argmax(sc[i]))//gw+.5)/gh,gts[i]) for i in range(n)])
    # ---- S1: does Delta_B / sigma_B grow like sqrt(|B|)?
    widths=[1,2,3,4,5,6,7,8,9,10,11]; ratios=[]
    for wd in widths:
        dep=A[:,C:min(C+wd,NL)].sum(1)
        m=dep.mean(0); u=dep-m[None]
        Delta=m.max()-np.median(m); sigma=u.std()
        ratios.append(Delta/max(sigma,1e-12))
    sl=np.polyfit(np.log(widths),np.log(ratios),1)[0]
    # ---- S2/S3: lift vs studentised
    dep=A[:,C:C+11].mean(1)
    m=dep.mean(0); s=dep.std(0)
    cb=covs(dep); cl=covs(dep/np.maximum(m[None],1e-12)); cz=covs((dep-m[None])/np.maximum(s[None],1e-12))
    gap=DWA-cb.mean()
    print(f"=== {which}  n={n}")
    print(f"   S1 log-log slope of Delta_B/sigma_B vs band width = {sl:+.2f}   (in [0.3,0.7]?) "
          f"{'PASS' if 0.3<=sl<=0.7 else 'FAIL'}    ratios {ratios[0]:.1f} -> {ratios[-1]:.1f}")
    print(f"   coverage: block mean {cb.mean():.3f} | lift {cl.mean():.3f} | studentised z {cz.mean():.3f} | DWA {DWA:.3f}")
    print(f"   S2 z >= lift ? {'PASS' if cz.mean()>=cl.mean() else 'FAIL'}   (z-lift {cz.mean()-cl.mean():+.3f})")
    print(f"   S3 z closes {100*(cz.mean()-cb.mean())/gap:.0f}% of the block-mean -> DWA gap  (>=85%?) "
          f"{'PASS' if (cz.mean()-cb.mean())/gap>=0.85 else 'FAIL'}")
    print(f"      lift closes {100*(cl.mean()-cb.mean())/gap:.0f}%")
