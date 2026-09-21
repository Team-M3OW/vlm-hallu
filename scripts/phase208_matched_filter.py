"""
Phase 208 (CPU): is DWA's fitted depth weighting a MATCHED FILTER over depth?

CANDIDATE THEORY (pre-registered here, before any number is computed):
  Model each layer's per-cell feature as  x_l(c) = s_l * t(c) + n_l(c),  t = indicator of the target cell,
  s_l >= 0 the layer's signal gain, n_l zero-mean noise with per-layer variance sigma_l^2, approximately
  uncorrelated across layers (§53 measured mean corr = -0.020 / -0.017, so this premise is already supported).
  A linear read-out sum_l w_l x_l has SNR (sum_l w_l s_l)^2 / sum_l w_l^2 sigma_l^2, maximised at
      w_l*  proportional to  s_l / sigma_l^2          (the matched filter / inverse-variance weighting).
  The block mean is w_l = 1 on a fixed band and 0 elsewhere.

PRE-REGISTERED PREDICTIONS
  P-A  the per-layer SNR s_l/sigma_l^2 is NOT uniform over the block-mean band. Threshold: coefficient of
       variation of s_l/sigma_l^2 across the band > 0.5 on BOTH models. (If it were uniform, the block mean
       would already be optimal and there would be nothing to explain.)
  P-B  the fitted log-attention weights track the matched filter: corr(w_l, s_l/sigma_l^2) > +0.3 on BOTH models.
  P-C  s_l >= 0 for most layers, which is why NON-NEGATIVE fits lose nothing (§53). Threshold: >= 80% of layers
       have s_l > 0 on both models.
If P-B fails on either model the matched-filter account is REJECTED and must not be written.
Features are the ones the model actually sees: log-attention per cell, UNMASKED maps, modal grid.
"""
import json, numpy as np
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"
MAPS={"qwen3":("phase30c_attn_maps_all.jsonl",(16,27)),"qwen2":("phase74_Qwen2_VL_7B_Instruct.jsonl",(15,27))}
for which,(fn,BLK) in MAPS.items():
    rows=[json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0=max({tuple(r["grid"]) for r in rows},key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]; gh,gw=g0; NL=28; n=len(sub)
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    S=np.zeros(NL); V=np.zeros(NL); k=0
    for r in sub:
        x0,y0,x1,y1=r["gt_box_frac"]
        t=((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(float)
        if t.sum()==0:
            # V*Bench boxes are often smaller than one cell, so no cell CENTRE lies inside.
            # Fall back to the single cell nearest the box centre; never drop the item.
            cx,cy=(x0+x1)/2,(y0+y1)/2
            t=np.zeros_like(fx); t[int(np.argmin((fx-cx)**2+(fy-cy)**2))]=1.0
        a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)])
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        X=np.log(a+1e-12)                       # the feature the model sees
        X=X-X.mean(1,keepdims=True)             # centre per layer
        S+=(X*t).sum(1)/t.sum()                 # mean feature on target cells = signal gain
        V+=X[:,t==0].var(1)                     # noise variance off target
        k+=1
    S/=k; V/=k
    snr=S/np.maximum(V,1e-12)
    w=np.load(f"{D}/data/fig_ridge_w_{which}.npy")[:NL]
    band=slice(BLK[0],BLK[1])
    cv=float(np.std(snr[band])/max(abs(np.mean(snr[band])),1e-12))
    rB=float(np.corrcoef(w,snr)[0,1]); pos=100*float((S>0).mean())
    # Cauchy-Schwarz: SNR of uniform-on-band vs optimal-over-all-layers
    wu=np.zeros(NL); wu[band]=1.0
    def snr_of(wv): return (wv@S)**2/max((wv**2*V).sum(),1e-12)
    eff=snr_of(wu)/max(snr_of(snr),1e-12)
    print(f"=== {which}  n={k} items, modal {gh}x{gw} grid")
    print(f"   P-A  CV of per-layer SNR across the block-mean band = {cv:.2f}   (>0.5?) {'PASS' if cv>0.5 else 'FAIL'}")
    print(f"   P-B  corr(fitted w_l, matched filter s_l/sigma_l^2)  = {rB:+.3f}   (>+0.3?) {'PASS' if rB>0.3 else 'FAIL'}")
    print(f"   P-C  layers with positive signal gain s_l            = {pos:.0f}%     (>=80%?) {'PASS' if pos>=80 else 'FAIL'}")
    print(f"   Cauchy-Schwarz efficiency of the block mean vs the matched filter: {100*eff:.1f}% of attainable SNR")
    top=np.argsort(-snr)[:6]
    print(f"   top-SNR layers: {sorted(int(i) for i in top)}   (§53's non-negative selection: Q3 [4,5,17,19], Q2 [19,21])")
