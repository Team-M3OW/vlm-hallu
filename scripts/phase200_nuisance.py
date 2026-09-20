"""
Phase 200 (CPU): does TWR's fitted depth weight anti-correlate with per-layer NUISANCE loading?

PRE-REGISTERED, definitions fixed before computing:
  nuisance at layer l := the item-MEAN map mbar_l, i.e. the component of the read-out that does not
      depend on which image it is. Chosen over a last-column indicator (assumes the answer) and over
      PC1 (may absorb genuine signal). §6A established this component is positionally stable.
  per-layer nuisance loading  L_l := mean_i cos( A_l^(i) , mbar_l )
  per-layer signal loading    S_l := mean_i cos( A_l^(i) , gt_l^(i) )   (gt = the item's GT box mask)
  PREDICTION P3: corr(w_l, L_l) < -0.3 and corr(w_l, S_l) > 0.
If P3 fails, the "signed weights cancel a common nuisance" account is WRONG and must not be written.
Maps are UNMASKED here: the deployed ring mask already removes most of the nuisance, so masking would
measure nothing (advisor's point).
"""
import json, sys, numpy as np
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"
MAPS={"qwen3":"phase30c_attn_maps_all.jsonl","qwen2":"phase74_Qwen2_VL_7B_Instruct.jsonl"}
for which in ("qwen3","qwen2"):
    rows=[json.loads(l) for l in open(f"{D}/data/{MAPS[which]}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    grids={tuple(r["grid"]) for r in rows}
    g0=max(grids,key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]      # item-mean needs a common grid
    gh,gw=g0; NL=28
    A=np.stack([np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)]) for r in sub])
    A=A/np.maximum(A.sum(2,keepdims=True),1e-12)       # (n, NL, cells)
    mbar=A.mean(0)                                     # (NL, cells) the item-independent component
    def cos(u,v): return float(u@v/max(np.linalg.norm(u)*np.linalg.norm(v),1e-12))
    L=np.array([np.mean([cos(A[i,l],mbar[l]) for i in range(len(sub))]) for l in range(NL)])
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    G=[]
    for r in sub:
        x0,y0,x1,y1=r["gt_box_frac"]
        G.append(((fx>=x0)&(fx<=x1)&(fy>=y0)&(fy<=y1)).astype(float))
    G=np.array(G)
    S=np.array([np.mean([cos(A[i,l],G[i]) for i in range(len(sub)) if G[i].sum()>0]) for l in range(NL)])
    w=np.load(f"{D}/data/fig_ridge_w_{which}.npy")[:NL]   # the log-A_l block of the fitted weights
    rL=np.corrcoef(w,L)[0,1]; rS=np.corrcoef(w,S)[0,1]
    print(f"=== {which}  (n={len(sub)} items on the modal {gh}x{gw} grid)")
    print(f"   corr(w_l, nuisance loading L_l) = {rL:+.3f}   [P3 predicts < -0.3]  {'PASS' if rL<-0.3 else 'FAIL'}")
    print(f"   corr(w_l, signal   loading S_l) = {rS:+.3f}   [P3 predicts > 0   ]  {'PASS' if rS>0 else 'FAIL'}")
    pre,post=slice(0,16),slice(16,28)
    print(f"   mean w  before boundary {w[pre].mean():+.4f}   after {w[post].mean():+.4f}")
    print(f"   mean L  before boundary {L[pre].mean():+.3f}   after {L[post].mean():+.3f}")
    print(f"   mean S  before boundary {S[pre].mean():+.3f}   after {S[post].mean():+.3f}")
    np.save(f"{D}/data/phase200_{which}_L.npy",L); np.save(f"{D}/data/phase200_{which}_S.npy",S)
    np.save(f"{D}/data/phase200_{which}_w.npy",w)
