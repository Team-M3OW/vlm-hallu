"""Recompute the OOF ridge score map per V*Bench item, exactly as phase184_allarms.placements().
Verifies against the cells logged in phase193 before writing. CPU only; no model is loaded."""
import json, numpy as np, sys
from sklearn.model_selection import GroupKFold
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"
WHICH=sys.argv[1] if len(sys.argv)>1 else "qwen3"
NL,BLK=(28,(16,27)) if WHICH=="qwen3" else (28,(15,27))
W=0.25; COV_HIT=0.5
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    inter=max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))
    return inter/max((gx1-gx0)*(gy1-gy0),1e-12)
MAPS={"qwen3":"phase30c_attn_maps_all.jsonl","qwen2":"phase74_Qwen2_VL_7B_Instruct.jsonl"}[WHICH]
rows=[json.loads(l) for l in open(f"{D}/data/{MAPS}")]
rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
Xr=[];Yr=[];Gr=[];meta=[]
for gi,q in enumerate(rows):
    gh,gw=q["grid"]; n=q["n_img_tokens"]
    NLq=len(q["attn"]); A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(NLq)]); A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
    assert NLq==NL, f"map has {NLq} layers, expected {NL}"
    LA=np.log(A+1e-12).T; R=(np.argsort(np.argsort(-A,axis=1),axis=1)/max(n-1,1)).T
    dep=A[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
    nb=(sum(pad[i:i+gh,j:j+gw] for i in range(3) for j in range(3))/9.0).ravel()
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
              np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
              (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
    Xr.append(np.c_[LA,R,geo]); Yr.append(np.array([cov(float(fx[i]),float(fy[i]),q["gt_box_frac"]) for i in range(n)]))
    Gr.append(np.full(n,gi)); meta.append(dict(qid=q["question_id_full"],grid=[gh,gw],n=n,gt=q["gt_box_frac"],
                                               category=q["category"],dep=dep.tolist()))
Xr=np.vstack(Xr); Yr=np.concatenate(Yr); Gr=np.concatenate(Gr)
Pr=np.zeros(len(Yr)); gs=np.unique(Gr); Wsum=np.zeros(Xr.shape[1]+1); nfit=0
for s in range(3):
    rng=np.random.default_rng(700+s); perm={g:i for i,g in enumerate(rng.permutation(gs))}
    Gp=np.vectorize(perm.get)(Gr)
    for tr,te in GroupKFold(5).split(Xr,Yr,Gp):
        mu,sd=Xr[tr].mean(0),Xr[tr].std(0)+1e-9; Xt=np.c_[(Xr[tr]-mu)/sd,np.ones(len(tr))]
        A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
        w=np.linalg.solve(A_,Xt.T@Yr[tr]); Pr[te]+=np.c_[(Xr[te]-mu)/sd,np.ones(len(te))]@w
        Wsum+=w; nfit+=1
Pr/=3
out=[]; off=0
for m in meta:
    n=m["n"]; gh,gw=m["grid"]
    sc=Pr[off:off+n].reshape(gh,gw); off+=n
    # the deployed pipeline masks the outer ring for EVERY placement rule (phase179/phase184)
    rm=np.zeros((gh,gw),bool)
    if gh>2 and gw>2: rm[1:-1,1:-1]=True
    else: rm[:]=True
    iy,ix=np.unravel_index(np.argmax(np.where(rm,sc,-1e9)),sc.shape)
    m["score"]=sc.tolist(); m["ring_mask"]=rm.tolist()
    m["ridge_cell"]=[float((ix+.5)/gw),float((iy+.5)/gh)]
    d=np.asarray(m["dep"]); jy,jx=np.unravel_index(np.argmax(np.where(rm,d,-1e9)),d.shape)
    m["block_cell"]=[float((jx+.5)/gw),float((jy+.5)/gh)]
    jy2,jx2=np.unravel_index(np.argmax(d),d.shape)
    m["block_cell_raw"]=[float((jx2+.5)/gw),float((jy2+.5)/gh)]
    out.append(m)
assert off==len(Pr)
np.save(f"{D}/data/fig_ridge_w_{WHICH}.npy", Wsum/nfit)
with open(f"{D}/data/fig_ridge_scores_{WHICH}.jsonl","w") as f:
    for m in out: f.write(json.dumps(m)+"\n")
# verification against the logged run
mc={json.loads(l)["question_id_full"]:json.loads(l) for l in open(f"{D}/data/phase193_multicrop_{WHICH}.jsonl")}
agree=sum(1 for m in out if m["qid"] in mc and
          np.allclose(m["ridge_cell"], mc[m["qid"]]["cells"][0], atol=1e-6))
print(f"ridge cell reproduces phase193 on {agree}/{len(mc)} items")
print(f"mean ridge coverage {np.mean([cov(*m['ridge_cell'],m['gt']) for m in out]):.3f}  "
      f"block {np.mean([cov(*m['block_cell'],m['gt']) for m in out]):.3f}")
