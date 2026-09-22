"""
Phase 218 (CPU): can DWA find the cell that covers BOTH objects if it is told what the crop contains?

WHY IT CURRENTLY CANNOT. DWA scores each cell from that cell's OWN features. "Be between two peaks" is a
property of the configuration, not of a cell, so a per-cell linear score cannot express it. Its only
spatial feature is a 3x3 neighbourhood mean, which on a 15x20 grid covers 20%x15% of the image while the
crop actually taken covers 25%x25% -- the feature is SMALLER than the window it is meant to score.
Measured consequence (§33): the union-covering cell sits at the 61st percentile of attention (39% of cells
outscore it) while the ridge's picks sit at rank ~6%.

THE FIX TESTED HERE: add a WINDOW-INTEGRATED feature -- the attention mass inside the W x W box centred on
each cell, per layer -- so the score answers "how much evidence would the crop I take actually contain".
Pre-registered: union coverage on cross-instance rises on BOTH models; single-instance does not fall.
Label for the fit is UNION coverage (all boxes), as deployed.
"""
import json, numpy as np, sys
from sklearn.model_selection import GroupKFold
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25
# TRUE multi-box annotations: V*Bench ships one JSON per image with a bbox LIST [x,y,w,h] in pixels.
# The HF dataset loader does not expose them; they live beside the images in the snapshot.
import glob, os
from PIL import Image as _I
_I.MAX_IMAGE_PIXELS=None
_SNAP=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*"))[0]
def load_boxes():
    out={}
    for cat in ("direct_attributes","relative_position"):
        for jf in glob.glob(f"{_SNAP}/{cat}/*.json"):
            d=json.load(open(jf))
            bb=d.get("bbox")
            if isinstance(bb,str): bb=json.loads(bb.replace("'",'"'))
            if not bb: continue
            if not isinstance(bb[0],list): bb=[bb]
            W_,H_=_I.open(jf.replace(".json",".jpg")).size
            out[os.path.basename(jf)[:-5]]=[[x/W_,y/H_,(x+w)/W_,(y+h)/H_] for x,y,w,h in bb]
    return out
BOXES=load_boxes()
QID2IMG={}
for _l in open(f"{_SNAP}/test_questions.jsonl"):
    _e=json.loads(_l); QID2IMG[f"{_e['category']}/{_e['question_id']}"]=os.path.basename(_e["image"])[:-4]
print(f"  loaded multi-box annotations for {len(BOXES)} images; "
      f"mean boxes {sum(len(v) for v in BOXES.values())/len(BOXES):.2f}")
def unioncov(cx,cy,boxes):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    got=0
    for gx0,gy0,gx1,gy1 in boxes:
        inter=max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))
        if inter/max((gx1-gx0)*(gy1-gy0),1e-12)>=0.5: got+=1
    return got/max(len(boxes),1)
def oof(X,Y,G,seeds=(700,701,702)):
    P=np.zeros(len(Y)); gs=np.unique(G)
    for s in seeds:
        rng=np.random.default_rng(s); perm={g:i for i,g in enumerate(rng.permutation(gs))}
        Gp=np.vectorize(perm.get)(G)
        for tr,te in GroupKFold(5).split(X,Y,Gp):
            mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9; Xt=np.c_[(X[tr]-mu)/sd,np.ones(len(tr))]
            A_=Xt.T@Xt+np.eye(Xt.shape[1]); A_[-1,-1]-=1.0
            w=np.linalg.solve(A_,Xt.T@Y[tr]); P[te]+=np.c_[(X[te]-mu)/sd,np.ones(len(te))]@w
    return P/len(seeds)
MAPS={"qwen3":("phase30c_attn_maps_all.jsonl",(16,27)),"qwen2":("phase74_Qwen2_VL_7B_Instruct.jsonl",(15,27))}
for which,(fn,BLK) in MAPS.items():
    rows=[json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows=[r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    g0=max({tuple(r["grid"]) for r in rows},key=lambda g:sum(1 for r in rows if tuple(r["grid"])==g))
    sub=[r for r in rows if tuple(r["grid"])==g0]; gh,gw=g0; NL=28; n=len(sub); ncell=gh*gw
    yy,xx=np.mgrid[0:gh,0:gw]; fy,fx=((yy+.5)/gh).ravel(),((xx+.5)/gw).ravel()
    kh,kw=max(1,int(round(W*gh))),max(1,int(round(W*gw)))     # the ACTUAL crop window, in cells
    rm=np.zeros((gh,gw),bool)
    if gh>2 and gw>2: rm[1:-1,1:-1]=True
    else: rm[:]=True
    rm=rm.ravel()
    Xb=[];Xw=[];Y=[];G=[];cats=[]
    for i,r in enumerate(sub):
        a=np.stack([np.asarray(r["attn"][f"L{l}"],float) for l in range(NL)])
        a=a/np.maximum(a.sum(1,keepdims=True),1e-12)
        LA=np.log(a+1e-12).T
        R=(np.argsort(np.argsort(-a,axis=1),axis=1)/max(ncell-1,1)).T
        dep=a[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); pad=np.pad(dep,1,mode="edge")
        nb=(sum(pad[p:p+gh,q:q+gw] for p in range(3) for q in range(3))/9.0).ravel()
        geo=np.c_[np.log(nb+1e-12),fx,fy,np.sqrt((fx-.5)**2+(fy-.5)**2),
                  np.minimum(np.minimum(fx,1-fx),np.minimum(fy,1-fy)),
                  (xx.ravel()==gw-1).astype(float),(yy.ravel()==gh-1).astype(float)]
        Xb.append(np.c_[LA,R,geo])
        # NEW: per-layer attention mass inside the W x W window centred on each cell
        wins=[]
        for l in range(NL):
            m=a[l].reshape(gh,gw)
            P2=np.pad(m,((kh,kh),(kw,kw)),mode="constant")
            C=P2.cumsum(0).cumsum(1)
            def box(r0,c0,r1,c1): return C[r1,c1]-C[r0,c1]-C[r1,c0]+C[r0,c0]
            out=np.zeros((gh,gw))
            for rr in range(gh):
                for cc in range(gw):
                    r0=rr+kh-kh//2; c0=cc+kw-kw//2
                    out[rr,cc]=box(r0,c0,r0+kh,c0+kw)
            wins.append(np.log(out.ravel()+1e-12))
        Xw.append(np.stack(wins,1))
        img_key=QID2IMG.get(r["question_id_full"])
        boxes=BOXES.get(img_key) or [r["gt_box_frac"]]
        Y.append(np.array([unioncov(float(fx[j]),float(fy[j]),boxes) for j in range(ncell)]))
        G.append(np.full(ncell,i)); cats.append(r["category"])
    Xb=np.vstack(Xb); Xw=np.vstack(Xw); Y=np.concatenate(Y); G=np.concatenate(G)
    covg=Y.reshape(n,ncell); cats=np.array(cats)
    def score(X):
        P=oof(X,Y,G).reshape(n,ncell)
        pick=[int(np.argmax(np.where(rm,P[i],-1e9))) for i in range(n)]
        c=np.array([covg[i,pick[i]] for i in range(n)])
        return c,np.array(pick)
    base,pb=score(Xb); wide,pw=score(np.c_[Xb,Xw])
    ch=(pb!=pw); dcov=wide-base
    print(f"   picks changed: {int(ch.sum())}/{n};  per-item coverage changed on {int((dcov!=0).sum())}"
          f"  (up {int((dcov>0).sum())}, down {int((dcov<0).sum())});  coverage values seen: "
          f"{sorted(set(np.round(base,3)))[:5]}")
    s=cats=="direct_attributes"; x=~s
    print(f"=== {which}  n={n}  window={kh}x{kw} cells (the actual crop)")
    print(f"   baseline  union-cov  ALL {base.mean():.5f}  single {base[s].mean():.5f}  cross {base[x].mean():.5f}")
    print(f"   +window   union-cov  ALL {wide.mean():.5f}  single {wide[s].mean():.5f}  cross {wide[x].mean():.5f}")
    print(f"   delta                ALL {wide.mean()-base.mean():+.5f}  single {wide[s].mean()-base[s].mean():+.5f}"
          f"  cross {wide[x].mean()-base[x].mean():+.5f}")
