"""Layer disagreement: the same image, the same question, a different answer about WHERE to look at every depth."""
import json, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from datasets import load_dataset
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25; Image.MAX_IMAGE_PIXELS=None
RED="#cf222e"; GREEN="#1a7f37"
MAPS={"qwen3":"phase30c_attn_maps_all.jsonl","qwen2":"phase74_Qwen2_VL_7B_Instruct.jsonl"}
BND={"qwen3":16,"qwen2":16}
def ringmask(gh,gw):
    rm=np.zeros((gh,gw),bool)
    if gh>2 and gw>2: rm[1:-1,1:-1]=True
    else: rm[:]=True
    return rm
def load(which):
    out={}
    for l in open(f"{D}/data/{MAPS[which]}"):
        r=json.loads(l)
        if "attn" in r and "grid" in r and "gt_box_frac" in r: out[r["question_id_full"]]=r
    return out
def cov(cx,cy,gt):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)

# ---------------- FIGURE A: qualitative, one image across depth ----------------
m3=load("qwen3")
ds=load_dataset("craigwu/vstar_bench")["test"]
root=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*'))[0]
lut={f"{r['category']}/{r['question_id']}":r for r in ds}
SHOW=[0,4,8,11,14,16,20,24,27]
PICKS=["direct_attributes/3","direct_attributes/18","relative_position/129"]
fig,ax=plt.subplots(len(PICKS),len(SHOW)+1,figsize=(2.05*(len(SHOW)+1),2.05*len(PICKS)))
for r,qid in enumerate(PICKS):
    q=m3[qid]; gh,gw=q["grid"]; rm=ringmask(gh,gw); gt=q["gt_box_frac"]
    im=Image.open(os.path.join(root,lut[qid]["image"])).convert("RGB"); Wp,Hp=im.size
    a=ax[r,0]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
    a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec="white",lw=3.2))
    a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec=RED,lw=1.9,ls=(0,(3,2))))
    qt=lut[qid]["text"].split("\n")[0].strip()
    a.set_ylabel((qt if len(qt)<=34 else qt[:32]+"..."),fontsize=8.2,labelpad=4)
    if r==0: a.set_title("image + GT",fontsize=9.5)
    A=np.stack([np.asarray(q["attn"][f"L{i}"],float) for i in range(len(q["attn"]))])
    A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
    for k,L in enumerate(SHOW):
        a=ax[r,k+1]; mp=A[L].reshape(gh,gw)
        hi=np.percentile(mp,98)   # clip the serialisation sink so the actual structure is visible
        a.imshow(np.clip(mp,0,hi),cmap="magma",vmin=0,vmax=max(hi,1e-9)); a.set_xticks([]); a.set_yticks([])
        j=int(np.argmax(np.where(rm.ravel(),A[L],-1e9))); jy,jx=j//gw,j%gw
        c=cov((jx+.5)/gw,(jy+.5)/gh,gt)
        a.plot(jx,jy,"x",color=GREEN if c>=.5 else RED,ms=9,mew=2.6)
        for sp in a.spines.values():
            sp.set_color(GREEN if c>=.5 else RED); sp.set_linewidth(2.0)
        if r==0: a.set_title(f"L{L}"+("  (boundary)" if L==16 else ""),fontsize=9.5,
                             color=GREEN if L==16 else "black")
plt.tight_layout(); plt.subplots_adjust(wspace=0.05,hspace=0.06)
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_layer_disagreement.{e}",dpi=165,bbox_inches="tight")
plt.close(); print("wrote fig_layer_disagreement")

# ---------------- FIGURE B: quantitative, agreement matrix + coverage curve ----------------
fig,ax=plt.subplots(1,3,figsize=(14.2,3.9))
for col,(which,lbl) in enumerate([("qwen3","Qwen3-VL-2B"),("qwen2","Qwen2-VL-7B")]):
    M=load(which); NL=28
    cells=np.zeros((len(M),NL),int); covs=np.zeros((len(M),NL))
    for i,(qid,q) in enumerate(sorted(M.items())):
        gh,gw=q["grid"]; rm=ringmask(gh,gw).ravel()
        A=np.stack([np.asarray(q["attn"][f"L{l}"],float) for l in range(NL)])
        A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        for l in range(NL):
            j=int(np.argmax(np.where(rm,A[l],-1e9))); cells[i,l]=j
            covs[i,l]=cov((j%gw+.5)/gw,(j//gw+.5)/gh,q["gt_box_frac"])
    Ag=np.zeros((NL,NL))
    for a_ in range(NL):
        for b_ in range(NL): Ag[a_,b_]=np.mean(cells[:,a_]==cells[:,b_])
    im=ax[col].imshow(Ag,cmap="viridis",vmin=0,vmax=1)
    ax[col].axhline(15.5,color="w",ls="--",lw=1.4); ax[col].axvline(15.5,color="w",ls="--",lw=1.4)
    ax[col].set_title(f"{lbl}: do two layers pick the same cell?",fontsize=10)
    ax[col].set_xlabel("layer"); ax[col].set_ylabel("layer")
    plt.colorbar(im,ax=ax[col],fraction=0.046)
    off=Ag[np.triu_indices(NL,1)]
    print(f"  {lbl}: mean off-diagonal arg-max agreement {off.mean()*100:.1f}%")
    ax[2].plot(range(NL),100*np.mean(covs>=.5,axis=0),"o-",ms=3.5,
               label=f"{lbl}", color="#0969da" if col==0 else "#d1710a")
ax[2].axvline(15.5,color=GREEN,ls="--",lw=2)
ax[2].text(16.3,8,"transport\nboundary",color=GREEN,fontsize=8.5)
ax[2].set_xlabel("read-out layer"); ax[2].set_ylabel("% of items whose crop\ncovers the evidence")
ax[2].set_title("single-layer read-out, by depth",fontsize=10); ax[2].legend(fontsize=8.5); ax[2].grid(alpha=.25)
plt.tight_layout()
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_layer_agreement.{e}",dpi=165,bbox_inches="tight")
plt.close(); print("wrote fig_layer_agreement")
