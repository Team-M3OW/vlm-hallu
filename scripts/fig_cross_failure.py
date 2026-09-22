"""Why the crop-based read-out fails on cross-instance questions: the cell it needs is in a valley.

Attention is MODE-seeking (peaks sit on the objects). The single window that covers both objects must be
centred near their midpoint, which is background. So the statistic the method reads and the quantity it
needs are different functions of the same configuration, and they coincide only when there is one object.
Boxes come from V*Bench's per-image JSONs (a LIST of [x,y,w,h]); the HF loader does not expose them.
"""
import json, os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25; Image.MAX_IMAGE_PIXELS=None
RED="#cf222e"; GREEN="#1a7f37"; BLUE="#0969da"; ORANGE="#d1710a"; GREY="#57606a"
SNAP=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*"))[0]
q2i={}; q2t={}
for l in open(f"{SNAP}/test_questions.jsonl"):
    e=json.loads(l); k=f"{e['category']}/{e['question_id']}"; q2i[k]=e["image"]; q2t[k]=e["text"]
def boxes_of(qid):
    img=q2i.get(qid)
    if not img: return None,None
    jf=os.path.join(SNAP,img.replace(".jpg",".json"))
    if not os.path.exists(jf): return None,None
    d=json.load(open(jf)); bb=d["bbox"]
    if isinstance(bb,str): bb=json.loads(bb.replace("'",'"'))
    if not isinstance(bb[0],list): bb=[bb]
    im=Image.open(os.path.join(SNAP,img)); Wp,Hp=im.size
    return im,[[x/Wp,y/Hp,(x+w)/Wp,(y+h)/Hp] for x,y,w,h in bb]
SC={json.loads(l)['qid']:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_qwen3.jsonl")}
rel=[q for q in SC if q.startswith("relative_position")]
# ---------- quantitative: where does each cell sit in the map's own ranking? ----------
pk_dwa=[]; pk_union=[]; pk_obj=[]
recs=[]
for q in rel:
    m=SC[q]; im,bxs=boxes_of(q)
    if im is None or len(bxs)<2: continue
    gh,gw=m["grid"]; dep=np.asarray(m["dep"],float).ravel()
    pct=100*np.argsort(np.argsort(dep))/(len(dep)-1)
    def cell(cx,cy): return min(int(cy*gh),gh-1)*gw+min(int(cx*gw),gw-1)
    ub=[min(b[0] for b in bxs),min(b[1] for b in bxs),max(b[2] for b in bxs),max(b[3] for b in bxs)]
    uc=cell((ub[0]+ub[2])/2,(ub[1]+ub[3])/2)
    dc=cell(*m["ridge_cell"])
    oc=[cell((b[0]+b[2])/2,(b[1]+b[3])/2) for b in bxs]
    pk_dwa.append(pct[dc]); pk_union.append(pct[uc]); pk_obj+= [pct[c] for c in oc]
    recs.append((q,im,bxs,ub,m,pct,uc,dc))
print(f"relational items with 2+ boxes: {len(recs)}")
print(f"  median attention percentile -- objects {np.median(pk_obj):.0f} | DWA's pick {np.median(pk_dwa):.0f}"
      f" | the union-covering cell {np.median(pk_union):.0f}")
# ---------- figure ----------
PICKS=[r for r in recs if r[5][r[6]]<60][:3]          # items where the union cell really is in a valley
fig=plt.figure(figsize=(13.8,3.15*len(PICKS)+1.0))
gs=fig.add_gridspec(len(PICKS),4,width_ratios=[1.5,1.15,1.15,1.25],hspace=0.30,wspace=0.16)
for r,(q,im,bxs,ub,m,pct,uc,dc) in enumerate(PICKS):
    gh,gw=m["grid"]; Wp,Hp=im.size
    dep=np.asarray(m["dep"],float).reshape(gh,gw)
    a=fig.add_subplot(gs[r,0]); a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for b in bxs:
        a.add_patch(Rectangle((b[0]*Wp,b[1]*Hp),(b[2]-b[0])*Wp,(b[3]-b[1])*Hp,fill=False,ec=RED,lw=2.2))
    a.add_patch(Rectangle((ub[0]*Wp,ub[1]*Hp),(ub[2]-ub[0])*Wp,(ub[3]-ub[1])*Hp,fill=False,ec="white",lw=2.4,ls=(0,(4,2))))
    def win(cx,cy):
        x0=min(max(0,cx-W/2),1-W); y0=min(max(0,cy-W/2),1-W); return x0,y0
    dx,dy=win(*m["ridge_cell"]); a.add_patch(Rectangle((dx*Wp,dy*Hp),W*Wp,W*Hp,fill=False,ec=GREEN,lw=2.4))
    ux,uy=win((ub[0]+ub[2])/2,(ub[1]+ub[3])/2); a.add_patch(Rectangle((ux*Wp,uy*Hp),W*Wp,W*Hp,fill=False,ec=BLUE,lw=2.4))
    a.set_ylabel(q2t[q].split("\n")[0][:44],fontsize=8.2,labelpad=4)
    if r==0: a.set_title("image: two objects, and two candidate crops",fontsize=9.6)
    a=fig.add_subplot(gs[r,1]); a.imshow(dep,cmap="magma"); a.set_xticks([]); a.set_yticks([])
    a.plot(dc%gw,dc//gw,"x",color=GREEN,ms=10,mew=2.8); a.plot(uc%gw,uc//gw,"+",color=BLUE,ms=12,mew=2.8)
    if r==0: a.set_title("attention: peaks on the objects",fontsize=9.6)
    # 1-D profile along the line joining the two object centres
    a=fig.add_subplot(gs[r,2])
    c0=((bxs[0][0]+bxs[0][2])/2,(bxs[0][1]+bxs[0][3])/2); c1=((bxs[1][0]+bxs[1][2])/2,(bxs[1][1]+bxs[1][3])/2)
    ts=np.linspace(0,1,60)
    prof=[dep[min(int((c0[1]+t*(c1[1]-c0[1]))*gh),gh-1),min(int((c0[0]+t*(c1[0]-c0[0]))*gw),gw-1)] for t in ts]
    a.plot(ts,prof,color=GREY,lw=1.8); a.axvline(0.5,color=BLUE,ls="--",lw=1.8)
    a.scatter([0,1],[prof[0],prof[-1]],color=RED,zorder=3,s=28)
    a.set_xticks([0,0.5,1]); a.set_xticklabels(["object A","midpoint","object B"],fontsize=7.8)
    a.set_yticks([]); a.grid(alpha=.2)
    if r==0: a.set_title("attention along the line between them",fontsize=9.6)
    a=fig.add_subplot(gs[r,3]); a.axis("off")
    a.text(0.0,0.72,f"DWA's pick sits at the {pct[dc]:.0f}th percentile",fontsize=9,color=GREEN,transform=a.transAxes)
    a.text(0.0,0.52,f"the cell covering both sits at the {pct[uc]:.0f}th",fontsize=9,color=BLUE,transform=a.transAxes)
    a.text(0.0,0.26,"the arg-max of a two-peaked map\nis a peak, never the valley between",
           fontsize=8.6,color="#24292f",style="italic",transform=a.transAxes)
fig.legend(handles=[Line2D([],[],color=RED,lw=2,label='ground-truth objects'),
                    Line2D([],[],color='k',lw=2,ls='--',label='union bounding box (the training label)'),
                    Line2D([],[],color=GREEN,lw=2,label="DWA's crop"),
                    Line2D([],[],color=BLUE,lw=2,label='crop centred to cover both')],
           loc='lower center',ncol=4,fontsize=9,frameon=False)
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_cross_failure.{e}",dpi=165,bbox_inches="tight")
plt.close()
# ---------- the population view ----------
plt.figure(figsize=(6.0,3.6))
bins=np.linspace(0,100,26)
plt.hist(pk_obj,bins=bins,alpha=.65,color=RED,label="cells on an object")
plt.hist(pk_dwa,bins=bins,alpha=.65,color=GREEN,label="DWA's chosen cell")
plt.hist(pk_union,bins=bins,alpha=.65,color=BLUE,label="cell that covers both")
plt.xlabel("percentile of that cell in the attention map"); plt.ylabel("relational items")
plt.title("What the read-out ranks highly, and what it needs",fontsize=10.5)
plt.legend(fontsize=8.4); plt.grid(alpha=.25); plt.tight_layout()
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_cross_failure_hist.{e}",dpi=165,bbox_inches="tight")
print("wrote fig_cross_failure and fig_cross_failure_hist")
