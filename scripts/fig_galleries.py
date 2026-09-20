"""Poster/paper galleries: many real inferences at a glance, plus the sink and the coverage distribution."""
import json, os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from datasets import load_dataset
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25; Image.MAX_IMAGE_PIXELS=None
GREEN="#1a7f37"; RED="#cf222e"; ORANGE="#d1710a"; BLUE="#0969da"
ds=load_dataset("craigwu/vstar_bench")["test"]
root=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*'))[0]
lut={f"{r['category']}/{r['question_id']}":r for r in ds}
def opts(t): return dict(re.findall(r"\(([A-D])\)\s*([^\n]+)",t))
def short(t,n=40):
    s=t.split("\n")[0].strip(); return s if len(s)<=n else s[:n-2]+"..."
def win(cx,cy):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    return x0,y0,x1,y1
def cov(cx,cy,gt):
    x0,y0,x1,y1=win(cx,cy); gx0,gy0,gx1,gy1=gt
    return max(0.,min(gx1,x1)-max(gx0,x0))*max(0.,min(gy1,y1)-max(gy0,y0))/max((gx1-gx0)*(gy1-gy0),1e-12)
def crop(im,cx,cy):
    Wp,Hp=im.size; x0,y0,x1,y1=win(cx,cy)
    return im.crop((int(x0*Wp),int(y0*Hp),int(x1*Wp),int(y1*Hp)))
SC={w:{json.loads(l)['qid']:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_{w}.jsonl")} for w in ("qwen3","qwen2")}
ARM={w:{json.loads(l)['question_id_full']:json.loads(l) for l in open(f"{D}/data/phase184_allarms_{w}.jsonl")} for w in ("qwen3","qwen2")}

# ---------- 1. TWR gallery: 8 inferences, block-mean vs TWR ----------
def twr_gallery(which,fname,n_show=8):
    sc,arm=SC[which],ARM[which]
    ids=sorted(set(sc)&set(arm))
    sel=[q for q in ids if cov(*sc[q]['ridge_cell'],sc[q]['gt'])>=.5
         and cov(*sc[q]['block_cell'],sc[q]['gt'])<.5
         and np.argmax(arm[q]['probs']['ridge'])==arm[q]['label']
         and np.argmax(arm[q]['probs']['vicrop_block'])!=arm[q]['label']]
    sel.sort(key=lambda q:(q.split('/')[0],int(q.split('/')[1])))
    sel=sel[:n_show]
    rows=(len(sel)+3)//4
    fig,ax=plt.subplots(rows*3,4,figsize=(13.6,2.55*rows*1.55),
                        gridspec_kw={'height_ratios':[1.35,1,0.10]*rows})
    for i,q in enumerate(sel):
        r,c=divmod(i,4); im=Image.open(os.path.join(root,lut[q]['image'])).convert("RGB")
        Wp,Hp=im.size; m=sc[q]; gt=m['gt']; O=opts(lut[q]['text']); lab=arm[q]['label']
        a=ax[r*3,c]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for cell,col in ((m['block_cell'],ORANGE),(m['ridge_cell'],GREEN)):
            x0,y0,x1,y1=win(*cell)
            a.add_patch(Rectangle((x0*Wp,y0*Hp),(x1-x0)*Wp,(y1-y0)*Hp,fill=False,ec=col,lw=2.2))
        a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec="white",lw=2.8))
        a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec=RED,lw=1.6,ls=(0,(3,2))))
        a.set_title(short(lut[q]['text']),fontsize=8.2,pad=3)
        sub=ax[r*3+1,c]; sub.axis("off")
        box=sub.get_position()
        for k,(cell,col,armk) in enumerate(((m['block_cell'],ORANGE,'vicrop_block'),(m['ridge_cell'],GREEN,'ridge'))):
            in_ax=fig.add_axes([box.x0+k*box.width/2,box.y0,box.width/2*0.93,box.height])
            in_ax.imshow(crop(im,*cell)); in_ax.set_xticks([]); in_ax.set_yticks([])
            for sp in in_ax.spines.values(): sp.set_color(col); sp.set_linewidth(2.2)
            j=int(np.argmax(arm[q]['probs'][armk])); ok=j==lab
            in_ax.set_xlabel(f"({'ABCD'[j]}) {O.get('ABCD'[j],'?')[:14]} {'✓' if ok else '✗'}",
                             fontsize=8.4,color=GREEN if ok else RED,labelpad=2)
        ax[r*3+2,c].axis("off")
    for j in range(len(sel),rows*4):
        r,c=divmod(j,4)
        for k in range(3): ax[r*3+k,c].axis("off")
    fig.legend(handles=[Line2D([],[],color=RED,ls='--',lw=2,label='ground truth'),
                        Line2D([],[],color=ORANGE,lw=2,label='block-mean crop'),
                        Line2D([],[],color=GREEN,lw=2,label='TWR crop')],
               loc='lower center',ncol=3,fontsize=9.5,frameon=False)
    for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/{fname}.{e}",dpi=155,bbox_inches="tight")
    plt.close(); print(f"wrote {fname}  ({len(sel)} inferences)")
twr_gallery("qwen3","fig_gallery_twr_q3")
twr_gallery("qwen2","fig_gallery_twr_q2")

# ---------- 2. the sink, averaged over the whole benchmark ----------
from numpy import interp
def resample(m,H=15,V=20):
    gh,gw=m.shape
    r=np.stack([interp(np.linspace(0,gw-1,V),np.arange(gw),m[i]) for i in range(gh)])
    return np.stack([interp(np.linspace(0,gh-1,H),np.arange(gh),r[:,j]) for j in range(V)],axis=1)
fig,ax=plt.subplots(1,3,figsize=(13.4,3.5))
for i,(w,lbl,BLK) in enumerate([("qwen3","Qwen3-VL-2B",(16,27)),("qwen2","Qwen2-VL-7B",(15,27))]):
    MAPS={"qwen3":"phase30c_attn_maps_all.jsonl","qwen2":"phase74_Qwen2_VL_7B_Instruct.jsonl"}[w]
    acc=np.zeros((15,20)); colbin=np.zeros(20); k=0
    for l in open(f"{D}/data/{MAPS}"):
        r=json.loads(l)
        if "attn" not in r or "grid" not in r: continue
        gh,gw=r["grid"]
        A=np.stack([np.asarray(r["attn"][f"L{j}"],float) for j in range(28)])
        A=A/np.maximum(A.sum(1,keepdims=True),1e-12)
        m=A[BLK[0]:BLK[1]].mean(0).reshape(gh,gw); m=m/max(m.sum(),1e-12)
        acc+=resample(m); k+=1
        cm=m.sum(0)
        colbin+=interp(np.linspace(0,gw-1,20),np.arange(gw),cm)/max(cm.sum(),1e-12)
    acc/=k; colbin/=k
    im=ax[i].imshow(acc,cmap="magma"); ax[i].set_xticks([]); ax[i].set_yticks([])
    ax[i].set_title(f"{lbl}: mean read-out map, all {k} items",fontsize=9.5)
    plt.colorbar(im,ax=ax[i],fraction=0.035)
    ax[2].plot(np.linspace(0,1,20),100*colbin/colbin.sum(),"o-",ms=3.5,label=lbl,color=BLUE if i==0 else ORANGE)
ax[2].axhline(5.0,color="grey",ls=":",lw=1.4); ax[2].text(0.02,5.5,"uniform share",fontsize=8,color="grey")
ax[2].set_xlabel("relative column position (0 = left, 1 = right edge)"); ax[2].set_ylabel("% of attention mass")
ax[2].set_title("mass by column: the raster row-wrap",fontsize=9.5); ax[2].legend(fontsize=8.5); ax[2].grid(alpha=.25)
plt.tight_layout()
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_sink_map.{e}",dpi=165,bbox_inches="tight")
plt.close(); print("wrote fig_sink_map")

# ---------- 3. coverage distribution ----------
plt.figure(figsize=(5.6,4.2))
for w,lbl,ls in [("qwen3","Qwen3-VL-2B","-"),("qwen2","Qwen2-VL-7B","--")]:
    sc=SC[w]; ids=sorted(sc)
    for key,col,nm in [("ridge_cell",GREEN,"TWR"),("block_cell",ORANGE,"block-mean"),("block_cell_raw","#8250df","block-mean, no ring mask")]:
        c=np.sort([cov(*sc[q][key],sc[q]['gt']) for q in ids])
        plt.plot(c,100*np.arange(len(c))/len(c),ls,color=col,lw=1.9,
                 label=f"{nm} ({lbl.split('-')[0]}{lbl[-3:]})" if ls=="-" else None)
        if ls=="--": plt.plot(c,100*np.arange(len(c))/len(c),ls,color=col,lw=1.5,alpha=.75)
plt.axvline(0.5,color="k",ls=":",lw=1.3); plt.text(0.52,4,"'covers the evidence'",fontsize=8)
plt.xlabel("fraction of the ground-truth box inside the crop"); plt.ylabel("% of items below")
plt.title("Crop quality, whole benchmark\n(solid: Qwen3-VL-2B, dashed: Qwen2-VL-7B)",fontsize=10)
plt.legend(fontsize=8.2,loc="upper left"); plt.grid(alpha=.25); plt.tight_layout()
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_coverage_cdf.{e}",dpi=165,bbox_inches="tight")
plt.close(); print("wrote fig_coverage_cdf")
