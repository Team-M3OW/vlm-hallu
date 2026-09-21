"""AVR inference panel: what survives the boundary, and why the choice of survivor does not matter."""
import json, os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
GREEN="#1a7f37"; RED="#cf222e"; BLUE="#0969da"; PURPLE="#8250df"
rows=json.load(open(f"{D}/data/fig_tsr_dump.json"))
root=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*'))[0]
def opts(t): return dict(re.findall(r"\(([A-D])\)\s*([^\n]+)",t))
ARMS=[("uniform@600",BLUE,"600 tok x 28 layers"),("tsr900",GREEN,"900 tok to 90 at L16"),
      ("tsr900_rand",PURPLE,"random keep"),("uniform@900",'#57606a',"900 tok, no prune")]
fig,ax=plt.subplots(len(rows),4,figsize=(14.2,3.0*len(rows)),
                    gridspec_kw={'width_ratios':[1.25,1.25,1.25,1.0]})
for r,m in enumerate(rows):
    im=Image.open(os.path.join(root,m['image'])).convert("RGB")
    gh,gw=m['grid']; n=m['n_tokens']; O=opts(m['text']); letters="ABCD"
    a=ax[r,0]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
    q=m['text'].split(chr(10))[0].strip()
    a.set_ylabel(q if len(q)<=46 else q[:44]+'...',fontsize=9,labelpad=6)
    if r==0: a.set_title(f"encoded at 900 visual tokens",fontsize=10)
    for k,(key,col,lbl) in enumerate([("keep_attn",GREEN,"kept by attention (top 10%)"),
                                      ("keep_rand",PURPLE,"kept at random (10%)")]):
        a=ax[r,1+k]
        msk=np.zeros(gh*gw); msk[[i for i in m[key] if i<gh*gw]]=1.0; msk=msk.reshape(gh,gw)
        a.imshow(im.resize((gw*12,gh*12),Image.BICUBIC),extent=(0,gw,gh,0),alpha=0.40)
        a.imshow(np.ma.masked_where(msk==0,msk),extent=(0,gw,gh,0),cmap=
                 matplotlib.colors.ListedColormap([col]),vmin=0,vmax=1,alpha=0.95,interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(0,gw); a.set_ylim(gh,0)
        if r==0: a.set_title(lbl,fontsize=10)
        if r==len(rows)-1: a.set_xlabel(f"89 of {n} tokens survive L16",fontsize=9)
    a=ax[r,3]; a.axis("off")
    y=0.90
    for key,col,note in ARMS:
        p=m['probs'][key]; j=int(np.argmax(p)); ok=j==m['label']
        a.text(0.0,y,f"{key.replace('_',' ')}",fontsize=9.6,color=col,weight="bold",transform=a.transAxes)
        a.text(0.0,y-0.085,f"({letters[j]}) {O.get(letters[j],'?')[:22]}  {'✓' if ok else '✗'}",
               fontsize=10.2,color=GREEN if ok else RED,transform=a.transAxes)
        a.text(0.0,y-0.155,note,fontsize=8.2,color="#57606a",transform=a.transAxes)
        y-=0.245
    a.text(0.0,y+0.03,f"survivor overlap {100*m['overlap']:.0f}%  (chance 10%)",
           fontsize=9,color="#24292f",transform=a.transAxes,style="italic")
    if r==0: a.set_title("answer",fontsize=10,loc="left")
plt.tight_layout(); plt.subplots_adjust(wspace=0.07,hspace=0.16)
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_infer_tsr.{e}",dpi=170,bbox_inches="tight")
print("wrote fig_infer_tsr")
