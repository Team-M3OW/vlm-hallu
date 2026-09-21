"""AVR gallery: 12 real inferences showing what survives the boundary under attention-keep vs random-keep."""
import json, os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; Image.MAX_IMAGE_PIXELS=None
GREEN="#1a7f37"; RED="#cf222e"; PURPLE="#8250df"; BLUE="#0969da"
rows=json.load(open(f"{D}/data/fig_tsr_gallery.json"))
root=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*'))[0]
def opts(t): return dict(re.findall(r"\(([A-D])\)\s*([^\n]+)",t))
def short(t,n=38):
    s=t.split("\n")[0].strip(); return s if len(s)<=n else s[:n-2]+"..."
N=len(rows); cols=4; rws=(N+cols-1)//cols
fig,ax=plt.subplots(rws*2,cols*3,figsize=(3.4*cols*1.35,2.15*rws*1.5))
for i,m in enumerate(rows):
    r,c=divmod(i,cols)
    im=Image.open(os.path.join(root,m['image'])).convert("RGB")
    gh,gw=m['grid']; O=opts(m['text']); lab=m['label']
    panels=[("image",None,None),("keep_attn",GREEN,"attention keep"),("keep_rand",PURPLE,"random keep")]
    for k,(key,col,lbl) in enumerate(panels):
        a=ax[r*2,c*3+k]; a.set_xticks([]); a.set_yticks([])
        if key=="image":
            a.imshow(im)
            a.set_title(short(m["text"],32),fontsize=7.4,pad=4)
        else:
            msk=np.zeros(gh*gw); msk[[j for j in m[key] if j<gh*gw]]=1.0; msk=msk.reshape(gh,gw)
            a.imshow(im.resize((gw*10,gh*10),Image.BICUBIC),extent=(0,gw,gh,0),alpha=0.38)
            a.imshow(np.ma.masked_where(msk==0,msk),extent=(0,gw,gh,0),
                     cmap=matplotlib.colors.ListedColormap([col]),vmin=0,vmax=1,alpha=0.95,interpolation="nearest")
            a.set_xlim(0,gw); a.set_ylim(gh,0)
            a.set_title(lbl,fontsize=7.4,color=col,pad=4)
    a=ax[r*2+1,c*3]; a.axis("off")
    txt=[]
    for key,col,note in [("uniform@600",BLUE,"bar"),("tsr900",GREEN,"AVR"),
                         ("tsr900_rand",PURPLE,"rand"),("uniform@900","#57606a","no prune")]:
        j=int(np.argmax(m['probs'][key])); ok=j==lab
        txt.append((note,f"({'ABCD'[j]}) {O.get('ABCD'[j],'?')[:13]}",'✓' if ok else '✗',col,GREEN if ok else RED))
    for t,(note,ans,tick,c1,c2) in enumerate(txt):
        a.text(0.02,0.86-t*0.235,note,fontsize=7.4,color=c1,weight="bold",transform=a.transAxes)
        a.text(0.34,0.86-t*0.235,f"{ans} {tick}",fontsize=7.8,color=c2,transform=a.transAxes)
    ax[r*2+1,c*3+1].axis("off"); ax[r*2+1,c*3+2].axis("off")
    ax[r*2+1,c*3+1].text(0.0,0.72,f"{len(m['keep_attn'])} of {m['n_tokens']} tokens\nsurvive L16",fontsize=7.4,
                         transform=ax[r*2+1,c*3+1].transAxes,color="#24292f")
    ax[r*2+1,c*3+1].text(0.0,0.30,f"keep-set overlap {100*m['overlap']:.0f}%\n(chance 10%)",fontsize=7.4,
                         transform=ax[r*2+1,c*3+1].transAxes,style="italic",color="#57606a")
for j in range(N,rws*cols):
    r,c=divmod(j,cols)
    for k in range(3): ax[r*2,c*3+k].axis("off"); ax[r*2+1,c*3+k].axis("off")
plt.tight_layout(); plt.subplots_adjust(wspace=0.10,hspace=0.30)
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_gallery_tsr.{e}",dpi=155,bbox_inches="tight")
print(f"wrote fig_gallery_tsr ({N} inferences)")
