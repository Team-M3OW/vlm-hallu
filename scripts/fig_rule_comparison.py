"""All five placement rules on the same images: what each one crops and what the model then answers."""
import json, os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from datasets import load_dataset
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25; Image.MAX_IMAGE_PIXELS=None
GREEN="#1a7f37"; RED="#cf222e"
RULES=[("vicrop_L14","fixed layer L14"),("vicrop_block","block-mean L16-26"),
       ("laser","LASER (per-item layer)"),("ridge","DWA (all layers, signed)"),("oracle","oracle placement")]
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
def crop(im,cx,cy):
    Wp,Hp=im.size; x0,y0,x1,y1=win(cx,cy)
    return im.crop((int(x0*Wp),int(y0*Hp),int(x1*Wp),int(y1*Hp)))
which="qwen3"
P179=json.load(open(f"{D}/data/phase179_placements_{which}.json"))
SC={json.loads(l)['qid']:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_{which}.jsonl")}
ARM={json.loads(l)['question_id_full']:json.loads(l) for l in open(f"{D}/data/phase184_allarms_{which}.jsonl")}
ids=sorted(set(P179)&set(SC)&set(ARM))
# Selection rule, stated in the caption: single-instance items on which the oracle crop is CORRECT
# (so the ceiling behaves like a ceiling) and DWA is correct while the fixed-layer rule is not.
# Ordered by question id; the first four are shown.
def ok(q,k): return int(np.argmax(ARM[q]['probs'][k])==ARM[q]['label'])
cand=[q for q in ids if q.startswith("direct_attributes")
      and ok(q,'oracle') and ok(q,'ridge') and not ok(q,'vicrop_L14')]
cand.sort(key=lambda q:int(q.split('/')[1]))
print(f"{len(cand)} items satisfy the selection rule")
PICKS=cand[:4]
fig,ax=plt.subplots(len(PICKS),len(RULES)+1,figsize=(2.32*(len(RULES)+1),2.05*len(PICKS)))
for r,q in enumerate(PICKS):
    im=Image.open(os.path.join(root,lut[q]['image'])).convert("RGB"); Wp,Hp=im.size
    gt=SC[q]['gt']; O=opts(lut[q]['text']); lab=ARM[q]['label']
    a=ax[r,0]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
    a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec="white",lw=3.0))
    a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec=RED,lw=1.8,ls=(0,(3,2))))
    a.set_ylabel(short(lut[q]['text'],34),fontsize=8.4,labelpad=4)
    if r==0: a.set_title("image + ground truth",fontsize=9.2)
    for c,(k,lbl) in enumerate(RULES):
        cell = SC[q]['ridge_cell'] if k=="ridge" else (
               [(gt[0]+gt[2])/2,(gt[1]+gt[3])/2] if k=="oracle" else P179[q]['cells'][k])
        a=ax[r,c+1]; a.imshow(crop(im,*cell)); a.set_xticks([]); a.set_yticks([])
        j=int(np.argmax(ARM[q]['probs'][k])); ok=j==lab
        a.set_xlabel(f"({'ABCD'[j]}) {O.get('ABCD'[j],'?')[:15]} {'✓' if ok else '✗'}",
                     fontsize=8.6,color=GREEN if ok else RED,labelpad=2)
        for sp in a.spines.values(): sp.set_color(GREEN if ok else RED); sp.set_linewidth(2.2)
        if r==0: a.set_title(lbl,fontsize=9.2)
plt.tight_layout(); plt.subplots_adjust(wspace=0.06,hspace=0.32)
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_rule_comparison.{e}",dpi=160,bbox_inches="tight")
print(f"wrote fig_rule_comparison ({len(PICKS)} items x {len(RULES)} rules = {len(PICKS)*len(RULES)} inferences)")
