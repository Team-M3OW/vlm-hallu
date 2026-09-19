"""Ridge inference panel: what the two read-out rules see and answer, on real V*Bench items.
Example selection rule (stated in the caption): single-instance items on which the ridge crop covers
>=50% of the ground-truth box and the block-mean crop does not, AND the ridge answer is correct while
the block-mean answer is not, restricted to items where the recomputed OOF ridge cell reproduces the
logged run exactly. 9 items qualify; the first three by question id are shown."""
import json, os, glob, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from datasets import load_dataset
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"; W=0.25
Image.MAX_IMAGE_PIXELS=None
GREEN="#1a7f37"; ORANGE="#d1710a"; RED="#cf222e"
def win(cx,cy):
    x0,x1,y0,y1=cx-W/2,cx+W/2,cy-W/2,cy+W/2
    if x0<0: x0,x1=0.,W
    if y0<0: y0,y1=0.,W
    if x1>1: x0,x1=1-W,1.
    if y1>1: y0,y1=1-W,1.
    return x0,y0,x1,y1
sc={json.loads(l)['qid']:json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_qwen3.jsonl")}
arm={json.loads(l)['question_id_full']:json.loads(l) for l in open(f"{D}/data/phase184_allarms_qwen3.jsonl")}
ds=load_dataset("craigwu/vstar_bench")['test']
root=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--craigwu--vstar_bench/snapshots/*'))[0]
lut={f"{r['category']}/{r['question_id']}":r for r in ds}
PICKS=["direct_attributes/3","direct_attributes/14","direct_attributes/18"]
def opts(t):
    o=dict(re.findall(r"\(([A-D])\)\s*([^\n]+)",t)); return o
def short(t): return t.split("\n")[0].strip()
fig,ax=plt.subplots(len(PICKS),5,figsize=(15.2,3.15*len(PICKS)))
for r,q in enumerate(PICKS):
    m=sc[q]; rec=lut[q]; im=Image.open(os.path.join(root,rec['image'])).convert("RGB")
    Wp,Hp=im.size; gt=m['gt']; O=opts(rec['text']); letters="ABCD"
    lab=arm[q]['label']; pb=arm[q]['probs']['vicrop_block']; pr=arm[q]['probs']['ridge']
    ab,ar=int(np.argmax(pb)),int(np.argmax(pr))
    # 1 full image + boxes
    a=ax[r,0]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for (cx,cy),c,lw in [(m['block_cell'],ORANGE,2.6),(m['ridge_cell'],GREEN,2.6)]:
        x0,y0,x1,y1=win(cx,cy)
        a.add_patch(Rectangle((x0*Wp,y0*Hp),(x1-x0)*Wp,(y1-y0)*Hp,fill=False,ec=c,lw=lw))
    a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec="white",lw=3.4))
    a.add_patch(Rectangle((gt[0]*Wp,gt[1]*Hp),(gt[2]-gt[0])*Wp,(gt[3]-gt[1])*Hp,fill=False,ec=RED,lw=2.0,ls=(0,(3,2))))
    q=short(rec['text']); a.set_ylabel(q if len(q)<=46 else q[:44]+'...',fontsize=9,labelpad=6)
    if r==0:
        a.set_title("input image",fontsize=10)
        from matplotlib.lines import Line2D
        a.legend(handles=[Line2D([],[],color=RED,ls='--',lw=2,label='ground truth'),
                          Line2D([],[],color=ORANGE,lw=2,label='block-mean crop'),
                          Line2D([],[],color=GREEN,lw=2,label='ridge crop')],
                 loc='lower left',fontsize=7.6,framealpha=0.9,handlelength=1.6)
    # 2 block-mean attention
    a=ax[r,1]; dep=np.asarray(m['dep']); a.imshow(dep,cmap="magma"); a.set_xticks([]); a.set_yticks([])
    jy,jx=np.unravel_index(np.argmax(dep),dep.shape); a.plot(jx,jy,"x",color=ORANGE,ms=11,mew=3)
    if r==0: a.set_title("block-mean attention L16--26",fontsize=10)
    # 3 ridge score
    a=ax[r,2]; s=np.asarray(m['score']); a.imshow(s,cmap="viridis"); a.set_xticks([]); a.set_yticks([])
    iy,ix=np.unravel_index(np.argmax(s),s.shape); a.plot(ix,iy,"x",color=GREEN,ms=11,mew=3)
    if r==0: a.set_title("TWR score (out-of-fold)",fontsize=10)
    # 4/5 the crops the model actually answers from
    for k,(cell,col,ans,name) in enumerate([(m['block_cell'],ORANGE,ab,"block-mean crop"),
                                            (m['ridge_cell'],GREEN,ar,"ridge crop")]):
        a=ax[r,3+k]; x0,y0,x1,y1=win(*cell)
        a.imshow(im.crop((int(x0*Wp),int(y0*Hp),int(x1*Wp),int(y1*Hp)))); a.set_xticks([]); a.set_yticks([])
        good=ans==lab
        a.set_xlabel(f"({letters[ans]}) {O.get(letters[ans],'?')}  {'✓' if good else '✗'}",
                     fontsize=10.5,color=GREEN if good else RED)
        for sp in a.spines.values(): sp.set_color(col); sp.set_linewidth(2.6)
        if r==0: a.set_title(name,fontsize=10)
plt.tight_layout(); plt.subplots_adjust(wspace=0.06,hspace=0.14)
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_infer_ridge.{e}",dpi=170,bbox_inches="tight")
print("wrote fig_infer_ridge")
