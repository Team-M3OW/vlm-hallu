"""The three-stage mechanism, each stage measured by a different intervention (phase 203 + 199)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"
BLUE="#0969da"; ORANGE="#d1710a"; GREEN="#1a7f37"; RED="#cf222e"; GREY="#57606a"
PL=[0,4,8,12,16,20,24]
def load(w): return list({json.loads(l)['question_id_full']:json.loads(l) for l in open(f"{D}/data/phase203_patchlens_{w}.jsonl")}.values())
fig,ax=plt.subplots(1,3,figsize=(14.6,4.0))
# (a) image-token patching by depth, both models
a=ax[0]
for w,lbl,col in (("qwen3","Qwen3-VL-2B",BLUE),("qwen2","Qwen2-VL-7B",ORANGE)):
    r=load(w); y=[np.mean([x['patch'][f'img_L{l}']['kl'] for x in r]) for l in PL]
    a.plot(PL,y,"o-",color=col,ms=5,lw=1.9,label=lbl)
a.axvline(16,color=GREEN,ls="--",lw=2)
a.text(16.5,0.35,"transport\nboundary",color=GREEN,fontsize=8.5)
a.set_yscale("log"); a.set_xlabel("layer at which image tokens are replaced")
a.set_ylabel("KL at the output"); a.legend(fontsize=8.5)
a.set_title("(a) swap the image's hidden states\nfor a different image's",fontsize=10)
a.grid(alpha=.25,which="both")
# (b) the crossover
a=ax[1]
w_=0.34; xs=np.arange(2)
for i,(w,lbl,col) in enumerate((("qwen3","Qwen3-VL-2B",BLUE),("qwen2","Qwen2-VL-7B",ORANGE))):
    r=load(w)
    v=[np.mean([x['patch']['img_L4']['kl'] for x in r])/max(np.mean([x['patch']['txt_L4']['kl'] for x in r]),1e-9),
       np.mean([x['patch']['txt_L20']['kl'] for x in r])/max(np.mean([x['patch']['img_L20']['kl'] for x in r]),1e-9)]
    a.bar(xs+(i-0.5)*w_,v,w_,color=col,label=lbl)
    for x,val in zip(xs+(i-0.5)*w_,v): a.text(x,val*1.15,f"{val:.0f}$\\times$",ha="center",fontsize=8.4,color=col)
a.set_yscale("log"); a.set_xticks(xs)
a.set_xticklabels(["at L4:\nimage matters, text does not","at L20:\ntext matters, image does not"],fontsize=9)
a.set_ylabel("ratio of output KL"); a.legend(fontsize=8.5)
a.set_title("(b) the hand-off, as a ratio",fontsize=10); a.grid(axis="y",alpha=.25,which="both")
# (c) logit lens
a=ax[2]
for w,lbl,col,cut in (("qwen3","Qwen3-VL-2B",BLUE,22),("qwen2","Qwen2-VL-7B",ORANGE,23)):
    r=load(w); NLp=len(r[0]['lens'])
    y=[100*np.mean([x['lens'][f'L{l}']['argmax']==x['label'] for x in r]) for l in range(NLp)]
    a.plot(range(NLp),y,"o-",color=col,ms=3.4,lw=1.7,label=lbl)
a.axhline(25,color=GREY,ls=":",lw=1.3); a.text(0.4,25.8,"chance",fontsize=8,color=GREY)
a.axvline(16,color=GREEN,ls="--",lw=2); a.axvline(22,color=RED,ls="--",lw=1.8)
a.text(22.4,17,"answer\nbecomes\ndecodable",color=RED,fontsize=8.2)
a.set_xlabel("layer read by the logit lens"); a.set_ylabel("lens arg-max accuracy (%)")
a.legend(fontsize=8.5,loc="upper left"); a.grid(alpha=.25)
a.set_title("(c) when the answer appears",fontsize=10)
plt.tight_layout()
for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/fig_mechanism.{e}",dpi=170,bbox_inches="tight")
print("wrote fig_mechanism")
