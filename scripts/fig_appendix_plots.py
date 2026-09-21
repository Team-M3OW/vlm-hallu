"""Four appendix figures. Every value is transcribed from the paper's own tables / the run logs."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D="/home/kavinder/ARNABI_ARSH/vlm-hallu"
G="#1a7f37"; R="#cf222e"; B="#0969da"; O="#d1710a"; GY="#57606a"
def save(n):
    for e in ("pdf","png"): plt.savefig(f"{D}/paper/figs/{n}.{e}",dpi=170,bbox_inches="tight")
    plt.close(); print("wrote",n)

# ---------- 1. transport profile ----------
q3=[0.0083,0.0049,0.0160,0.0065,0.0061,0.0071,0.0288,0.0066,0.0122,0.0589,0.0089,0.5036,0.0630,0.1012,
    0.0152,0.0084,0.0031,0.0086,0.0012,0.0017,0.0016,0.0012,0.0005,0.0006,0.0006,0.0003,0.0007,0.0004]
q2=[0.0181,0.0070,0.0070,0.0063,0.0067,0.0083,0.0064,0.0211,0.0247,0.0425,0.0063,0.0108,0.0162,0.0112,
    0.0958,0.1179,0.0112,0.0035,0.0023,0.0070,0.0023,0.0020,0.0015,0.0015,0.0016,0.0011,0.0016,0.0007]
L=[4,8,12,16,20,24]
pre3=[0.082,0.549,1.278,1.259,1.215,1.257]; suf3=[0.967,0.880,0.320,0.014,0.002,0.001]
pre2=[0.037,0.105,0.364,0.547,0.534,0.538]; suf2=[0.518,0.465,0.429,0.029,0.004,0.002]
fig,ax=plt.subplots(1,3,figsize=(13.6,3.5))
a=ax[0]
a.bar(np.arange(28)-0.2,q3,0.4,color=B,label="Qwen3-VL-2B")
a.bar(np.arange(28)+0.2,q2,0.4,color=O,label="Qwen2-VL-7B")
a.axvline(15.5,color=G,ls="--",lw=2); a.text(16.1,0.44,"transport\nboundary",color=G,fontsize=9)
a.set_xlabel("layer masked (single layer)"); a.set_ylabel("KL at the output"); a.legend(fontsize=8.5)
a.set_title("(a) which layer carries the image",fontsize=10.5)
a=ax[1]
a.plot(L,pre3,"o-",color=B,label="Qwen3 prefix $L_0..\\ell$")
a.plot(L,pre2,"o-",color=O,label="Qwen2 prefix $L_0..\\ell$")
a.axhline(1.266,color=B,ls=":",lw=1.2); a.axhline(0.532,color=O,ls=":",lw=1.2)
a.axvline(15.5,color=G,ls="--",lw=2)
a.set_xlabel("$\\ell$"); a.set_ylabel("KL"); a.legend(fontsize=8.5,loc="lower right")
a.set_title("(b) prefix mask saturates at the boundary",fontsize=10.5)
a=ax[2]
a.plot(L,suf3,"o-",color=B,label="Qwen3 suffix $L_\\ell..$")
a.plot(L,suf2,"o-",color=O,label="Qwen2 suffix $L_\\ell..$")
a.axvline(15.5,color=G,ls="--",lw=2); a.axhline(0,color="k",lw=.7)
a.set_xlabel("$\\ell$"); a.set_ylabel("KL"); a.legend(fontsize=8.5)
a.set_title("(c) suffix mask is free after it",fontsize=10.5)
plt.tight_layout(); save("fig_transport")

# ---------- 2. headroom identity ----------
cells=[("V* Qwen2.5-7B","single",12.2,12.2),("V* Qwen2.5-7B","cross",7.9,5.2),
       ("V* Qwen3-8B","single",7.0,7.0),("V* Qwen3-8B","cross",7.9,6.6),
       ("HR-4K Qwen3-2B","single",4.0,4.0),("HR-4K Qwen3-2B","cross",-2.8,-3.0),
       ("HR-4K Qwen2-7B","single",2.8,2.8),("HR-4K Qwen2-7B","cross",0.8,0.2),
       ("HR-8K Qwen3-2B","single",6.5,6.8),("HR-8K Qwen3-2B","cross",1.8,3.5)]
plt.figure(figsize=(5.0,4.6))
for nm,st,g,h in cells:
    plt.scatter(h,g,s=64,color=G if st=="single" else B,marker="o" if st=="single" else "s",
                zorder=3,edgecolor="white",lw=.8)
lim=[-4.5,14]; plt.plot(lim,lim,color=GY,ls="--",lw=1.3,label="gain = headroom")
plt.axhline(0,color="k",lw=.7); plt.axvline(0,color="k",lw=.7)
plt.xlim(lim); plt.ylim(lim)
plt.xlabel("resolution headroom  acc@900 $-$ acc@600"); plt.ylabel("AVR $-$ equal-compute bar")
from matplotlib.lines import Line2D
plt.legend(handles=[Line2D([],[],color=GY,ls="--",label="gain = headroom"),
                    Line2D([],[],marker="o",ls="",color=G,label="single-instance"),
                    Line2D([],[],marker="s",ls="",color=B,label="cross-instance")],fontsize=8.5,loc="upper left")
plt.title("AVR converts headroom at rate 1.0\n$r=0.966$, slope $1.01$, mean $|$error$|$ 0.7 pts",fontsize=10.5)
plt.grid(alpha=.25); plt.tight_layout(); save("fig_identity")

# ---------- 3. scope law ----------
names=["V* Qwen3-2B","V* Qwen2-7B","V* Qwen2.5-7B","V* Qwen3-8B","HR-4K Qwen3-2B","HR-4K Qwen2-7B","HR-8K Qwen3-2B"]
rs=[14.8,13.9,12.2,6.1,10.2,9.2,8.8]; rc=[0.0,9.2,-2.6,-3.9,-8.8,-6.0,-8.8]
ts=[3.5,3.5,12.2,7.0,4.0,2.8,6.5];    tc=[10.5,1.3,7.9,7.9,-2.8,0.8,1.8]
y=np.arange(len(names)); h=0.36
fig,ax=plt.subplots(1,2,figsize=(11.4,3.6),sharey=True)
for a,(s,c,t) in zip(ax,[(rs,rc,"DWA crop"),(ts,tc,"AVR")]):
    a.barh(y+h/2,s,h,color=G,label="single-instance")
    a.barh(y-h/2,c,h,color=R,label="cross-instance")
    a.axvline(0,color="k",lw=.9); a.set_yticks(y); a.set_yticklabels(names,fontsize=9)
    a.invert_yaxis(); a.set_xlabel("accuracy $-$ equal-compute bar (points)")
    a.set_title(t,fontsize=11); a.grid(axis="x",alpha=.25); a.set_xlim(-11,16)
ax[0].legend(fontsize=8.5,loc="lower right")
plt.tight_layout(); save("fig_scope")

# ---------- 4. accuracy vs compute ----------
arms=[("uniform@600 (bar)",600,63.7,58.1,GY),("block-mean arg-max",600,62.1,59.2,O),
      ("LASER",600,64.7,61.3,B),("AVR (ours)",600,70.0,61.8,"#8250df"),
      ("DWA (ours)",600,72.6,70.2,G),("oracle crop",600,87.9,87.4,"#b45309")]
fig,ax=plt.subplots(1,2,figsize=(11.4,3.9))
for a,(k,nat,ntok,mdl) in zip(ax,[(2,76.3,3290,"Qwen3-VL-2B ($n{=}190$)"),(3,71.2,4320,"Qwen2-VL-7B ($n{=}191$)")]):
    a.axhline(nat,color=R,ls="--",lw=1.8)
    a.scatter([ntok],[nat],s=90,color=R,zorder=4,marker="*")
    a.annotate(f"native dynamic resolution\n{ntok:,} tokens",(ntok,nat),textcoords="offset points",
               xytext=(-8,-30),fontsize=8.6,color=R,ha="right")
    pts=sorted([(v3 if k==2 else v2,name,tk,col) for name,tk,v3,v2,col in arms])
    lo=min(p[0] for p in pts); hi=max(p[0] for p in pts); gap=(hi-lo)*0.075
    ylab=[]
    for v,name,tk,col in pts:
        yl=v if not ylab else max(v,ylab[-1]+gap); ylab.append(yl)
    for (v,name,tk,col),yl in zip(pts,ylab):
        a.scatter([tk],[v],s=62,color=col,zorder=3,edgecolor="white",lw=.7)
        a.annotate(name,(tk,v),xytext=(tk*1.35,yl),textcoords="data",fontsize=8.4,color=col,
                   va="center",arrowprops=dict(arrowstyle="-",color=col,lw=.6,alpha=.55))
    a.set_xscale("log"); a.set_xlim(420,11000); a.set_xlabel("visual tokens (log)")
    a.set_ylabel("accuracy (%)"); a.set_title(mdl,fontsize=10.5); a.grid(alpha=.25)
plt.tight_layout(); save("fig_native")
