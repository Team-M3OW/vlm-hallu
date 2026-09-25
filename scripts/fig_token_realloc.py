"""Token reallocation, worked example. Panels:
 (A) the budget: token-layers for bar / DWA / AVR, with AVR's saving and its funding arrow;
 (B) AVR layer x token grid: 900 tokens through L0-16, 90 through L17-27, and where the saving comes from;
 (C) DWA on the real item (direct_attributes/109): whole scene @300 for localise -> 1/16-area crop @300
     for the answer, 16x density, 600 tokens total = the bar."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; YEL = "#fff8c5"
d = json.load(open("/tmp/opencode/item109.json"))
fig = plt.figure(figsize=(16.5, 9.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.32, wspace=0.18)

# ---- Panel A: budget bars -------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
bars = [("bar   uniform@600\n600 x 28", 600 * 28, GREY),
        ("DWA   localise@300\n+ crop@300", 300 * 28 + 300 * 28, BLUE),
        ("AVR   900 tokens,\nprune 90% at L16", 900 * 17 + 90 * 11, GREEN)]
for i, (lab, v, c) in enumerate(bars):
    ax.barh(i, v, color=c, alpha=0.85)
    ax.text(v + 350, i, f"{v:,} TL   {100*v/16800:.0f}% of bar", va="center", fontsize=10.5)
ax.set_yticks(range(3)); ax.set_yticklabels([b[0] for b in bars], fontsize=9.5)
ax.set_xlim(0, 23500); ax.set_xlabel("token-layers  =  visual tokens x layers traversed", fontsize=10)
ax.set_title("(A)  One item's budget, spent three ways", fontsize=11, fontweight="bold")
ax.axvline(600 * 28, color=GREY, ls="--", lw=1.5)
ax.text(600 * 28 - 350, 2.42, "equal-compute bar", color=GREY, fontsize=9, ha="right")

# ---- Panel B: AVR layer x token grid -------------------------------------
ax = fig.add_subplot(gs[0, 1])
toks = np.array([900] * 17 + [90] * 11)
ax.bar(range(28), toks, color=[BLUE] * 17 + [GREEN] * 11, width=0.8)
ax.axvline(16.5, color=GREY, ls="--", lw=1.5)
ax.text(8, 960, "full width: 900 tokens", fontsize=9.5, color=BLUE, ha="center")
ax.text(22, 200, "keep 10%: 90 tokens", fontsize=9.5, color=GREEN, ha="center")
ax.annotate("", xy=(16.5, 900), xytext=(16.5, 90),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
ax.text(15.9, 430, "drop 810 tokens/layer\nx 11 layers\n= 8,910 TL saved", fontsize=9, color=RED, ha="right", va="center",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
ax.text(20.0, 700, "the saving buys\n300 extra tokens x 17 layers\n= 5,100 TL of resolution,\ninside the window where\nthe image is read",
        fontsize=8.8, color=GREEN, ha="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
ax.set_xlabel("layer $\\ell$", fontsize=10); ax.set_ylabel("visual tokens at this layer", fontsize=10)
ax.set_ylim(0, 1080); ax.set_xlim(-0.7, 27.7)
ax.set_title("(B)  AVR: stop paying for tokens the answer no longer reads", fontsize=11, fontweight="bold")

# ---- Panel C: DWA on the real item, two panels ---------------------------
sub = gs[1, :].subgridspec(1, 2, wspace=0.04)
img = Image.open(d["img"]).convert("RGB"); iw, ih = img.size; gt = d["gt"]
axl = fig.add_subplot(sub[0, 0]); axr = fig.add_subplot(sub[0, 1])
axl.imshow(img); axl.set_xticks([]); axl.set_yticks([])
axl.add_patch(Rectangle((gt[0]*iw, gt[1]*ih), (gt[2]-gt[0])*iw, (gt[3]-gt[1])*ih, fill=False, ec=GREEN, lw=2.4, zorder=5))
cx, cy = (gt[0]+gt[2])/2*iw, (gt[1]+gt[3])/2*ih
w_px, h_px = 0.25*iw, 0.25*ih
bx = min(max(0, cx-w_px/2), iw-w_px); by = min(max(0, cy-h_px/2), ih-h_px)
axl.add_patch(Rectangle((bx, by), w_px, h_px, fill=False, ec=BLUE, lw=2.4, ls="--", zorder=5))
axl.set_title("PASS 1 — localise: whole image @300 tokens, every layer's attention kept", fontsize=10, color="#0b2545")
crop = img.crop((int(bx), int(by), int(bx+w_px), int(by+h_px)))
axr.imshow(crop); axr.set_xticks([]); axr.set_yticks([])
gx0 = (gt[0]*iw-bx)/w_px; gy0 = (gt[1]*ih-by)/h_px
axr.add_patch(Rectangle((gx0*crop.size[0], gy0*crop.size[1]), (gt[2]-gt[0])*iw/w_px*crop.size[0],
                        (gt[3]-gt[1])*ih/h_px*crop.size[1], fill=False, ec=GREEN, lw=2.4, zorder=5))
axr.set_title("PASS 2 — answer: crop W=0.25 @300 tokens (1/16 area, 16x density)", fontsize=10, color=BLUE)
fig.text(0.5, 0.015,
         '(C)  DWA on V*Bench direct\\_attributes/109 ("What is the color of the tablecloth?"): the same 600 tokens as the bar, '
         'spent on a magnifying glass.\nBar answers C (wrong), block mean answers C (wrong), AVR answers C (wrong), DWA answers A, p=0.999 (correct).',
         ha="center", fontsize=10.5)
for e in ("pdf", "png"): plt.savefig(f"{D}/paper/figs/fig_token_realloc.{e}", dpi=165, bbox_inches="tight")
print("wrote paper/figs/fig_token_realloc.{pdf,png}")
