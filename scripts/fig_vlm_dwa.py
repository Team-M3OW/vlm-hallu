"""VLM internals + DWA diagram: the encoder->projection->decoder pipeline with the transport
boundary and read-out bands (top), and the depth-weighted read-out and crop loop (bottom).
Renders paper/figs/fig_vlm_dwa.{pdf,png}."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; YEL = "#fff8c5"
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(15.5, 8.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc="#dbeafe", ec=BLUE, fs=9.2, bold=False, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(x1, y1, x2, y2, color=GREY, lw=1.7, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, zorder=1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ================= (a) VLM internals =================
ax.text(1, 97, "(a)  VLM internals — where the map comes from", fontsize=12, fontweight="bold", color=BLUE)
box(1, 76, 9, 11, "image\n$I$", fc="#dbeafe", ec=BLUE)
box(13, 76, 14, 11, "vision\nencoder $E$", fc="#dbeafe", ec=BLUE)
box(30, 76, 16, 11, "projection $P$\n$V=P(E(I))\\in\\mathbb{R}^{N\\times d}$", fc="#dbeafe", ec=BLUE, fs=8.6)
arrow(10, 81.5, 13, 81.5); arrow(27, 81.5, 30, 81.5)

dx, dy, dw, dh = 50, 62, 12, 29
nL = 28; bh = dh / nL
for l in range(nL):
    y = dy + l * bh
    if l < 16:   fc, ec = "#dbeafe", BLUE
    elif l == 16: fc, ec = "#fde8e8", RED
    elif l <= 21: fc, ec = "#fff3b0", "#b58900"
    else:        fc, ec = "#eceff1", GREY
    ax.add_patch(Rectangle((dx, y), dw, bh, fc=fc, ec=ec, lw=0.7, zorder=2))
ax.add_patch(Rectangle((dx, dy), dw, dh, fill=False, ec=BLUE, lw=1.4, zorder=3))
ax.text(dx + dw / 2, 92.2, "language decoder $F_\\theta$   ($L=28$)", fontsize=9.2, ha="center", color=BLUE)
arrow(46, 81.5, 50, 81.5)
ax.text(dx + dw / 2, 59.8, "read at every layer", fontsize=8.0, color=BLUE, ha="center")

box(66, 87, 12, 7, "logits $z$\nat final prompt position", fc="#dcfce7", ec=GREEN, fs=8.2)
arrow(62, 90, 66, 90, color=GREEN)
box(81, 87, 8, 7, "answer", fc="#dcfce7", ec=GREEN, bold=True, fs=8.8)
arrow(78, 90.5, 81, 90.5, color=GREEN)
box(66, 54, 29, 11, "per-layer attention $A_\\ell(c)$\nfinal prompt token $\\to$ image cells\nhead-mean, row-normalised per layer\n(28 maps, one per layer)",
    fc="white", ec=BLUE, fs=8.2)
arrow(62, 66, 66, 59.5, color=BLUE, lw=1.5)

ax.text(63.6, 84.5, "answer formation L22+  (grey)\nDWA still reads these layers", fontsize=8.0, color=GREY, va="center")
ax.text(63.6, 77.5, "read-out band L17–21  (yellow)\nblock-mean methods read here", fontsize=8.0, color="#8a6d00", va="center")
ax.text(63.6, 72.0, "L16 — transport complete ($0.57L$): image$\\to$text,\nlater image attention is causally inert", fontsize=8.0, color=RED, va="center")
ax.text(63.6, 68.3, "transport window L0–15  (blue)", fontsize=8.0, color=BLUE, va="center")

box(33, 60, 15, 8, "question + options\nprompt ends at the\nanswer-emission token", fc="white", ec=GREY, fs=8.0)
arrow(45, 64, 50, 66, color=GREY)

# ================= (b) DWA =================
ax.text(1, 52, "(b)  DWA — depth-weighted read-out and crop", fontsize=12, fontweight="bold", color=GREEN)
box(1, 33, 10, 13, "28 maps\n$A_\\ell(c)$", fc="#f8c9c9", ec=RED, fs=8.8)
box(14, 34, 20, 11, "per-cell depth profile\n$x_c=\\log A_\\ell(c)\\in\\mathbb{R}^{L}$\n$+$ rank $+$ geometry: $2L{+}7$", fc="white", ec=BLUE, fs=8.6)
box(37, 34, 20, 11, "closed-form ridge\n$w=(X^\\top X+\\alpha I)^{-1}X^\\top y$\n$\\alpha{=}1$; $\\sim$50 boxed examples", fc="#dcfce7", ec=GREEN, fs=8.6)
box(60, 34, 17, 11, "score map\n$s(c)=w_0+\\sum_\\ell w_\\ell\\log A_\\ell(c)$", fc=YEL, ec=BLUE, fs=8.6)
box(80, 34, 12, 11, "ring-masked\nargmax $c^\\star$", fc="white", ec=BLUE, fs=8.6)
arrow(11, 39.5, 14, 39.5); arrow(34, 39.5, 37, 39.5); arrow(57, 39.5, 60, 39.5); arrow(77, 39.5, 80, 39.5)
arrow(86, 34, 86, 27)
box(77, 16, 15, 11, "crop $W{=}0.25$\nat $c^\\star$ from the\noriginal image", fc="#dbeafe", ec=BLUE, fs=8.6)
box(55, 16, 19, 11, "re-encode crop\n@300 tokens\n(pass 2)", fc="#dbeafe", ec=BLUE, fs=8.6)
box(31, 16, 20, 11, "answer\nargmax over option letters\n(option-letter scoring)", fc="#dcfce7", ec=GREEN, fs=8.6)
arrow(77, 21.5, 74, 21.5); arrow(55, 21.5, 51, 21.5)
ax.text(1, 9.5, "cost: 300 (localise) + 300 (answer) = 600 visual tokens = the equal-compute bar",
        fontsize=10, fontweight="bold", color=GREY)
ax.text(1, 4.5, "the ridge runs on the full depth: the causally inert late layers receive small or negative weight, the question-conditioned band is up-weighted",
        fontsize=9, color=GREEN)
for e in ("pdf", "png"): fig.savefig(f"{D}/paper/figs/fig_vlm_dwa.{e}", dpi=170, bbox_inches="tight")
print("saved fig_vlm_dwa")
