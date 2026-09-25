"""Combined VLM-internals + DWA diagram: one pipeline from image to answer, with the ridge read-out
inline (attention at every layer -> per-cell log-attention profile -> ridge -> score -> argmax ->
crop -> re-encode -> pass 2). Renders paper/figs/fig_vlm_dwa.{pdf,png}."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; YEL = "#fff8c5"
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(15.5, 8.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc="#dbeafe", ec=BLUE, fs=8.6, bold=False, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(x1, y1, x2, y2, color=GREY, lw=1.7, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, zorder=1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


ax.text(1, 99.0, "VLM internals + DWA — how the ridge read-out fits into the model", fontsize=13, fontweight="bold", color=BLUE)

# ---------------- top: the VLM pipeline ----------------
box(1, 80, 8, 10, "image\n$I$", fs=8.8)
box(11, 80, 12, 10, "vision\nencoder $E$", fs=8.8)
box(25, 80, 14, 10, "projection $P$\n$V=P(E(I))\\in\\mathbb{R}^{N\\times d}$", fs=8.2)
arrow(9, 85, 11, 85); arrow(23, 85, 25, 85)

dx, dy, dw, dh = 42, 66, 12, 30
nL = 28; bh = dh / nL
for l in range(nL):
    y = dy + l * bh
    if l < 16:   fc, ec = "#dbeafe", BLUE
    elif l == 16: fc, ec = "#fde8e8", RED
    elif l <= 21: fc, ec = "#fff3b0", "#b58900"
    else:        fc, ec = "#eceff1", GREY
    ax.add_patch(Rectangle((dx, y), dw, bh, fc=fc, ec=ec, lw=0.7, zorder=2))
ax.add_patch(Rectangle((dx, dy), dw, dh, fill=False, ec=BLUE, lw=1.4, zorder=3))
ax.text(dx + dw + 1.0, 93.5, "language decoder $F_\\theta$  ($L=28$)", fontsize=9, ha="left", color=BLUE)
arrow(39, 85, 42, 85)
box(58, 84, 13, 7, "logits $z$ at final\nprompt position", fc="#dcfce7", ec=GREEN, fs=8.0)
arrow(54, 88, 58, 88, color=GREEN)
box(74, 84, 9, 7, "answer", fc="#dcfce7", ec=GREEN, bold=True, fs=8.8)
arrow(71, 87.5, 74, 87.5, color=GREEN)
box(40, 54, 14, 8, "question + options\nprompt ends at the\nanswer-emission token", fc="white", ec=GREY, fs=7.8)
arrow(47, 62, 47, 66, color=GREY)

ax.text(55.5, 79.5, "L16 — transport complete ($0.57L$): image$\\to$text,\nlater image attention is causally inert",
        fontsize=7.8, color=RED, va="center")
ax.text(55.5, 72.0, "blue: transport window L0–15        yellow: read-out band L17–21\n"
                    "red: boundary L16                        grey: answer formation L22+ (DWA still reads these)",
        fontsize=7.8, color="#333333", va="center")

# ---------------- the ridge chain, inline ----------------
arrow(44, 66, 18, 46, color=BLUE, lw=1.6, rad=0.18)
ax.text(27, 61.0, "attention $A_\\ell(c)$ read at every layer:\nfinal prompt token $\\to$ image cells, head-mean, row-normalised",
        fontsize=7.8, color=BLUE, ha="center")
box(1, 34, 17, 12, "per-layer attention\n$A_\\ell(c)$ — 28 maps\n(one per layer)", fc="#f8c9c9", ec=RED, fs=7.8)
box(21, 35, 16, 10, "per-cell depth profile\n$x_c=\\log A_\\ell(c)\\in\\mathbb{R}^{L}$\n(one feature per layer)", fc="white", ec=BLUE, fs=7.8)
box(40, 35, 18, 10, "closed-form ridge\n$w=(X^\\top X+\\alpha I)^{-1}X^\\top y$\n$\\alpha{=}1$; $\\sim$50 boxed examples", fc="#dcfce7", ec=GREEN, fs=7.8)
box(61, 35, 16, 10, "score map\n$s(c)=w_0+\\sum_\\ell w_\\ell\\log A_\\ell(c)$", fc=YEL, ec=BLUE, fs=7.8)
box(80, 35, 12, 10, "ring-masked\nargmax $c^\\star$", fc="white", ec=BLUE, fs=7.8)
arrow(18, 40, 21, 40); arrow(37, 40, 40, 40); arrow(58, 40, 61, 40); arrow(77, 40, 80, 40)
arrow(86, 35, 86, 28)
box(80, 17, 12, 11, "crop $W{=}0.25$\nat $c^\\star$ from the\noriginal image", fc="#dbeafe", ec=BLUE, fs=7.8)
box(56, 17, 19, 11, "re-encode crop\n@300 tokens\n(PASS 2)", fc="#dbeafe", ec=BLUE, fs=7.8)
arrow(80, 22.5, 75, 22.5)
arrow(92, 22.5, 96.5, 22.5, color=GREEN, lw=1.8)
arrow(96.5, 22.5, 96.5, 80.0, color=GREEN, lw=1.8)
arrow(96.5, 80.0, 83, 84.8, color=GREEN, lw=1.8)
ax.text(98.0, 52, "PASS 2: answer on the crop", fontsize=7.8, color=GREEN, rotation=90, va="center")

ax.text(1, 9.0, "cost: 300 (localise) + 300 (answer) = 600 visual tokens = the equal-compute bar",
        fontsize=9.6, fontweight="bold", color=GREY)
ax.text(1, 4.0, "the ridge runs on the full depth: the causally inert late layers receive small or negative weight, the question-conditioned band is up-weighted",
        fontsize=8.4, color=GREEN)
for e in ("pdf", "png"): fig.savefig(f"{D}/paper/figs/fig_vlm_dwa.{e}", dpi=170, bbox_inches="tight")
print("saved fig_vlm_dwa")
