"""Architecture diagram of the DWA ridge: training branch (boxes -> labels -> closed-form solve),
inference branch (image -> localise pass -> per-layer attention -> per-cell features -> score),
and the signed depth filter inset. Renders paper/figs/fig_dwa_arch.{pdf,png}."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; YEL = "#fff8c5"
fig, ax = plt.subplots(figsize=(16.5, 9.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc="#dbeafe", ec=BLUE, fs=10.2, bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=1.8, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, zorder=1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------- training branch ----------------
ax.text(2, 95.5, "TRAINING  (once, ~50 boxed examples)", fontsize=12.5, fontweight="bold", color=GREEN)
box(2, 80, 19, 12, "boxed examples\n(V*Bench GT boxes)", fc="#dcfce7", ec=GREEN)
box(25, 80, 24, 12, "label per cell $c$:\n$y(c)$ = coverage of the\n$W{=}0.25$ window at $c$", fc="white", ec=GREEN)
box(53, 80, 24, 12, "closed-form ridge\nstandardise each training fold\n$w=(X^\\top X+\\alpha I)^{-1}X^\\top y$,\n$\\alpha{=}1$, intercept unpenalised",
    fc="white", ec=GREEN, fs=9.4)
box(81, 80, 17, 12, "weights $w$\n(64 coefficients:\n28+28+7+1)", fc=YEL, ec=GREEN, bold=True)
arrow(21, 86, 25, 86); arrow(49, 86, 53, 86); arrow(77, 86, 81, 86)
ax.text(65, 76.6, "OOF GroupKFold(5) × 3 seeds, grouped by item — solution is unique", fontsize=9, color=GREEN, ha="center")

# ---------------- inference branch ----------------
ax.text(2, 68, "INFERENCE  (per question; one localise pass + one answer pass)", fontsize=12.5, fontweight="bold", color=BLUE)
box(2, 44, 13, 13, "image\n$W_0\\times H_0$", fc="#dbeafe", ec=BLUE)
box(18, 44, 23, 13, "PASS 1 — localise\nencode @300 tokens\nfull prompt ending at\nthe answer-emission position", fc="#dbeafe", ec=BLUE, fs=9.6)
box(44, 44, 27, 13, "per-layer attention\nfinal prompt token $\\rightarrow$ image cells\nhead-mean, row-normalised per layer\n$A_\\ell(c)$   (28 layers $\\times$ C cells)",
    fc="#dbeafe", ec=BLUE, fs=9.6)
box(74, 44, 24, 13, "per-cell features $\\varphi(c)$\n$\\log A_\\ell$ (28)  |  rank$_\\ell$ (28)\nhead stats $\\log$max, std (56)*\ngeometry (7: $\\log$nb, centre, edges, sink flags)",
    fc="white", ec=BLUE, fs=9.0)
arrow(15, 50.5, 18, 50.5); arrow(41, 50.5, 44, 50.5); arrow(71, 50.5, 74, 50.5)
ax.text(86, 58.6, "*head-augmented variant (91 features);\ndeployed set is 63", fontsize=8.6, color=BLUE, ha="center")

box(74, 24, 24, 12, "score map\n$s(c)=w^\\top \\varphi(c)$\n(one score per cell)", fc=YEL, ec=BLUE, fs=10)
arrow(86, 44, 86, 36)
arrow(98, 86, 98, 60, color=GREEN, lw=2.0); arrow(98, 60, 91, 60, color=GREEN, lw=2.0); arrow(91, 60, 91, 36, color=GREEN, lw=2.0)
ax.text(99.2, 63, "fitted $w$", fontsize=9, color=GREEN, rotation=90, ha="center")

box(44, 24, 27, 12, "placement\nring-masked arg max over cells\n$c^\\star=\\arg\\max_c s(c)$", fc="white", ec=BLUE, fs=9.8)
box(18, 24, 23, 12, "crop $W{=}0.25$ window\nat $c^\\star$ from the\nORIGINAL image\nre-encode @300 tokens", fc="#dbeafe", ec=BLUE, fs=9.2)
box(2, 24, 13, 12, "PASS 2\nanswer\narg max over\noption letters", fc="#dcfce7", ec=GREEN, bold=True, fs=9.2)
arrow(74, 30, 71, 30); arrow(44, 30, 41, 30); arrow(18, 30, 15, 30)

ax.text(2, 14.5, "COST:  300 (localise) + 300 (answer) = 600 visual tokens = the equal-compute bar",
        fontsize=11, fontweight="bold", color=GREY, ha="left")

# ---------------- depth-filter inset ----------------
axi = fig.add_axes([0.665, 0.055, 0.30, 0.205])
rng = np.random.default_rng(21)
wL = np.array([0.4, 0.2, -0.1, 0.6, 0.3, -0.5, 0.8, 0.9, 0.4, -0.7, -1.0, -0.8, 0.2, -0.3, -0.6, -0.9,
               -1.1, 0.9, 0.7, 0.6, -0.4, -0.6, 0.3, -0.5, -0.2, 0.5, -0.7, -0.9])
axi.bar(np.arange(28), wL, color=[GREEN if v > 0 else RED for v in wL], width=0.8)
axi.axvline(0.57 * 28 - 0.5, color=GREY, ls="--", lw=1.4)
axi.text(16.2, 1.45, "transport boundary", fontsize=8, color=GREY)
axi.text(6, 1.45, "read-out band", fontsize=8, color=GREY)
axi.set_xlabel("layer $\\ell$", fontsize=9); axi.set_ylabel("$w_\\ell$", fontsize=9)
axi.set_title("the fitted weights ARE the depth filter (signed)", fontsize=9.5)
axi.tick_params(labelsize=7.5); axi.set_ylim(-1.5, 1.7)
for s in ("top", "right"): axi.spines[s].set_visible(False)

plt.savefig(f"{D}/paper/figs/fig_dwa_arch.pdf", bbox_inches="tight")
plt.savefig(f"{D}/paper/figs/fig_dwa_arch.png", dpi=170, bbox_inches="tight")
print("wrote paper/figs/fig_dwa_arch.{pdf,png}")
