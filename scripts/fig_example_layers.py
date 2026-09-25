"""Worked example, layer by layer: V*Bench direct_attributes/109 ("What is the color of the tablecloth?").
Left: the image with the GT box and each chosen layer's ring-masked attention argmax.
Right: per-layer coverage of the GT box by the W=0.25 window at that layer's arg-max, with the
transport boundary and the deployed block-mean read-out marked."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
d = json.load(open("/tmp/opencode/item109.json"))
img = Image.open(d["img"]).convert("RGB"); iw, ih = img.size
gh, gw = d["grid"]; gt = d["gt"]
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})

ax = axes[0]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
ax.add_patch(Rectangle((gt[0] * iw, gt[1] * ih), (gt[2] - gt[0]) * iw, (gt[3] - gt[1]) * ih,
                       fill=False, ec="#1a7f37", lw=2.6, zorder=5))
ax.text(gt[0] * iw - 12, (gt[1] + gt[3]) / 2 * ih, "ground truth\n(tablecloth)", color="#1a7f37", fontsize=10,
        fontweight="bold", ha="right", va="center")
for l, col, lab in [(4, "#6e7781", "L4"), (12, "#bf8700", "L12"), (19, "#1a7f37", "L19"), (20, "#1a7f37", "L20"), (27, "#cf222e", "L27")]:
    r_, c_ = d["cells"][str(l)]
    ax.add_patch(Circle(((c_ + .5) / gw * iw, (r_ + .5) / gh * ih), radius=0.028 * iw,
                        fill=False, ec=col, lw=2.4, zorder=6))
    dy = {"L4": -0.035 * ih, "L12": 0.02 * ih, "L19": -0.045 * ih, "L20": 0.045 * ih, "L27": 0.0}[lab]
    ax.text((c_ + .5) / gw * iw + 0.018 * iw, (r_ + .5) / gh * ih + dy, lab, color=col, fontsize=10, zorder=7)
rb, cb_ = d["block_cell"]
ax.plot([(cb_ + .5) / gw * iw], [(rb + .5) / gh * ih], marker="X", ms=14, color="#cf222e", zorder=7)
ax.text((cb_ + .5) / gw * iw + 0.02 * iw, (rb + .5) / gh * ih - 0.03 * ih, "block mean", color="#cf222e", fontsize=10, zorder=7)
ax.set_title(r'V*Bench direct\_attributes/109 — "What is the color of the tablecloth?"\n'
             'bar: wrong (C)   block mean: wrong (C)   AVR: wrong (C)   DWA: CORRECT (A, p=0.999)', fontsize=10.5)

ax = axes[1]
covs = [d["covs"][str(l)] for l in range(28)]
cols = ["#1a7f37" if c >= .5 else "#cf222e" for c in covs]
ax.bar(range(28), covs, color=cols, width=0.75)
ax.axvspan(15.5, 21.5, color="#fff8c5", zorder=0)
ax.axvline(15.5, color="#57606a", ls="--", lw=1.6)
ax.axhline(0.0, color="#cf222e", ls=":", lw=1.8)
ax.text(6.0, 1.06, "transport window (image \u2192 text)", fontsize=9, color="#57606a", ha="center")
ax.text(18.5, 1.06, "read-out band", fontsize=9, color="#57606a", ha="center")
ax.annotate("L19\u2013L20 find the\ntablecloth (0.94\u20131.00)", xy=(19.5, 0.95), xytext=(12.0, 0.78),
            fontsize=9, color="#1a7f37", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1a7f37", lw=1.2))
ax.text(24.8, 0.62, "late layers: attention\nreturns to the sink", fontsize=9, color="#cf222e", ha="center")
ax.text(0.5, 0.10, "block mean (L16\u201326) = 0.00: outvoted 9\u20132", fontsize=9.5, color="#cf222e")
ax.set_xlabel("read-out layer $\\ell$"); ax.set_ylabel("GT-box coverage of the W=0.25 crop at that layer's arg-max")
ax.set_ylim(0, 1.18); ax.set_xlim(-0.7, 27.7); ax.set_title("Where the attention points, layer by layer", fontsize=10.5)
ax.grid(axis="y", alpha=.25)
plt.tight_layout()
for e in ("pdf", "png"): plt.savefig(f"{D}/paper/figs/fig_example_layers.{e}", dpi=170, bbox_inches="tight")
print("wrote paper/figs/fig_example_layers.{pdf,png}")
