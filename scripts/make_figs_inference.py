"""Figures built from ACTUAL INFERENCE (phase 85), not from illustration.

Every pixel shown is something the model really received, and every label is something it really
answered. Nothing is hand-placed and nothing is recomputed at figure time.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gs
from PIL import Image

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
F = "/home/kavinder/ARNABI_ARSH/vlm-hallu/paper/figs"
CR = f"{F}/crops"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 220, "savefig.bbox": "tight"})
BAD, GOOD, HL = "#c0392b", "#1a6b54", "#2c5aa0"
R = json.load(open(f"{D}/phase85_figure_inference.json"))


def save(fig, n):
    fig.savefig(f"{F}/{n}.png"); fig.savefig(f"{F}/{n}.pdf"); plt.close(fig)


def box(ax, b, W, H, c, lw=1.6):
    ax.add_patch(Rectangle((b[0]*W, b[1]*H), (b[2]-b[0])*W, (b[3]-b[1])*H,
                           fill=False, ec=c, lw=lw))


def heat_on_image(ax, img, m, gh, gw, alpha=0.62):
    """Upsample the real attention map onto the real image and blend."""
    ax.imshow(img)
    h = np.asarray(Image.fromarray((m / m.max() * 255).astype(np.uint8))
                   .resize(img.size, Image.BILINEAR)) / 255.0
    ax.imshow(h, cmap="inferno", alpha=alpha, extent=(0, img.size[0], img.size[1], 0))
    ax.axis("off")


# ---------------------------------------------------------------- FIG 1: attention ON the image
def fig1():
    r = R[0]
    gh, gw = r["grid"]
    img = Image.open(f"{CR}/{r['qid'].replace('/','_')}_full.png").convert("RGB")
    W, H = img.size
    late = np.asarray(r["attn_late"]).reshape(gh, gw)
    l2 = np.asarray(r["attn_layer2"]).reshape(gh, gw)
    fig = plt.figure(figsize=(7.2, 2.35))
    g = gs.GridSpec(1, 3, wspace=0.06)
    for k, (m, t, c) in enumerate([(None, "the image the model sees", "k"),
                                   (l2, "attention at layer 2", BAD),
                                   (late, "attention at layers 16–26", GOOD)]):
        ax = fig.add_subplot(g[k])
        if m is None:
            ax.imshow(img); ax.axis("off")
        else:
            heat_on_image(ax, img, m, gh, gw)
            j = int(np.argmax(np.where(np.arange(gh*gw).reshape(gh, gw) >= 0, m, m)))
            ax.plot(((j % gw)+.5)/gw*W, ((j//gw)+.5)/gh*H, "x", ms=9, mew=2.2, color=c)
        box(ax, r["gt_box_frac"], W, H, "#f1c40f", 2.0)
        ax.set_title(t, fontsize=8, color=c)
    fig.text(0.5, -0.03, f'"{r["question"].split("(A)")[0].strip()}"',
             ha="center", fontsize=7.4, style="italic")
    save(fig, "fig1_defect")
    print("fig1 ok (attention overlaid on the real image)")


# ---------------------------------------------------------------- FIG: what pruning throws away
def fig_prune_masks():
    # STATED RULE: the first three items on which the two rankings produce DIFFERENT answers.
    # Selection is on disagreement, never on which ranking turns out to be right -- on the first
    # item both are wrong, and that is shown.
    items = [r for r in R
             if r["answers"]["pruned_layer2"]["pred"] != r["answers"]["pruned_late"]["pred"]][:3]
    agree_late = sum(r["answers"]["pruned_late"]["pred"] == r["answers"]["full"]["pred"] for r in R)
    agree_l2 = sum(r["answers"]["pruned_layer2"]["pred"] == r["answers"]["full"]["pred"] for r in R)
    print(f"  pool n={len(R)}: late-layer pruning preserves the full-image answer on "
          f"{agree_late}/{len(R)}, layer-2 on {agree_l2}/{len(R)}")
    fig, axes = plt.subplots(len(items), 3, figsize=(6.6, 2.32*len(items)))
    if len(items) == 1:
        axes = axes[None, :]
    for rI, r in enumerate(items):
        gh, gw = r["grid"]
        img = Image.open(f"{CR}/{r['qid'].replace('/','_')}_full.png").convert("RGB")
        W, H = img.size
        a = r["answers"]
        cols = [("full image", None, a["full"], "k"),
                (f"keep 10% — layer 2", r["masks"]["layer2"], a["pruned_layer2"], BAD),
                (f"keep 10% — layers 16–26", r["masks"]["late"], a["pruned_late"], GOOD)]
        for cI, (t, mask, ans, col) in enumerate(cols):
            ax = axes[rI, cI]
            ax.imshow(img); ax.axis("off")
            if mask is not None:
                keep = np.zeros(gh*gw, bool); keep[np.asarray(mask)] = True
                dim = np.asarray(Image.fromarray((~keep.reshape(gh, gw)).astype(np.uint8)*255)
                                 .resize(img.size, Image.NEAREST))/255.0
                ax.imshow(np.dstack([np.zeros_like(dim)]*3 + [dim*0.72]),
                          extent=(0, W, H, 0))
            box(ax, r["gt_box_frac"], W, H, "#f1c40f", 1.8)
            ok = ans["pred"] == r["gold"]
            if rI == 0:
                ax.set_title(t, fontsize=7.6, color=col)
            ax.text(0.5, -0.11, f'answers {ans["pred"]}  ({"correct" if ok else "wrong"})',
                    transform=ax.transAxes, ha="center", fontsize=7,
                    color=GOOD if ok else BAD,
                    fontweight="bold" if cI > 0 else "normal")
        axes[rI, 0].text(-0.04, 0.5, f'gold {r["gold"]}', transform=axes[rI, 0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=7)
    fig.subplots_adjust(hspace=0.42)
    save(fig, "fig_prune_masks")
    print(f"fig_prune_masks ok ({len(items)} items, real masks + real answers)")


# ---------------------------------------------------------------- FIG: the actual crops
def fig_crops():
    items = R[:4]
    fig = plt.figure(figsize=(7.2, 4.6))
    g = gs.GridSpec(3, len(items), height_ratios=[1.35, 1, 1], hspace=0.40, wspace=0.13)
    for c, r in enumerate(items):
        img = Image.open(f"{CR}/{r['qid'].replace('/','_')}_full.png").convert("RGB")
        W, H = img.size
        ax = fig.add_subplot(g[0, c]); ax.imshow(img); ax.axis("off")
        box(ax, r["gt_box_frac"], W, H, "#f1c40f", 1.7)
        for key, col in [("argmax", BAD), ("head", GOOD)]:
            cx, cy = r[key]; Wn = 0.15
            x0 = min(max(cx-Wn/2, 0), 1-Wn); y0 = min(max(cy-Wn/2, 0), 1-Wn)
            ax.add_patch(Rectangle((x0*W, y0*H), Wn*W, Wn*H, fill=False, ec=col, lw=1.7))
        q = r["question"].split("(A)")[0].strip()
        ax.set_title("\n".join([q[:26], q[26:50] + ("…" if len(q) > 50 else "")]).strip(),
                     fontsize=6.0, linespacing=1.15)
        for rI, (nm, lab, col) in enumerate([("standard", "standard read-out", BAD),
                                             ("learned", "learned read-out", GOOD)]):
            ax = fig.add_subplot(g[rI+1, c])
            ax.imshow(Image.open(f"{CR}/{r['crops'][nm]}").convert("RGB")); ax.axis("off")
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(1.6)
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
            a = r["answers"][f"crop_{nm}"]
            ok = a["pred"] == r["gold"]
            ax.set_xlabel(f'{a["pred"]} ({"✓" if ok else "✗"}) p={a["conf"]:.2f}',
                          fontsize=6.8, color=GOOD if ok else BAD, labelpad=1.5)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7, color=col)
    save(fig, "fig_crops")
    print(f"fig_crops ok ({len(items)} items, actual crops + actual predictions)")


if __name__ == "__main__":
    for fn in (fig1, fig_prune_masks, fig_crops):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
