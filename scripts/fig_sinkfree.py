"""Attention with the serialisation sink removed, on the two policy example items.

Sink removal = subtract each layer's projection onto the ITEM-MEAN map m_l (the item-independent
component; §49/§50A), shift to positive and renormalise. The raw block mean is shown for contrast,
and the deployed DWA score map (signed depth filter) as the third column.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; W = 0.25
ITEMS = [("direct_attributes/71", "SINGLE-INSTANCE", "What is the color of the paraglider?"),
         ("relative_position/174", "CROSS-INSTANCE", "Is the slide on the left or right side of the life buoy?")]
maps = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl") if "attn" in json.loads(l)}
ridge = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_qwen3.jsonl")}
root = snapshot_download("craigwu/vstar_bench", repo_type="dataset"); ds = load_dataset("craigwu/vstar_bench")["test"]
lut = {f"{e['category']}/{e['question_id']}": e for e in ds}


def maps_on_grid(gh, gw):
    """per-layer item-mean map on a target grid: resize every item's map and average"""
    acc = None
    for r in maps.values():
        gh0, gw0 = r["grid"]
        A = np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(28)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        M = np.stack([np.array(Image.fromarray(A[l].reshape(gh0, gw0).astype(np.float32)).resize((gw, gh), Image.BILINEAR))
                      for l in range(28)]).reshape(28, -1)
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
        acc = M if acc is None else acc + M
    return acc / len(maps)


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W/2, cx + W/2, cy - W/2, cy + W/2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1-gx0)*(gy1-gy0), 1e-12)


fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.0))
for row, (qid, tag, question) in enumerate(ITEMS):
    ex = lut[qid]; img = Image.open(os.path.join(root, ex["image"])).convert("RGB"); iw, ih = img.size
    r = maps[qid]; gh, gw = r["grid"]
    A = np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(28)]); A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    m = maps_on_grid(gh, gw)
    den = np.maximum((m * m).sum(1), 1e-12); coef = (A * m).sum(1) / den
    Aab = A - coef[:, None] * m
    Aab = Aab - Aab.min(axis=1, keepdims=True) + 1e-9
    Aab = Aab / np.maximum(Aab.sum(1, keepdims=True), 1e-12)
    raw = A[16:27].mean(0).reshape(gh, gw)
    proj = Aab[16:27].mean(0).reshape(gh, gw)
    score = np.array(ridge[qid]["score"], float).reshape(gh, gw)
    side = json.load(open(os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json")).get("bbox") or []
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    ring = np.zeros((gh, gw), bool); ring[1:-1, 1:-1] = True
    for col, (M, title, cmap) in enumerate([(raw, "raw block mean (L16\u201326): the serialisation sink", "inferno"),
                                            (proj, "after removing the item-independent component", "inferno"),
                                            (score, "DWA score (signed depth filter)", "viridis")]):
        ax = axes[row, col]
        ax.imshow(img)
        Z = (M - M.min()) / max(M.max() - M.min(), 1e-9)
        heat = np.array(Image.fromarray((Z * 255).astype(np.uint8)).resize((iw, ih), Image.BICUBIC)) / 255.0
        ax.imshow(heat, cmap=cmap, alpha=0.55 * heat, vmin=0, vmax=1)
        j = int(np.argmax(np.where(ring, M, -1e9))); jy, jx = j // gw, j % gw
        ax.plot((jx + .5) / gw * iw, (jy + .5) / gh * ih, marker="x", ms=13, mew=3, color=BLUE, zorder=7)
        for b in side:
            ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GREEN, lw=2.4, zorder=6))
        c = cov((jx + .5) / gw, (jy + .5) / gh, r["gt_box_frac"])
        if col == 0:
            ax.set_ylabel(f"{tag}\n\"{question}\"", fontsize=11, fontweight="bold")
        ax.set_title(title + f"\narg-max coverage {c:.2f}", fontsize=10.5, color=(RED if col == 0 else (GREEN if col == 1 else BLUE)))
        ax.set_xticks([]); ax.set_yticks([])
fig.text(0.5, 0.015, "Removing the sink = subtracting each layer's projection onto the item-mean map (the component that does not depend on the image), "
         "then renormalising.\nThe border mass disappears; what is left is the region the question actually concerns. "
         "DWA reaches the same place by giving the sink-heavy late layers negative weight instead of subtracting a mean.",
         ha="center", fontsize=10.5)
plt.tight_layout(rect=[0, 0.05, 1, 1])
for e in ("pdf", "png"): plt.savefig(f"{D}/paper/figs/fig_sinkfree.{e}", dpi=160, bbox_inches="tight")
print("wrote paper/figs/fig_sinkfree.{pdf,png}")
