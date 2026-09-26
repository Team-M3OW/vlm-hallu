"""Inference-side visualisation of the depth boundary on real V*Bench items.

Top: one sample's attention map at selected layers (the map character changes at the boundary).
Bottom: per-item single-layer masking KL for all 40 calibrated items (thin), the mean (bold), and the
prefix/suffix schedules, both models, with the boundary marked. All numbers from phase176c; the maps
from phase30c/phase74; the image is the dataset's.
Renders paper/figs/fig_boundary_samples.{pdf,png}."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; F = f"{D}/paper/figs"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BAD, GOOD, HL, GREY = "#c0392b", "#1a6b54", "#2c5aa0", "#95a5a6"
SAMPLE = "direct_attributes/35"
PRE = [2, 6, 10, 14]; POST = [16, 20, 24, 26]
BOUND = 16

runs = {"Qwen3-VL-2B": (json.load(open(f"{D}/data/phase176c_transport_qwen3.json")),
                        {r["question_id_full"]: r for r in (json.loads(l) for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl"))}),
        "Qwen2-VL-7B": (json.load(open(f"{D}/data/phase176c_transport_qwen2.json")),
                        {r["question_id_full"]: r for r in (json.loads(l) for l in open(f"{D}/data/phase74_Qwen2_VL_7B_Instruct.jsonl"))})}

root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
ds = load_dataset("craigwu/vstar_bench")["test"]
ex = {f"{e['category']}/{e['question_id']}": e for e in ds}[SAMPLE]
img = Image.open(os.path.join(root, ex["image"])).convert("RGB"); iw, ih = img.size
side = json.load(open(os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json")).get("bbox") or []
maps3 = runs["Qwen3-VL-2B"][1][SAMPLE]

fig = plt.figure(figsize=(15.5, 9.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.26, wspace=0.16)

# ---------------- top: the sample's maps, pre-boundary above / post-boundary below ----------------
top = gs[0].subgridspec(2, 4, hspace=0.13, wspace=0.05)
axs = {}
for i, l in enumerate(PRE + POST):
    ax = fig.add_subplot(top[i // 4, i % 4]); axs[l] = ax
    A = np.asarray(maps3["attn"][f"L{l}"], float)
    M = (A - A.min()) / max(A.max() - A.min(), 1e-9)
    heat = np.array(Image.fromarray((M * 255).astype(np.uint8)).resize((iw, ih), Image.BICUBIC)) / 255.0
    heat = np.where(heat < 0.18, 0.0, heat)
    ax.imshow(img); ax.imshow(heat, cmap="inferno", alpha=0.85 * heat, vmin=0, vmax=1)
    for b in side: ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GOOD, lw=2.0))
    if l < BOUND: ax.set_title(f"L{l}", fontsize=9, color=HL)
    else: ax.text(0.5, -0.05, f"L{l}", transform=ax.transAxes, ha="center", fontsize=9, color=BAD)
    ax.axis("off")
p_pre = axs[14].get_position(); p_post = axs[16].get_position()
ysep = (p_pre.y0 + p_post.y1) / 2
fig.add_artist(plt.Line2D([p_pre.x0, p_post.x1], [ysep, ysep], color=BAD, lw=1.8, ls="--", transform=fig.transFigure))
fig.text(p_pre.x0, ysep + 0.014, "boundary L16", fontsize=8.5, color=BAD, va="bottom", ha="left")
fig.text(0.5, p_pre.y1 + 0.03, f"attention map by read-out depth, one real item ({SAMPLE.replace('_', ' ')})",
         ha="center", fontsize=10, fontweight="bold", color="#333333")

# ---------------- bottom: per-item masking curves ----------------
for j, (name, (rows, _)) in enumerate(runs.items()):
    ax = fig.add_subplot(gs[1, j])
    K = np.array([r["kl"] for r in rows])
    for i, r in enumerate(rows):
        c = "k" if r["question_id_full"] == SAMPLE and name.startswith("Qwen3") else GREY
        ax.plot(np.arange(28), np.array(r["kl"]) + 1e-4, lw=0.9 if c == "k" else 0.6,
                color=c, alpha=0.9 if c == "k" else 0.35, zorder=3 if c == "k" else 1)
    ax.plot(np.arange(28), K.mean(0) + 1e-4, lw=2.0, color=BAD, label="single-layer mean", zorder=4)
    ks = sorted(rows[0]["suffix"], key=int)
    ax.plot([int(k) for k in ks], [np.mean([r["suffix"][k][0] for r in rows]) + 1e-4 for k in ks],
            "-o", ms=3, lw=1.6, color=HL, label="suffix mask")
    ax.plot([int(k) for k in ks], [np.mean([r["prefix"][k][0] for r in rows]) + 1e-4 for k in ks],
            "-s", ms=3, lw=1.4, color=GOOD, label="prefix mask")
    ax.axvline(BOUND, ls=":", lw=1.2, color="k")
    ax.text(BOUND + 0.3, 1.2, "L16 ($0.57L$)", fontsize=7, color="k")
    ax.set_yscale("log"); ax.set_ylim(5e-5, 4); ax.set_xlim(-0.5, 27.5)
    ax.set_xlabel("layer"); ax.set_title(f"{name} — masking one layer at a time ($n=40$)", fontsize=8.5)
    if j == 0: ax.set_ylabel("KL divergence (log)")
    ax.legend(fontsize=6.4, frameon=False, loc="upper right", ncol=3)
fig.text(0.5, -0.02, "masking a single layer moves the answer only before the boundary; after it the output is untouched (black: the sample above)",
         ha="center", fontsize=7.2, color="#333333")
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_boundary_samples.{e}")
print("saved fig_boundary_samples")
