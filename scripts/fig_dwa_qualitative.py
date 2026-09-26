"""Qualitative DWA figure on a V*Bench item where the block-mean read-out fails at the grid edge.

Item: direct_attributes/32, "What is the color of the water bottle?" (A) black (B) red (C) white
(D) blue. GT box at the left edge (x ~ 2-3%); the block-mean (L16-26) arg-max lands in the last
column, so its crop answers A; the DWA ridge isolates the bottle and answers B correctly.

Panels: input + GT box | block-mean heatmap + wrong crop | DWA score heatmap + correct crop.
Renders paper/figs/fig_dwa_qualitative.{pdf,png}."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; F = f"{D}/paper/figs"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 220, "savefig.bbox": "tight"})
GREEN, RED, GREY = "#00c000", "#c0392b", "#57606a"
QID = "direct_attributes/32"; W = 0.25
LET = "ABCD"

m = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl")}[QID]
r = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/phase225_qwen3_2b_vstar.jsonl")}[QID]
Wv = np.load(f"{D}/data/fig_ridge_w_qwen3.npy")
NL = 28; BLK = (round(0.57 * NL), NL - 1)
gh, gw = m["grid"]; gt = m["gt_box_frac"]

A = np.stack([np.asarray(m["attn"][f"L{l}"], float) for l in range(NL)]); A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
dep = A[BLK[0]:BLK[1]].mean(0).reshape(gh, gw)

# DWA score (same feature builder as the deployed read-out)
a = A; nc = gh * gw
yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
LA = np.log(a + 1e-12).T; R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
pad = np.pad(dep, 1, mode="edge")
nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
            np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
            (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
X = np.c_[LA, R, geo]; mu, sd = X.mean(0), X.std(0) + 1e-9
score = (np.c_[(X - mu) / sd, np.ones(len(X))] @ Wv).reshape(gh, gw)
rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True
biy, bix = np.unravel_index(int(np.argmax(np.where(rm, dep, -1e9))), dep.shape)
diy, dix = np.unravel_index(int(np.argmax(np.where(rm, score, -1e9))), score.shape)

root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
ds = load_dataset("craigwu/vstar_bench")["test"]
ex = {f"{e['category']}/{e['question_id']}": e for e in ds}[QID]
img = Image.open(os.path.join(root, ex["image"])).convert("RGB"); iw, ih = img.size
side = json.load(open(os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json")).get("bbox") or []
crop_block = Image.open(f"{F}/crops/direct_attributes_32_standard.png").convert("RGB")
crop_dwa = Image.open(f"{F}/crops/direct_attributes_32_learned.png").convert("RGB")
bp = int(np.argmax(r["probs"]["block"])); dp = int(np.argmax(r["probs"]["dwa_t"])); gold = r["gold"]
OPTS = {"A": "black", "B": "red", "C": "white", "D": "blue"}


def overlay(ax, M, cmap, alpha, title, cell=None):
    M = np.asarray(M, float); M = (M - M.min()) / max(np.percentile(M, 99.5) - M.min(), 1e-9)
    heat = np.array(Image.fromarray((np.clip(M, 0, 1) * 255).astype(np.uint8)).resize((iw, ih), Image.BICUBIC)) / 255.0
    ax.imshow(img); ax.imshow(heat, cmap=cmap, alpha=alpha * np.clip(heat, 0, 1), vmin=0, vmax=1)
    for b in side: ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GREEN, lw=1.2))
    if cell is not None:
        cx, cy = (cell[1] + .5) / gw * iw, (cell[0] + .5) / gh * ih
        ax.add_patch(Rectangle((cx - W / 2 * iw, cy - W / 2 * ih), W * iw, W * ih, fill=False, ec=RED, lw=1.4, ls="--"))
    ax.set_title(title, fontsize=8.5); ax.axis("off")


fig = plt.figure(figsize=(12.6, 5.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.85, 1.0], hspace=0.16, wspace=0.06)

ax = fig.add_subplot(gs[0, 0]); ax.imshow(img)
for b in side: ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GREEN, lw=1.2))
ax.set_title("Input & Ground Truth", fontsize=8.5); ax.axis("off")

overlay(fig.add_subplot(gs[0, 1]), dep, "magma", 0.55, "Block-Mean (Layers 16--26)", (biy, bix))
overlay(fig.add_subplot(gs[0, 2]), score, "turbo", 0.55, "DWA Spatial Routing", (diy, dix))

ax = fig.add_subplot(gs[1, 0]); ax.imshow(img); ax.axis("off")
ax.text(0.5, -0.06, "target: 26$\\times$68 px at the left edge\n(0.25 merged tokens)", transform=ax.transAxes,
        ha="center", fontsize=7.5, color=GREY)

ax = fig.add_subplot(gs[1, 1]); ax.imshow(crop_block); ax.axis("off")
ax.set_title(f"crop at last column ({bix+1},{biy+1})", fontsize=7.5, color=RED)
ax.text(0.5, -0.06, f"answer: {LET[bp]} ({OPTS[LET[bp]]}) $\\times$   p={r['probs']['block'][bp]:.2f}",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=RED)

ax = fig.add_subplot(gs[1, 2]); ax.imshow(crop_dwa); ax.axis("off")
ax.set_title(f"crop at DWA arg-max ({dix+1},{diy+1})", fontsize=7.5, color="#0a7d00")
ax.text(0.5, -0.06, f"answer: {LET[dp]} ({OPTS[LET[dp]]}) $\\checkmark$   p={r['probs']['dwa_t'][dp]:.2f}",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#0a7d00")

fig.text(0.5, -0.03, "Q: What is the color of the water bottle?  (A) black  (B) red  (C) white  (D) blue",
         ha="center", fontsize=8)
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_dwa_qualitative.{e}")
print("saved fig_dwa_qualitative")
