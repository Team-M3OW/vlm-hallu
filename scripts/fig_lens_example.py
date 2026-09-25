"""Qualitative logit-lens figure on a real V*Bench item (direct_attributes/96, strict on both
models: localised, uniform wrong at every layer, oracle correct from L22).

Top: the image as the uniform pass sees it (full, GT boxed) and the oracle crop. Middle: the decoded
probability of the correct option per layer, both arms. Bottom: the decoded top-1 option letter at
every layer (green = correct) -- the answer never appears under uniform and switches on at L22 under
the crop. All numbers from cached lens runs (phase79/phase84); the image is the dataset's."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; F = f"{D}/paper/figs"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BAD, GOOD, HL, GREY = "#c0392b", "#1a6b54", "#2c5aa0", "#95a5a6"
QID = "direct_attributes/96"
LET = "ABCD"

q3 = {r["question_id_full"]: r for r in (json.loads(l) for l in open(f"{D}/data/phase79_lens_mechanism.jsonl"))}
q2 = {r["question_id_full"]: r for r in (json.loads(l) for l in open(f"{D}/data/phase84_lens_qwen2vl.jsonl"))}
maps = {r["question_id_full"]: r for r in (json.loads(l) for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl"))}

root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
ds = load_dataset("craigwu/vstar_bench")["test"]
ex = {f"{e['category']}/{e['question_id']}": e for e in ds}[QID]
img = Image.open(os.path.join(root, ex["image"])).convert("RGB")
side = json.load(open(os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json")).get("bbox") or []
iw, ih = img.size
box = side[0]
sq = max(box[2], box[3])
cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
x0 = int(max(0, min(cx - sq / 2, iw - sq))); y0 = int(max(0, min(cy - sq / 2, ih - sq)))
crop = img.crop((x0, y0, int(x0 + sq), int(y0 + sq)))
g = maps[QID]["gt_box_frac"]; tok_target = (g[2] - g[0]) * (g[3] - g[1]) * 300

fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.1, 0.55], hspace=0.34, wspace=0.14)

ax = fig.add_subplot(gs[0, 0]); ax.imshow(img)
for b in side:
    ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GOOD, lw=2.0))
    ax.add_patch(Circle((b[0] + b[2] / 2, b[1] + b[3] / 2), max(38, 4 * max(b[2], b[3])), fill=False, ec=GOOD, lw=1.4))
ax.annotate(f"{side[0][2]}$\\times${side[0][3]} px target", xy=(box[0] + box[2] / 2, box[1] + box[3] / 2),
            xytext=(box[0] + 330, box[1] - 190), color=GOOD, fontsize=6.5,
            arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.0))
ax.set_title(f"uniform encoding: full image, 300 tokens\nGT box = {tok_target:.2f} merged tokens", fontsize=8)
ax.axis("off")
ax = fig.add_subplot(gs[0, 1]); ax.imshow(crop)
ax.set_title("oracle crop: GT region, same 300 tokens\n(12$\\times$11 px source, upscaled)", fontsize=8); ax.axis("off")

for j, (tag, data) in enumerate((("Qwen3-VL-2B", q3), ("Qwen2-VL-7B", q2))):
    r = data[QID]; lab = r["label"]
    ax = fig.add_subplot(gs[1, j])
    for arm, c, lab_ in (("uniform@300", BAD, "uniform"), ("oracle@0.15", GOOD, "oracle crop")):
        p = np.array([row[lab] for row in r["traj"][arm]]) * 100
        ax.plot(np.arange(28), p, "-o", ms=2.2, lw=1.4, color=c, label=lab_)
    ax.axvline(22, ls=":", lw=1.0, color="k")
    ax.axhline(25, ls="--", lw=0.8, color=GREY); ax.text(0.3, 27, "chance", fontsize=6, color=GREY)
    ax.set_ylim(0, 105); ax.set_xlim(-0.5, 27.5)
    ax.set_xlabel("layer"); ax.set_title(f"{tag}  (correct option: {LET[lab]})", fontsize=8)
    if j == 0: ax.set_ylabel("decoded P(correct) %")
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[2, j]); ax.set_xlim(-0.5, 27.5); ax.set_ylim(-0.6, 1.6); ax.axis("off")
    for row_i, (arm, nm) in enumerate((("uniform@300", "uniform"), ("oracle@0.15", "oracle"))):
        for l in range(28):
            top = int(np.argmax(r["traj"][arm][l]))
            ax.text(l, 1 - row_i, LET[top], ha="center", va="center", fontsize=6.2,
                    color=(GOOD if top == lab else BAD), fontweight="bold" if top == lab else "normal")
        ax.text(-1.2, 1 - row_i, nm, ha="right", va="center", fontsize=6.2, color="#333333")
    ax.text(28.4, 0.5, "decoded\ntop-1", ha="left", va="center", fontsize=6.2, color="#333333")

fig.text(0.5, 1.02, "Q: What is the color of the bottle cap?  (A) blue  (B) white  (C) red  (D) orange", ha="center", fontsize=8.5)
fig.text(0.5, -0.02, "the correct option is decoded at no layer under the uniform encoding; under the crop it appears at L22",
         ha="center", fontsize=6.8, color="#333333")
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_lens_example.{e}")
print("saved", f"{F}/fig_lens_example.pdf")
