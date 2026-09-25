"""The disp policy, illustrated on two real V*Bench items: a single-instance question (concentrated
map -> crop) and a cross-instance one (dispersed map -> full-resolution/AVR). Panels show the image,
the block-mean attention map, the ground-truth box(es), and where DWA would crop; the strip below
gives each arm's outcome and the decision rule."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; YEL = "#fff8c5"
W = 0.25
SINGLE, CROSS = "direct_attributes/71", "relative_position/174"
disp = {json.loads(l)["qid"]: json.loads(l)["disp"] for l in open(f"{D}/data/phase235_qwen3_2b_vstar.jsonl")}
tau = float(np.median(list(disp.values())))
grid225 = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/phase225_qwen3_2b_vstar.jsonl")}
ridge = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/fig_ridge_scores_qwen3.jsonl")}
maps = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase30c_attn_maps_all.jsonl") if "attn" in json.loads(l)}
root = snapshot_download("craigwu/vstar_bench", repo_type="dataset"); ds = load_dataset("craigwu/vstar_bench")["test"]
lut = {f"{e['category']}/{e['question_id']}": e for e in ds}

fig = plt.figure(figsize=(16.5, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.28, wspace=0.08)

for j, (qid, tag) in enumerate([(SINGLE, "SINGLE-INSTANCE"), (CROSS, "CROSS-INSTANCE")]):
    ex = lut[qid]; img = Image.open(os.path.join(root, ex["image"])).convert("RGB"); iw, ih = img.size
    r = maps[qid]; gh, gw = r["grid"]
    A = np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(28)]); A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    M = A[16:27].mean(0).reshape(gh, gw)
    M = (M - M.min()) / max(M.max() - M.min(), 1e-9)
    heat = np.array(Image.fromarray((M * 255).astype(np.uint8)).resize((iw, ih), Image.BICUBIC)) / 255.0
    ax = fig.add_subplot(gs[0, j]); ax.imshow(img)
    ax.imshow(heat, cmap="inferno", alpha=0.45 * heat, vmin=0, vmax=1)
    # individual GT boxes from the sidecar (one for single, two for the relation)
    side = json.load(open(os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json")).get("bbox") or []
    for b in side:
        ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec=GREEN, lw=2.6, zorder=6))
    # DWA crop window at the ridge cell
    cx, cy = ridge[qid]["ridge_cell"]; w_px, h_px = W * iw, W * ih
    bx = min(max(0, cx * iw - w_px / 2), iw - w_px); by = min(max(0, cy * ih - h_px / 2), ih - h_px)
    ax.add_patch(Rectangle((bx, by), w_px, h_px, fill=False, ec=(BLUE if tag.startswith("SINGLE") else RED), lw=2.6, ls="--", zorder=6))
    d = disp[qid]; act = "CROP" if d > tau else "FULL-RES (AVR)"
    col = BLUE if d > tau else GREEN
    ax.set_title(f"{tag}  —  {qid}\n\"{ex['text'].splitlines()[0]}\"\n"
                 f"disp = {d:.3f}   {'>' if d>tau else '<'}   median tau = {tau:.3f}   ->   {act}",
                 fontsize=11, fontweight="bold", color=col)
    ax.set_xticks([]); ax.set_yticks([])
    # outcome strip
    g = grid225[qid]
    ax2 = fig.add_subplot(gs[1, j]); ax2.axis("off")
    def mk(a):
        p = g["probs"][a]; return ("OK " if int(np.argmax(p)) == int(g["gold"]) else "X  "), ("#1a7f37" if int(np.argmax(p)) == int(g["gold"]) else "#cf222e")
    rows = [("bar (600, no crop)", "uniform@lo"), ("block mean (published read-out)", "block"),
            ("DWA (ridge crop W=0.25)", "dwa_t"), ("AVR (900 + prune)", "avr")]
    ax2.text(0.0, 0.92, "arm outcomes on this item", fontsize=10.5, fontweight="bold")
    for i, (lab, a) in enumerate(rows):
        if a not in g["probs"]: continue
        s, c = mk(a)
        ax2.text(0.02, 0.74 - i * 0.20, lab, fontsize=10)
        ax2.text(0.62, 0.74 - i * 0.20, s, fontsize=11, fontweight="bold", color=c)
    ax2.text(0.02, -0.08, "green box = ground truth;  dashed box = where DWA would crop;  heat = block-mean attention",
             fontsize=9.5, color=GREY)

fig.text(0.5, 0.015,
         "The policy reads only the concentration of the model's own attention: a tight map (disp above the median) means one region holds the "
         "evidence, so crop it at 16x density;\na dispersed map means the evidence is spread over the scene, so keep the whole image and buy resolution instead "
         "(the crop is what the dashed box would have shown).",
         ha="center", fontsize=10.5)
for e in ("pdf", "png"): plt.savefig(f"{D}/paper/figs/fig_disp_policy.{e}", dpi=165, bbox_inches="tight")
print(f"single {SINGLE} disp {disp[SINGLE]:.3f} | cross {CROSS} disp {disp[CROSS]:.3f} | tau {tau:.3f}")
print("wrote paper/figs/fig_disp_policy.{pdf,png}")
