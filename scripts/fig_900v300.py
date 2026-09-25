"""900 vs 300 visual tokens on a real dataset sample (V*Bench direct_attributes/109).
Panels: the same image resized to 300 and to 900 tokens, with the model's actual cell grid drawn,
the ground-truth box marked, and an inset of exactly the pixels the model gets for the target.
Plus the per-layer allocation strip: bar / DWA / AVR."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
GREEN = "#1a7f37"; RED = "#cf222e"; BLUE = "#0969da"; GREY = "#57606a"; YEL = "#fff8c5"
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
from transformers import AutoProcessor
pr = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
itid = pr.tokenizer.convert_tokens_to_ids("<|image_pad|>") if "<|image_pad|>" in pr.tokenizer.get_vocab() else None

d = json.load(open("/tmp/opencode/item109.json")); gt = d["gt"]
img0 = Image.open(d["img"]).convert("RGB")


def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
def build(i): return pr(images=i, text=chat("x"), return_tensors="pt")
def nt(i):
    g = build(i)["image_grid_thw"].tolist()[0]
    return int(g[1] * g[2] // 4), (g[1] // 2, g[2] // 2)
def fit(img, target, refine=8, tol=0.04):
    W_, H_ = img.size; sc = (target / max(nt(img)[0], 1)) ** 0.5; best = None
    for _ in range(refine):
        cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC)
        n_, g_ = nt(cur)
        if best is None or abs(n_ - target) < abs(best[1] - target): best = (cur, n_, g_)
        if n_ == 0 or abs(n_ - target) / target <= tol: break
        sc *= (target / max(n_, 1)) ** 0.5
    return best

fig = plt.figure(figsize=(16.5, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.10)
panels = []
for j, target in enumerate((300, 600, 900)):
    im, n_, (gh, gw) = fit(img0, target)
    iw, ih = im.size
    ax = fig.add_subplot(gs[0, j]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    # cell grid
    for c in range(1, gw): ax.axvline(c * iw / gw - 0.5, color="white", lw=0.6, alpha=0.85)
    for r in range(1, gh): ax.axhline(r * ih / gh - 0.5, color="white", lw=0.6, alpha=0.85)
    ax.add_patch(Rectangle((gt[0] * iw, gt[1] * ih), (gt[2] - gt[0]) * iw, (gt[3] - gt[1]) * ih,
                           fill=False, ec=GREEN, lw=2.4, zorder=5))
    # how many cells does the box touch?
    x0c, x1c = int(gt[0] * gw), min(int(np.ceil(gt[2] * gw)) - 1, gw - 1)
    y0c, y1c = int(gt[1] * gh), min(int(np.ceil(gt[3] * gh)) - 1, gh - 1)
    ncells = (x1c - x0c + 1) * (y1c - y0c + 1)
    ax.set_title(f"{n_} visual tokens  ({gh}x{gw} cells)\nthe tablecloth spans {ncells} cell{'s' if ncells>1 else ''}",
                 fontsize=11.5, fontweight="bold", color=(BLUE if target == 300 else (GREY if target == 600 else GREEN)))
    panels.append((im, gh, gw, ncells))

# insets: exactly the pixels the model gets for the target, upscaled x6 nearest
for j, (im, gh, gw, ncells) in enumerate(panels):
    iw, ih = im.size
    box = (int(gt[0] * iw), int(gt[1] * ih), int(np.ceil(gt[2] * iw)), int(np.ceil(gt[3] * ih)))
    box = (max(box[0] - 12, 0), max(box[1] - 12, 0), min(box[2] + 12, iw), min(box[3] + 12, ih))
    crop = im.crop(box)
    ax = fig.add_subplot(gs[1, j]); ax.imshow(crop.resize((crop.size[0] * 6, crop.size[1] * 6), Image.NEAREST))
    ax.set_xticks([]); ax.set_yticks([])
    tgt_px = max(1, int(round((gt[2] - gt[0]) * im.size[0])))
    ax.set_title(f"target region as encoded\n{tgt_px} px wide, inside a {im.size[0]}x{im.size[1]} image", fontsize=10)

fig.subplots_adjust(bottom=0.16)
fig.text(0.5, 0.065,
         "300 tokens: the tablecloth spans 2 cells \u2014 sub-token evidence, the model cannot read the colour.   600 (the equal-compute bar): still 2 cells.\n"
         "900 tokens: 4 cells, readable \u2014 but 900 x 28 layers is 50% over budget. AVR buys it by dropping the last 11 layers' tokens "
         "(900x17 + 90x11 = 16,290 = 97% of the bar); DWA instead spends the same 600 tokens as 300 localise + 300 crop at 16x density.",
         ha="center", fontsize=10.5)
for e in ("pdf", "png"): plt.savefig(f"{D}/paper/figs/fig_900v300.{e}", dpi=165, bbox_inches="tight")
print("grids:", [(p[1], p[2], p[3]) for p in panels])
print("wrote paper/figs/fig_900v300.{pdf,png}")
