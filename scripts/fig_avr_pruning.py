"""AVR inference phases on one V*Bench item, visualised.

1. encode @900 tokens; 2. read the final prompt token's attention at L12-16; 3. rank tokens and keep
the top 10% (90); 4. from L17 only those 90 keys remain accessible (additive -1e4 bias; tensors keep
their length); answer with no crop. Kept-cell coverage of the ground-truth box is compared with the
10% a random keep-set would give. Renders paper/figs/fig_avr_pruning.{pdf,png}."""
import json, os, importlib, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"; F = f"{D}/paper/figs"
plt.rcParams.update({"font.size": 8, "figure.dpi": 220, "savefig.bbox": "tight"})
GREEN, RED, GREY = "#00c000", "#c0392b", "#57606a"
import sys
QID = sys.argv[1] if len(sys.argv) > 1 else "direct_attributes/32"
NL = 28; P = 16; KEEP = 0.10; LET = "ABCD"

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16,
                                                    device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct"); tok = pr.tokenizer
layers = model.model.language_model.layers
itid = model.config.image_token_id
QM = importlib.import_module(type(layers[0].self_attn).__module__)


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
        b = getattr(module, "_prune_bias", None)
        if b is not None and b.shape[-1] == w.shape[-1]: w = w + b.to(w.dtype).view(1, 1, 1, -1)
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        if getattr(module, "_capture", False):
            module._lastrow = w[0, :, -1, :].float().mean(0).detach().cpu().numpy()
        return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
    return patched
QM.eager_attention_forward = make_patched(QM)


def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                                           tokenize=False, add_generation_prompt=True)
def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0] == itid).sum())


def fit_budget(img, target=900, refine=6, tol=0.06):
    W_, H_ = img.size; r0 = max(ntok(build(img, "x")), 1); sc = (target / r0) ** 0.5; best = None
    for _ in range(refine):
        cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC); r = ntok(build(cur, "x"))
        if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
        if r == 0 or abs(r - target) / target <= tol: break
        sc *= (target / max(r, 1)) ** 0.5
    return best[0]


root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
ds = load_dataset("craigwu/vstar_bench")["test"]
ex = {f"{e['category']}/{e['question_id']}": e for e in ds}[QID]
img0 = Image.open(os.path.join(root, ex["image"])).convert("RGB")
side = json.load(open(os.path.splitext(os.path.join(root, ex["image"]))[0] + ".json")).get("bbox") or []
src = fit_budget(img0, 900); inp = build(src, ex["text"]).to(model.device)
g = inp["image_grid_thw"].tolist()[0]; gh, gw = g[1] // 2, g[2] // 2; nc = gh * gw
pos = (inp["input_ids"][0] == itid).nonzero().flatten(); base = int(pos[0])

# phase 2: collect attention at L12-16
for l in layers: l.self_attn._capture = True
with torch.no_grad(): base_logits = model(**inp).logits[0, -1].float()
A = np.stack([l.self_attn._lastrow for l in layers])[:, base:base + nc]
for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
score = A[P - 4:P + 1].mean(0)

# phase 3: keep top 10%
keep = max(1, int(round(KEEP * nc)))
keep_idx = np.argsort(-score)[:keep]

# phase 4: pruned pass from L17
drop = [base + i for i in range(nc) if i not in set(keep_idx.tolist())]
bias = torch.zeros(inp["input_ids"].shape[1], device=model.device)
bias[torch.as_tensor(drop, device=model.device)] = -1e4
for li in range(P + 1, NL): layers[li].self_attn._prune_bias = bias
letters = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
with torch.no_grad(): pruned = model(**inp).logits[0, -1].float()
for li in range(P + 1, NL):
    if hasattr(layers[li].self_attn, "_prune_bias"): del layers[li].self_attn._prune_bias
pb = int(np.argmax([float(pruned[i]) for i in letters]))
bb = int(np.argmax([float(base_logits[i]) for i in letters]))

# coverage metrics (grid cells whose centre falls in the GT box)
iw, ih = src.size; yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
W0, H0 = img0.size
gx0 = min(b[0] for b in side) / W0; gy0 = min(b[1] for b in side) / H0
gx1 = max(b[0] + b[2] for b in side) / W0; gy1 = max(b[1] + b[3] for b in side) / H0
inside = (fx >= gx0) & (fx <= gx1) & (fy >= gy0) & (fy <= gy1)
keep_in = inside[keep_idx].mean() * 100
gt_cells = inside.sum()
gt_cov = len(set(keep_idx.tolist()) & set(np.where(inside)[0].tolist())) / max(gt_cells, 1) * 100
import benchmarks as B
GOLD = {q: g for q, _, _, g, _, _ in B.load("vstar")}
gold = GOLD[QID]

fig = plt.figure(figsize=(13.0, 3.9))
gs = fig.add_gridspec(1, 4, wspace=0.05)
im = src

ax = fig.add_subplot(gs[0, 0]); ax.imshow(im)
for bb_ in side: ax.add_patch(Rectangle((bb_[0] * iw / img0.size[0], bb_[1] * ih / img0.size[1]), bb_[2] * iw / img0.size[0], bb_[3] * ih / img0.size[1], fill=False, ec=GREEN, lw=1.2))
ax.set_title("1. encode @900 tokens", fontsize=8.5); ax.axis("off")

ax = fig.add_subplot(gs[0, 1]); M = (score - score.min()) / max(np.percentile(score, 99.5) - score.min(), 1e-9)
heat = np.array(Image.fromarray((np.clip(M, 0, 1) * 255).astype(np.uint8).reshape(gh, gw)).resize((iw, ih), Image.BICUBIC)) / 255.0
ax.imshow(im); ax.imshow(heat, cmap="turbo", alpha=0.6 * np.clip(heat, 0, 1), vmin=0, vmax=1)
for bb_ in side: ax.add_patch(Rectangle((bb_[0] * iw / img0.size[0], bb_[1] * ih / img0.size[1]), bb_[2] * iw / img0.size[0], bb_[3] * ih / img0.size[1], fill=False, ec=GREEN, lw=1.2))
ax.set_title("2. attention L12–16, final prompt token", fontsize=8.5); ax.axis("off")

ax = fig.add_subplot(gs[0, 2]); ax.imshow(im)
keep_mask = np.zeros((gh, gw), bool); keep_mask.ravel()[keep_idx] = True
cell_w, cell_h = iw / gw, ih / gh
for iy in range(gh):
    for ix in range(gw):
        if keep_mask[iy, ix]:
            ax.add_patch(Rectangle((ix * cell_w, iy * cell_h), cell_w, cell_h, fc=GREEN, ec="none", alpha=0.55))
for bb_ in side: ax.add_patch(Rectangle((bb_[0] * iw / img0.size[0], bb_[1] * ih / img0.size[1]), bb_[2] * iw / img0.size[0], bb_[3] * ih / img0.size[1], fill=False, ec="k", lw=1.2))
ax.set_title(f"3. keep top 10% ({keep}/{nc} tokens)", fontsize=8.5); ax.axis("off")
ax.text(0.5, -0.055, f"{gt_cov:.0f}% of GT cells kept (chance 10%, {gt_cov/10:.1f}$\\times$)",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=GREY)

ax = fig.add_subplot(gs[0, 3])
keep_full = np.array(Image.fromarray(keep_mask.astype(np.uint8) * 255).resize((iw, ih), Image.NEAREST)) > 127
view = np.array(im).copy(); view[~keep_full] //= 4
ax.imshow(view)
ax.set_title("4. L17+: only kept keys remain", fontsize=8.5); ax.axis("off")
col = GREEN if pb == gold else RED
ax.text(0.5, -0.055, f"AVR answer: {LET[pb]} {'✓' if pb == gold else '×'}   (uncropped: {LET[bb]} {'✓' if bb == gold else '×'})",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=col)
fig.text(0.5, -0.10, "Q: " + ex["text"].split("\n")[0] + "   " + " ".join(ex["text"].split("\n")[1:5]), ha="center", fontsize=8)
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_avr_pruning.{e}")
print(f"kept_in {keep_in:.1f}%  gt_cov {gt_cov:.1f}%  base {LET[bb]} pruned {LET[pb]} gold {LET[gold]}")
print("saved fig_avr_pruning")
