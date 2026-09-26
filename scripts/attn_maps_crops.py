"""Attention maps for every image in paper/figs/crops/ (Qwen3-VL-2B).

For each crop (named <category>_<id>_<variant>), the original V*Bench question for that item is used
as the prompt; the final prompt token's head-mean attention over image cells is captured at all 28
layers. Output per image: one PNG with the read-out-band mean (L17-21) overlaid on the image, plus a
strip of six single layers. Written to paper/figs/crops_attn/.
"""
import json, os, glob, re, importlib, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
from datasets import load_dataset
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = f"{D}/paper/figs/crops"; OUT = f"{D}/paper/figs/crops_attn"
os.makedirs(OUT, exist_ok=True)
STRIP = [4, 10, 16, 20, 24, 27]
plt.rcParams.update({"font.size": 7, "figure.dpi": 150, "savefig.bbox": "tight"})

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16,
                                                    device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct"); tok = pr.tokenizer
layers = model.model.language_model.layers; NL = len(layers)
itid = model.config.image_token_id
modname = type(layers[0].self_attn).__module__; QM = importlib.import_module(modname)


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
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


def fit_budget(img, target=300, refine=6, tol=0.10):
    W_, H_ = img.size; r0 = max(ntok(build(img, "x")), 1); sc = (target / r0) ** 0.5; best = None
    for _ in range(refine):
        cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC); r = ntok(build(cur, "x"))
        if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
        if r == 0 or abs(r - target) / target <= tol: break
        sc *= (target / max(r, 1)) ** 0.5
    return best[0]


root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
ds = load_dataset("craigwu/vstar_bench")["test"]
lut = {f"{e['category']}/{e['question_id']}": e["text"] for e in ds}

paths = sorted(glob.glob(f"{CROPS}/*.png"))
print(f"{len(paths)} images", flush=True)
for k, path in enumerate(paths):
    name = os.path.basename(path)[:-4]
    cat, qid, variant = name.rsplit("_", 2)
    q = lut.get(f"{cat}/{qid}", "Describe this image.")
    img = Image.open(path).convert("RGB")
    src = fit_budget(img, 300); inp = build(src, q).to(model.device)
    if "image_grid_thw" not in inp:
        del inp; torch.cuda.empty_cache(); continue
    g = inp["image_grid_thw"].tolist()[0]; gh, gw = g[1] // 2, g[2] // 2
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    A = np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]); A = A[:, b:b + gh * gw]
    del inp; torch.cuda.empty_cache()
    if A.shape[1] != gh * gw: continue
    iw, ih = img.size

    def norm(M):
        M = np.asarray(M, float).reshape(gh, gw)
        M = M - M.min(); return M / max(np.percentile(M, 99.5), 1e-9)

    def overlay(ax, M, title=None):
        heat = np.array(Image.fromarray((np.clip(norm(M), 0, 1) * 255).astype(np.uint8)).resize((iw, ih), Image.BICUBIC)) / 255.0
        ax.imshow(img); ax.imshow(heat, cmap="turbo", alpha=0.75 * heat, vmin=0, vmax=1)
        ax.axis("off")
        if title: ax.set_title(title, fontsize=7)

    def heat(ax, M, title=None):
        ax.imshow(np.clip(norm(M), 0, 1), cmap="turbo", vmin=0, vmax=1, aspect="auto")
        ax.axis("off")
        if title: ax.set_title(title, fontsize=7)

    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.5, 1.0], hspace=0.12, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img); ax.axis("off"); ax.set_title(f"{cat}/{qid} — {variant}", fontsize=8)
    heat(fig.add_subplot(gs[0, 1:3]), A[17:22].mean(0), "band L17--21 mean")
    overlay(fig.add_subplot(gs[0, 3:6]), A[17:22].mean(0), "overlay")
    for j, l in enumerate(STRIP):
        heat(fig.add_subplot(gs[1, j]), A[l], f"L{l}")
    fig.text(0.5, -0.01, q.replace("\n", " ")[:170], ha="center", fontsize=7, color="#333333")
    fig.savefig(f"{OUT}/{name}_attn.png"); plt.close(fig)
    if (k + 1) % 10 == 0: print(f"  [{k+1}/{len(paths)}]", flush=True)
print(f"Done -> {OUT}")
