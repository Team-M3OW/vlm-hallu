"""
Phase 229: window-size ablation for DWA -- W=0.25 vs W=0.5 vs DYNAMIC W, equal answer budget (300 tokens).

Dynamic W: the ridge score's top-5% cells (ring-masked) form a bounding box; the window is sized by
max(bbox width, height) in normalised coordinates, clamped to [0.15, 0.50], centred on the ridge
argmax. Concentrated evidence -> tight window; dispersed -> wide. Label-free at test time.

Both arms get exactly 300 visual tokens for the answer pass; the crop trades 16x density for
1/16 coverage, the full image does the opposite. The localise pass is extra for the crops (it is
not charged here -- this isolates the operation, not the method's cost).

Arms: uniform@300 (full image, 300 tok), crop25@300 (W=0.25 at the DWA cell),
      crop50@300 (W=0.5 at the DWA cell), rand25@300 (floor).
Usage: phase228_equal300.py <model-key> <vstar|hr4k|textvqa|docvqa|cvbench> [n]
"""
import json, os, sys, time, importlib, numpy as np, torch
from PIL import Image
for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"): os.environ.pop(v, None)
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
from transformers import AutoProcessor, AutoModelForImageTextToText
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
MODELS = {"qwen3_2b": ("Qwen/Qwen3-VL-2B-Instruct", "qwen3"), "qwen2_7b": ("Qwen/Qwen2-VL-7B-Instruct", "qwen2")}
MK, BK = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL_ID, WT = MODELS[MK]; OUT = f"{D}/data/phase229_{MK}_{BK}.jsonl"
Image.MAX_IMAGE_PIXELS = None
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
layers = model.model.language_model.layers; NL = len(layers); P = round(0.57 * NL); BLK = (P, NL - 1)
WV = np.load(f"{D}/data/fig_ridge_w_{WT}.npy")
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


def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0] == itid).sum())
def fit_budget(img, target, refine=6, tol=0.10):
    W_, H_ = img.size; sc = (target / max(ntok(build(img, "x")), 1)) ** 0.5; best = None
    for _ in range(refine):
        cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC); r = ntok(build(cur, "x"))
        if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
        if r == 0 or abs(r - target) / target <= tol: break
        sc *= (target / max(r, 1)) ** 0.5
    return best
def feats(A, gh, gw):
    a = A / np.maximum(A.sum(1, keepdims=True), 1e-12); nc = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    LA = np.log(a + 1e-12).T; R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo]
def localise(img, q):
    src = fit_budget(img, 300)[0]; inp = build(src, q).to(model.device)
    if "image_grid_thw" not in inp: return None
    g = inp["image_grid_thw"].tolist()[0]; gh, gw = g[1] // 2, g[2] // 2
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    A = np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]); A = A[:, b:b + gh * gw]
    del inp; torch.cuda.empty_cache()
    return (A, gh, gw) if A.shape[1] == gh * gw else None
def crop(img, cx, cy, w):
    iw, ih = img.size; x0 = min(max(0, (cx - w / 2) * iw), iw - w * iw); y0 = min(max(0, (cy - w / 2) * ih), ih - w * ih)
    return img.crop((int(x0), int(y0), int(x0 + w * iw), int(y0 + w * ih)))
def answer(img, q, kind, target):
    src = fit_budget(img, target)[0]; inp = build(src, q).to(model.device)
    p, t = B.score_item(model, pr, tok, inp, kind)
    nt = ntok(inp); del inp; torch.cuda.empty_cache(); return p, t, nt


rng = np.random.default_rng(228)
items = B.load(BK, N or None)
print(f"{MK}/{BK}: {len(items)} items", flush=True)
done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
n = 0; t0 = time.time()
with open(OUT, "a") as f:
    for qid, img, q, gold, stratum, kind in items:
        if qid in done: continue
        img = img.convert("RGB")
        L = localise(img, q)
        if L is None: continue
        A, gh, gw = L; X = feats(A, gh, gw)
        if X.shape[1] != len(WV) - 1: continue
        mu, sd = X.mean(0), X.std(0) + 1e-9
        s = np.c_[(X - mu) / sd, np.ones(len(X))] @ WV
        rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
        j = int(np.argmax(np.where(rm, s, -1e9)))
        cx, cy = (j % gw + .5) / gw, (j // gw + .5) / gh
        rj = int(rng.integers(len(rm))); rx, ry = (rj % gw + .5) / gw, (rj // gw + .5) / gh
        # dynamic window from the top-5% of the ridge score
        srm = np.where(rm, s, -1e9)
        k = max(4, int(round(0.05 * len(srm))))
        top = np.argsort(-srm)[:k]
        rows_ = top // gw; cols_ = top % gw
        bbw = (cols_.max() - cols_.min() + 1) / gw
        bbh = (rows_.max() - rows_.min() + 1) / gh
        Wdyn = float(min(0.50, max(0.15, max(bbw, bbh))))
        rec = {"qid": qid, "gold": gold, "stratum": stratum, "kind": kind, "probs": {}, "preds": {}, "tokens": {}, "cell": [cx, cy], "W_dyn": Wdyn}
        for nm, im in (("uniform@300", img), ("crop25@300", crop(img, cx, cy, 0.25)),
                       ("crop50@300", crop(img, cx, cy, 0.50)), ("dynW@300", crop(img, cx, cy, Wdyn)),
                       ("rand25@300", crop(img, rx, ry, 0.25))):
            p, t, nt = answer(im, q, kind, 300)
            if p is not None: rec["probs"][nm] = p
            if t is not None: rec["preds"][nm] = t
            rec["tokens"][nm] = nt
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 25 == 0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item", flush=True)
print(f"Done -> {OUT}", flush=True)
