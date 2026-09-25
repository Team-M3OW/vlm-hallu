"""
Phase 233: does the head-augmented ridge CONVERT? (deployed mean+ranks+geo vs AH+geo, end task)

Arms, all crops W=0.25 at 300 tokens, bar = uniform@600:
  uniform@600   the equal-compute bar
  block@300     published read-out (block-mean argmax)
  dwa_mean@300  deployed ridge (63 features: log-A, ranks, geo) -- fig_ridge_w_*.npy
  dwa_head@300  head-augmented ridge (91: log-A, log head-max, head-std, geo) -- fig_ridge_w_head_*.npz
Same items. Cells recorded so the two read-outs can be compared directly.
Usage: phase233_endtask_head.py <qwen3_2b|qwen2_7b> <vstar|textvqa|docvqa> [n]
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
MODEL_ID, WT = MODELS[MK]; OUT = f"{D}/data/phase233_{MK}_{BK}.jsonl"; W = 0.25
Image.MAX_IMAGE_PIXELS = None
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
layers = model.model.language_model.layers; NL = len(layers); P = round(0.57 * NL); BLK = (P, NL - 1)
WV = np.load(f"{D}/data/fig_ridge_w_{WT}.npy")
WH = np.load(f"{D}/data/fig_ridge_w_head_{WT}.npz"); wH, muH, sdH = WH["w"], WH["mu"], WH["sd"]
modname = type(layers[0].self_attn).__module__; QM = importlib.import_module(modname)


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        if getattr(module, "_capture", False):
            module._lastrow = w[0, :, -1, :].float().detach().cpu().numpy()      # (H, seq)
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
def feats_mean(A, gh, gw):
    a = A / np.maximum(A.sum(1, keepdims=True), 1e-12); nc = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    LA = np.log(a + 1e-12).T; R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo]
def feats_head(Ph, gh, gw):
    Al = Ph.mean(1)                                       # (NL, cells): head-mean per layer, as deployed
    a = Al / np.maximum(Al.sum(1, keepdims=True), 1e-12); nc = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    Ps = Ph / np.maximum(Ph.sum(1, keepdims=True), 1e-9)  # per-cell head distribution (NL,H,cells)
    LA = np.log(a + 1e-12).T; HM = np.log(Ps.max(1) + 1e-12).T; HS = Ps.std(1).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, HM, HS, geo]
def localise(img, q):
    src = fit_budget(img, 300)[0]; inp = build(src, q).to(model.device)
    if "image_grid_thw" not in inp: return None
    g = inp["image_grid_thw"].tolist()[0]; gh, gw = g[1] // 2, g[2] // 2
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    Ph = np.stack([l.self_attn._lastrow for l in layers])          # (NL, H, seq)
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]); Ph = Ph[:, :, b:b + gh * gw]                  # (NL, H, cells)
    del inp; torch.cuda.empty_cache()
    return (Ph, gh, gw) if Ph.shape[2] == gh * gw else None
def crop(img, cx, cy, w):
    iw, ih = img.size; x0 = min(max(0, (cx - w / 2) * iw), iw - w * iw); y0 = min(max(0, (cy - w / 2) * ih), ih - w * ih)
    return img.crop((int(x0), int(y0), int(x0 + w * iw), int(y0 + w * ih)))
def answer(img, q, kind, target=300):
    src = fit_budget(img, target)[0]; inp = build(src, q).to(model.device)
    p, t = B.score_item(model, pr, tok, inp, kind); nt = ntok(inp); del inp; torch.cuda.empty_cache(); return p, t, nt


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
        Ph, gh, gw = L
        Al = Ph.mean(1)
        Xm = feats_mean(Al, gh, gw); Xh = feats_head(Ph, gh, gw)
        if Xm.shape[1] != len(WV) - 1 or Xh.shape[1] != len(wH) - 1: continue
        rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
        sm = np.c_[(Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9), np.ones(len(Xm))] @ WV
        sh = (Xh - muH) / sdH @ wH[:-1] + wH[-1]
        dep = Al[BLK[0]:BLK[1]].mean(0)
        jm = int(np.argmax(np.where(rm, sm, -1e9))); jh = int(np.argmax(np.where(rm, sh, -1e9)))
        jb = int(np.argmax(np.where(rm, dep, -1e9)))
        rec = {"qid": qid, "gold": gold, "stratum": stratum, "kind": kind, "probs": {}, "preds": {}, "tokens": {},
               "cell_mean": [jm // gw, jm % gw], "cell_head": [jh // gw, jh % gw], "cell_block": [jb // gw, jb % gw]}
        p, t, nt = answer(img, q, kind, 600)
        if p is not None: rec["probs"]["uniform@600"] = p
        if t is not None: rec["preds"]["uniform@600"] = t
        rec["tokens"]["uniform@600"] = nt
        for nm, j in (("block@300", jb), ("dwa_mean@300", jm), ("dwa_head@300", jh)):
            p, t, nt = answer(crop(img, (j % gw + .5) / gw, (j // gw + .5) / gh, W), q, kind, 300)
            if p is not None: rec["probs"][nm] = p
            if t is not None: rec["preds"][nm] = t
            rec["tokens"][nm] = nt
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 25 == 0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item", flush=True)
print(f"Done -> {OUT}", flush=True)
