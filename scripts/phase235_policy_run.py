"""
Phase 235 (GPU): PROSPECTIVE run of the map-only per-item policy (pick first, then run only the
chosen arm). No off-policy re-use of recorded outcomes.

Policy (pre-registered here, before the run):
  disp  = ring-masked top-1 share of the block-mean attention map from the localise pass @300
  tau   = median of disp over the target benchmark's items (label-free, transductive, NO tuning)
  action: disp > tau -> DWA  (ridge-placed crop W=0.25, answer @300)
          else        -> AVR  (encode @900, keep 10% at round(0.57*NL), no crop)
Arms recorded for the same items so the policy can be compared to always-DWA / always-AVR / bar.
Usage: phase235_policy_run.py <qwen3_2b|qwen2_7b> <vstar|hr4k|cvbench|textvqa|docvqa> [n]
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
MODEL_ID, WT = MODELS[MK]; OUT = f"{D}/data/phase235_{MK}_{BK}.jsonl"; W = 0.25
E_LO, E_HI, K = 600, 900, 0.10; Image.MAX_IMAGE_PIXELS = None
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
        b = getattr(module, "_prune_bias", None)
        if b is not None and b.shape[-1] == w.shape[-1]: w = w + b.to(w.dtype).view(1, 1, 1, -1)
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        if getattr(module, "_capture", False):
            module._lastrow = w[0, :, -1, :].float().detach().cpu().numpy()
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
def feats(a, gh, gw):
    nc = gh * gw; yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    LA = np.log(a + 1e-12).T; R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo], a[BLK[0]:BLK[1]].mean(0)
def clear():
    for l in layers:
        if hasattr(l.self_attn, "_prune_bias"): delattr(l.self_attn, "_prune_bias")
def run_arm(img, q, kind, target=None, prune=None, from_layer=None):
    clear()
    src = fit_budget(img, target)[0]
    inp = build(src, q).to(model.device); n = ntok(inp)
    if prune is not None:
        pos = (inp["input_ids"][0] == itid).nonzero().flatten()
        keep = set(prune(len(pos)))
        bias = torch.zeros(inp["input_ids"].shape[1], device=model.device)
        bias[[int(pos[i]) for i in range(len(pos)) if i not in keep]] = -1e4
        for l in layers[from_layer:]: l.self_attn._prune_bias = bias
    p, t = B.score_item(model, pr, tok, inp, kind)
    del inp; clear(); torch.cuda.empty_cache(); return p, t, n
def localise(img, q):
    src = fit_budget(img, 300)[0]; inp = build(src, q).to(model.device)
    if "image_grid_thw" not in inp: return None
    g = inp["image_grid_thw"].tolist()[0]; gh, gw = g[1] // 2, g[2] // 2
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    Ph = np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]); Ph = Ph[:, :, b:b + gh * gw]
    del inp; torch.cuda.empty_cache()
    return (Ph, gh, gw) if Ph.shape[2] == gh * gw else None
def crop(img, cx, cy, w=W):
    iw, ih = img.size; x0 = min(max(0, (cx - w / 2) * iw), iw - w * iw); y0 = min(max(0, (cy - w / 2) * ih), ih - w * ih)
    return img.crop((int(x0), int(y0), int(x0 + w * iw), int(y0 + w * ih)))


items = B.load(BK, N or None)
print(f"{MK}/{BK}: {len(items)} items", flush=True)
# pass 0: compute disp and the ridge cell for every item (the policy's inputs are label-free)
t0 = time.time(); info = []
for i, (qid, img, q, gold, stratum, kind) in enumerate(items):
    img = img.convert("RGB")
    L = localise(img, q)
    if L is None: info.append(None); continue
    Ph, gh, gw = L; A = Ph.mean(1); a = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
    disp = float(np.max(np.where(rm, a[BLK[0]:BLK[1]].mean(0), -1e9)) / max(a[BLK[0]:BLK[1]].mean(0).sum(), 1e-12))
    X, dep = feats(a, gh, gw)
    if X.shape[1] != len(WV) - 1: info.append(None); continue
    s = np.c_[(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones(len(X))] @ WV
    jh = int(np.argmax(np.where(rm, s, -1e9))); jb = int(np.argmax(np.where(rm, dep, -1e9)))
    rank = a[max(0, P - 4):P + 1].mean(0)
    info.append({"disp": disp, "cell": (jh // gw, jh % gw), "block": (jb // gw, jb % gw), "gh": gh, "gw": gw,
                 "rank": rank, "img": img, "q": q, "kind": kind, "gold": gold, "stratum": stratum, "qid": qid})
    if (i + 1) % 25 == 0: print(f"  localise [{i+1}] {(time.time()-t0)/(i+1):.1f}s/item", flush=True)
vals = [r["disp"] for r in info if r]
tau = float(np.median(vals)); print(f"  tau (median disp) = {tau:.4f}", flush=True)

done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
n = 0
with open(OUT, "a") as f:
    for r in info:
        if r is None or r["qid"] in done: continue
        img, q, kind = r["img"], r["q"], r["kind"]; gh_gw = None
        rec = {"qid": r["qid"], "stratum": r["stratum"], "kind": kind, "gold": r["gold"], "disp": r["disp"],
               "policy": "dwa" if r["disp"] > tau else "avr", "probs": {}, "preds": {}, "tokens": {}}
        # bar
        p, t, nt = run_arm(img, q, kind, E_LO); rec["probs"]["bar"] = p; rec["preds"]["bar"] = t; rec["tokens"]["bar"] = nt
        # DWA
        j = r["cell"]; gh, gw = r["gh"], r["gw"]
        p, t, nt = run_arm(crop(img, (j[1] + .5) / gw, (j[0] + .5) / gh), q, kind, 300)
        rec["probs"]["dwa"] = p; rec["preds"]["dwa"] = t; rec["tokens"]["dwa"] = nt
        # AVR: keep top 10% by the attention available at the boundary (L12-16 mean)
        rank = r["rank"]
        kk = max(1, int(round(K * len(rank))))
        keep_idx = np.argsort(-rank)[:kk]
        p, t, nt = run_arm(img, q, kind, E_HI, prune=lambda m, ki=keep_idx: set(int(x) for x in ki if x < m), from_layer=P + 1)
        rec["probs"]["avr"] = p; rec["preds"]["avr"] = t; rec["tokens"]["avr"] = nt
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 20 == 0: print(f"  arms [{n}]", flush=True)
print(f"Done -> {OUT}", flush=True)
