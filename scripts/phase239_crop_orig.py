"""Phase 239: DWA crop + original image, and multi-crop + original (all bar-matched at 600 tokens).

The two-pass DWA loses on cross-instance questions because the W=0.25 crop discards the second
object. Feeding the crop AND the original image in one prompt restores global coverage while
keeping the crop's resolution on the target region. Three arms, each 600 visual tokens = the bar:

  crop+orig    1 crop @300  + original @300
  crop2+orig   2 crops @150 + original @300   (NMS peaks of the ridge score)
  crop4+orig   4 crops @75  + original @300

PRE-REGISTERED (before running):
  P1 crop+orig recovers the cross stratum to ~bar (the original restores the second object) while
     keeping part of DWA's single-stratum gain; pooled positive on single, non-negative on cross.
  P2 crop2+orig beats crop+orig on cross if the two crops land on the two objects; lower crop
     resolution (150 vs 300) costs some single.
  P3 crop4+orig starves crops (75 tokens): single falls.
  If both strata are at/below bar on both models, the add-the-original-back family is rejected.

Cached phase225 arms (uniform@lo = bar, dwa_t = crop alone) are reused for the paired comparison;
only the new arms run here. Usage: phase239_crop_orig.py <model> <bench> [n]
"""
import json, os, sys, time, importlib, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Image.MAX_IMAGE_PIXELS = None
W = 0.25
MODELS = {"qwen3_2b": ("Qwen/Qwen3-VL-2B-Instruct", "qwen3"), "qwen2_7b": ("Qwen/Qwen2-VL-7B-Instruct", "qwen2")}
MK, BK = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
mid, wtag = MODELS[MK]
OUT = f"{D}/data/phase239_crop_orig_{MK}_{BK}.jsonl"
E_BAR, CROP_B = 600, 300
LAYOUT = json.load(open(f"{D}/data/phase224_layout.json")) if os.path.exists(f"{D}/data/phase224_layout.json") else {}
WP = f"{D}/data/fig_ridge_w_{wtag}.npy"; Wv = np.load(WP) if os.path.exists(WP) else None
assert Wv is not None, "ridge weights missing; DWA arm impossible"
print(f"{MK}/{BK}: weights yes", flush=True)

model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(mid); tok = pr.tokenizer
layers = None
for f in (lambda m: m.model.language_model.layers, lambda m: m.language_model.model.layers,
          lambda m: m.model.text_model.layers, lambda m: m.model.layers):
    try: layers = f(model); break
    except Exception: pass
assert layers is not None
NL = len(layers); P = round(0.57 * NL); BLK = (P, NL - 1)
itid = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
modname = type(layers[0].self_attn).__module__; QM = importlib.import_module(modname)
assert hasattr(QM, "eager_attention_forward"), f"{modname} has no eager_attention_forward"


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
print(f"  NL={NL} boundary L{P} attn={modname}", flush=True)


def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
def chat_multi(t, k): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * k + [{"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
def build_multi(imgs, t): return pr(images=imgs, text=chat_multi(t, len(imgs)), return_tensors="pt")
def ntok(inp): return int((inp["input_ids"][0] == itid).sum())


def fit_budget(img, target, refine=6, tol=0.10):
    W_, H_ = img.size; r0 = max(ntok(build(img, "x")), 1); sc = (target / r0) ** 0.5; best = None
    for _ in range(refine):
        cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC); r = ntok(build(cur, "x"))
        if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
        if r == 0 or abs(r - target) / target <= tol: break
        sc *= (target / max(r, 1)) ** 0.5
    return best


def grid_of(inp):
    if "image_grid_thw" in inp:
        g = inp["image_grid_thw"].tolist()[0]; return g[1] // 2, g[2] // 2, 0
    n = int((inp["input_ids"][0] == itid).sum()); L = LAYOUT.get(MK)
    if L:
        gh, gw = L["gh"], L["gw"]; off = n - gh * gw if L.get("mode", "suffix") == "suffix" else 0
        if off >= 0 and gh * gw <= n: return gh, gw, off
    s = int(round(n ** 0.5)); return (s, s, 0) if s * s == n else (0, 0, 0)


def localise(img, q):
    L = LAYOUT.get(MK) or {}
    src = img.resize((L["resize"], L["resize"]), Image.BICUBIC) if L.get("resize") else fit_budget(img, CROP_B)[0]
    inp = build(src, q).to(model.device)
    gh, gw, off = grid_of(inp)
    if gh * gw == 0: del inp; torch.cuda.empty_cache(); return None
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    if len(pos) < off + gh * gw: del inp; torch.cuda.empty_cache(); return None
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    A = np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]) + off; A = A[:, b:b + gh * gw]; nt = ntok(inp)
    del inp; torch.cuda.empty_cache()
    return (A, gh, gw, nt) if A.shape[1] == gh * gw else None


def feats(A, gh, gw):
    a = A / np.maximum(A.sum(1, keepdims=True), 1e-12); nc = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    LA = np.log(a + 1e-12).T; R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo], dep


def crop(img, cx, cy):
    iw, ih = img.size; x0 = min(max(0, (cx - W / 2) * iw), iw - W * iw); y0 = min(max(0, (cy - W / 2) * ih), ih - W * ih)
    return img.crop((int(x0), int(y0), int(x0 + W * iw), int(y0 + W * ih)))


def topk_centers(s, gh, gw, k):
    s = np.asarray(s, float).reshape(gh, gw)
    rm = np.zeros((gh, gw), bool)
    if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
    else: rm[:] = True
    m = np.where(rm, s, -1e9).copy(); cs = []
    ry, rx = max(1, int(round(gh * W / 2))), max(1, int(round(gw * W / 2)))
    for _ in range(k):
        iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
        cs.append(((ix + .5) / gw, (iy + .5) / gh))
        m[max(0, iy - ry):min(gh, iy + ry + 1), max(0, ix - rx):min(gw, ix + rx + 1)] = -1e9
    return cs


def run_multi(imgs, budgets, q, kind):
    srcs = [fit_budget(im, b)[0] if b else im for im, b in zip(imgs, budgets)]
    inp = build_multi(srcs, q).to(model.device); n = ntok(inp)
    p, txt = B.score_item(model, pr, tok, inp, kind)
    del inp; torch.cuda.empty_cache()
    return p, txt, n


items = B.load(BK, N or None)
import collections as _c
print(f"  {len(items)} items  strata={dict(_c.Counter(i[4] for i in items))}", flush=True)
done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
t0 = time.time(); n = 0
with open(OUT, "a") as f:
    for qid, img, q, gold, stratum, kind in items:
        if qid in done: continue
        img = img.convert("RGB")
        L = localise(img, q)
        if L is None: continue
        A, gh, gw, nloc = L
        X, dep = feats(A, gh, gw)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        s = np.c_[(X - mu) / sd, np.ones(len(X))] @ Wv
        cs = topk_centers(s, gh, gw, 4)
        rec = {"qid": qid, "stratum": stratum, "kind": kind, "gold": gold, "probs": {}, "tokens": {}}
        for name, k, bud in (("crop+orig", 1, 300), ("crop2+orig", 2, 150), ("crop4+orig", 4, 75)):
            imgs = [crop(img, *cs[i]) for i in range(k)] + [img]
            budgets = [bud] * k + [300]
            p, txt, nt = run_multi(imgs, budgets, q, kind)
            if p is not None: rec["probs"][name] = p
            rec["tokens"][name] = nt
        rec["tokens"]["localise"] = nloc
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 10 == 0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item", flush=True)
print(f"Done -> {OUT}  ({n} new)", flush=True)

# ---- paired analysis vs cached phase225 arms -------------------------------------------------
rng = np.random.default_rng(239)
def boot(d, Bn=8000):
    d = np.asarray(d, float); m = d[rng.integers(0, len(d), (Bn, len(d)))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100
cache = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/phase225_{MK}_{BK}.jsonl")}
new = [json.loads(l) for l in open(OUT)]
q = [r for r in new if r["qid"] in cache and "uniform@lo" in cache[r["qid"]].get("probs", {})]
print(f"\npaired n={len(q)}")
lab = np.array([r["stratum"] in ("direct_attributes", "single") for r in q])
clab = np.array([r["stratum"] in ("relative_position", "cross") for r in q])
def corr(rec, arm):
    if arm in rec.get("probs", {}): return 1.0 if int(np.argmax(rec["probs"][arm])) == int(rec["gold"]) else 0.0
    return None
for arm in ("dwa_t", "crop+orig", "crop2+orig", "crop4+orig"):
    d = []; ds = []; dc = []
    for r in q:
        src = cache[r["qid"]] if arm == "dwa_t" else r
        a = corr(src, arm); b = corr(cache[r["qid"]], "uniform@lo")
        if a is None or b is None: continue
        d.append(a - b)
        if r["stratum"] in ("direct_attributes", "single"): ds.append(a - b)
        if r["stratum"] in ("relative_position", "cross"): dc.append(a - b)
    if not d: continue
    m, lo, hi = boot(d); ms = boot(ds) if len(ds) > 20 else (float("nan"),) * 3; mc = boot(dc) if len(dc) > 20 else (float("nan"),) * 3
    toks = np.mean([r["tokens"].get(arm, 0) for r in q])
    print(f"  {arm:12s} tok~{toks:5.0f}  policy-bar {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]  single {ms[0]:+5.1f} [{ms[1]:+5.1f},{ms[2]:+5.1f}]  cross {mc[0]:+5.1f} [{mc[1]:+5.1f},{mc[2]:+5.1f}]")
