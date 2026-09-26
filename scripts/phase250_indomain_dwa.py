"""Phase 250: in-domain DWA on CV-Bench Depth+Distance (the only CV-Bench tasks with boxes).

For each item: one 300-token localise pass captures per-layer attention; the ridge is fitted
out-of-fold on the benchmark's own boxes (coverage of the union box by the W=0.25 window); the
in-domain crop cell is the ring-masked argmax of the OOF score; a second pass answers on the crop.
Zero-shot DWA, AVR, block, and the reference are reused from phase225 on the same qids.

Usage: phase250_indomain_dwa.py <tag> [n_per_task]    tag in {qwen3_2b, qwen2_7b}
"""
import json, os, sys, time, importlib, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.model_selection import GroupKFold
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Image.MAX_IMAGE_PIXELS = None
W = 0.25
MODELS = {"qwen3_2b": "Qwen/Qwen3-VL-2B-Instruct", "qwen2_7b": "Qwen/Qwen2-VL-7B-Instruct"}
TAG = sys.argv[1]; NPT = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUT = f"{D}/data/phase250_indomain_{TAG}.jsonl"
TASKS = ("Depth", "Distance")

model = AutoModelForImageTextToText.from_pretrained(MODELS[TAG], dtype=torch.bfloat16, device_map={"": 0},
                                                    attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(MODELS[TAG]); tok = pr.tokenizer
layers = None
for f in (lambda m: m.model.language_model.layers, lambda m: m.language_model.model.layers,
          lambda m: m.model.text_model.layers, lambda m: m.model.layers):
    try: layers = f(model); break
    except Exception: pass
NL = len(layers); BLK = (round(0.57 * NL), NL - 1)
itid = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
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


def localise(img, q):
    src = fit_budget(img, 300); inp = build(src, q).to(model.device)
    if "image_grid_thw" not in inp: del inp; torch.cuda.empty_cache(); return None
    g = inp["image_grid_thw"].tolist()[0]; gh, gw = g[1] // 2, g[2] // 2
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    A = np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]); A = A[:, b:b + gh * gw]
    del inp; torch.cuda.empty_cache()
    return (A, gh, gw) if A.shape[1] == gh * gw else None


def feats(A, gh, gw):
    a = A / np.maximum(A.sum(1, keepdims=True), 1e-12); nc = gh * gw
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    LA = np.log(a + 1e-12).T; R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q_:q_ + gw] for p in range(3) for q_ in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo]


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def crop(img, cx, cy):
    iw, ih = img.size; x0 = min(max(0, (cx - W / 2) * iw), iw - W * iw); y0 = min(max(0, (cy - W / 2) * ih), ih - W * ih)
    return img.crop((int(x0), int(y0), int(x0 + W * iw), int(y0 + W * ih)))


from datasets import load_dataset
ds = load_dataset("nyu-visionx/CV-Bench")["test"]
meta = {f"cvbench/{e['idx']}": (e["task"], e.get("bbox")) for e in ds}
items = [(qid, img, q, gold, st, kind) for qid, img, q, gold, st, kind in B.load("cvbench")
         if st in TASKS and meta.get(qid) and meta[qid][1]]
# take the first NPT per task
sel = []
cnt = {t: 0 for t in TASKS}
for it in items:
    t = meta[it[0]][0]
    if cnt[t] < NPT: sel.append(it); cnt[t] += 1
print(f"{TAG}: {len(sel)} items {cnt}", flush=True)

done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}

# ---- pass 1: localise every item, collect features and coverage labels ----
data = []
t0 = time.time()
for qid, img, q, gold, st, kind in sel:
    if qid in done: continue
    img = img.convert("RGB"); bbox = meta[qid][1]
    gx0 = min(b[0] for b in bbox); gy0 = min(b[1] for b in bbox)
    gx1 = max(b[0] + b[2] for b in bbox); gy1 = max(b[1] + b[3] for b in bbox)
    iw, ih = img.size; gt = (gx0 / iw, gy0 / ih, gx1 / iw, gy1 / ih)
    L = localise(img, q)
    if L is None: continue
    A, gh, gw = L
    X = feats(A, gh, gw)
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    Y = np.array([cov(float(fx[i]), float(fy[i]), gt) for i in range(gh * gw)])
    data.append({"qid": qid, "img": img, "q": q, "gold": gold, "kind": kind, "task": st,
                 "gt": gt, "gh": gh, "gw": gw, "X": X, "Y": Y})
    if len(data) % 25 == 0: print(f"  localise [{len(data)}/{len(sel)}] {(time.time()-t0)/len(data):.1f}s/item", flush=True)

# ---- fit once, out-of-fold over items ----
X = np.vstack([d["X"] for d in data]); Y = np.concatenate([d["Y"] for d in data])
G = np.concatenate([np.full(len(d["Y"]), i) for i, d in enumerate(data)])
P = np.zeros(len(Y)); Wsum = np.zeros(X.shape[1] + 1); nfit = 0
for perm_seed in (700, 701, 702):
    rngp = np.random.default_rng(perm_seed); perm = {g: i for i, g in enumerate(rngp.permutation(np.unique(G)))}
    Gp = np.vectorize(perm.get)(G)
    for tr, te in GroupKFold(5).split(X, Y, Gp):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
        A_ = Xt.T @ Xt + np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
        w = np.linalg.solve(A_, Xt.T @ Y[tr]); Wsum += w; nfit += 1
        P[te] += np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
P /= 3

# ---- pass 2: in-domain crop per item, answer on it ----
recs = []
with open(OUT, "a") as f:
    off = 0
    for d in data:
        nc = d["gh"] * d["gw"]; sc = P[off:off + nc].reshape(d["gh"], d["gw"]); off += nc
        rm = np.zeros((d["gh"], d["gw"]), bool)
        if d["gh"] > 2 and d["gw"] > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        iy, ix = np.unravel_index(int(np.argmax(np.where(rm, sc, -1e9))), sc.shape)
        ccx, ccy = (ix + .5) / d["gw"], (iy + .5) / d["gh"]
        crop_ok = cov(ccx, ccy, d["gt"])
        src = fit_budget(crop(d["img"], ccx, ccy), 300); inp = build(src, d["q"]).to(model.device)
        p, _ = B.score_item(model, pr, tok, inp, d["kind"])
        del inp; torch.cuda.empty_cache()
        rec = {"qid": d["qid"], "task": d["task"], "gold": d["gold"], "kind": d["kind"], "gt": d["gt"],
               "cov_indomain": float(crop_ok), "probs": {"indomain": p}}
        f.write(json.dumps(rec) + "\n"); f.flush(); recs.append(rec)
        if len(recs) % 25 == 0: print(f"  answer [{len(recs)}/{len(data)}]", flush=True)
print(f"Done -> {OUT}", flush=True)

# quick table vs phase225 on the same qids
cache = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/data/phase225_{TAG}_cvbench.jsonl")}
q = [r for r in recs if r["qid"] in cache]
rng = np.random.default_rng(250)
def ok(r, arm):
    src = r["probs"].get(arm) if arm in r["probs"] else cache[r["qid"]]["probs"].get(arm)
    return None if src is None else float(int(np.argmax(src)) == r["gold"])
def ci(d):
    d = np.asarray(d, float); m = d[rng.integers(0, len(d), (8000, len(d)))].mean(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100
ref = np.array([ok(r, "uniform@lo") for r in q])
print(f"\nCV-Bench Depth+Distance, {TAG}, n={len(q)}  reference {ref.mean()*100:.1f}%")
for arm in ("avr", "dwa_t", "block", "indomain"):
    v = [ok(r, arm) for r in q]
    if any(x is None for x in v): print(f"  {arm:9s} n/a"); continue
    m, l, h = ci(np.array(v) - ref)
    print(f"  {arm:9s} acc {np.mean(v)*100:5.1f}%   -ref {m:+5.1f} [{l:+5.1f},{h:+5.1f}]")
