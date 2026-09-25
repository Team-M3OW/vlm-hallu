"""
Phase 237a: dump a battery of LABEL-FREE attention-map statistics per item, to find a better gate
signal than `disp` (block-mean top-1 share), which is at chance on HR-Bench (AUROC 0.544).

All signals come from the pass-1 localise map at 300 tokens (per-head last-row attention):
  disp       ring-masked top-1 mass share of the block-mean map          (current gate; higher = single)
  ent        entropy of the ring-masked block-mean map / log(n_masked)   (lower = single)
  top5       top-5 cell mass share of the block-mean map                 (higher = single)
  pom        max/median of the ring-masked block-mean map                (higher = single)
  band_disp  top-1 share of the read-out-band mean map (L17-21 / L19-23) (higher = single)
  band_ent   entropy of the read-out-band map                            (lower = single)
  ratio      read-out-band mass / block-mean mass                        (higher = single?)
  agree      mean cosine between adjacent layer maps                     (lower = single?)
  headmax    ring-max share of the per-cell max-over-heads map, block-mean(higher = single)
  vrh        block-mean mass inside the ridge cell's W=0.25 window       (higher = single)
Usage: phase237_signal_dump.py <model-key> <bench> [n]
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
MODEL_ID, WT = MODELS[MK]; OUT = f"{D}/data/phase237_sig_{MK}_{BK}.jsonl"; W = 0.25
Image.MAX_IMAGE_PIXELS = None
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
layers = model.model.language_model.layers; NL = len(layers)
P = round(0.57 * NL); BLK = (P, NL - 1); RO = (17, 22) if WT == "qwen3" else (19, 23)
WV = np.load(f"{D}/data/fig_ridge_w_{WT}.npy")
modname = type(layers[0].self_attn).__module__; QM = importlib.import_module(modname)


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
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


def signals_of(img, q):
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
    nc = gh * gw
    if Ph.shape[2] != nc: return None
    A = Ph.mean(1); a = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    M = a[BLK[0]:BLK[1]].mean(0); ROm = a[RO[0]:RO[1]].mean(0)
    rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
    Mm = np.where(rm, M, 0.0); Rm = np.where(rm, ROm, 0.0)
    n_m = max(rm.sum(), 1)
    ent = -np.sum((Mm / max(Mm.sum(), 1e-12)) * np.log(Mm / max(Mm.sum(), 1e-12) + 1e-12)) / np.log(n_m)
    b_m = Rm / max(Rm.sum(), 1e-12); be = -np.sum(b_m * np.log(b_m + 1e-12)) / np.log(n_m)
    top5 = np.sort(Mm)[-5:].sum() / max(Mm.sum(), 1e-12)
    pom = float(np.max(Mm)) / max(float(np.median(Mm[Mm > 0])) if (Mm > 0).any() else 1e-12, 1e-12)
    disp = float(np.max(Mm)) / max(Mm.sum(), 1e-12)
    bd = float(np.max(Rm)) / max(Rm.sum(), 1e-12)
    ratio = Rm.sum() / max(Mm.sum(), 1e-12)
    agree = float(np.mean([np.dot(a[l], a[l + 1]) / max(np.linalg.norm(a[l]) * np.linalg.norm(a[l + 1]), 1e-12) for l in range(NL - 1)]))
    # per-cell head concentration: normalise each cell over heads, take the max head share
    Ps = Ph / np.maximum(Ph.sum(1, keepdims=True), 1e-9)          # per-cell head distribution
    Hmap = Ps.max(1)[BLK[0]:BLK[1]].mean(0)
    headmax = float(np.max(np.where(rm, Hmap, -1e9))) / max(Hmap.sum(), 1e-12)
    # VRH: block-mean mass in the deployed ridge cell's W window
    yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
    LA = np.log(a + 1e-12).T; Rk = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(nc - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    X = np.c_[LA, Rk, geo]
    if X.shape[1] == len(WV) - 1:
        s = np.c_[(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones(len(X))] @ WV
        j = int(np.argmax(np.where(rm, s, -1e9))); cx, cy = (j % gw + .5) / gw, (j // gw + .5) / gh
        inwin = (np.abs(fx - cx) <= W / 2) & (np.abs(fy - cy) <= W / 2)
        vrh = float(M[inwin].sum()) / max(M.sum(), 1e-12)
    else:
        vrh = float("nan")
    return {"disp": disp, "ent": float(ent), "top5": float(top5), "pom": float(pom), "band_disp": float(bd),
            "band_ent": float(be), "ratio": float(ratio), "agree": agree, "headmax": float(headmax), "vrh": vrh}


items = B.load(BK, N or None)
print(f"{MK}/{BK}: {len(items)} items", flush=True)
done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
n = 0; t0 = time.time()
with open(OUT, "a") as f:
    for qid, img, q, gold, stratum, kind in items:
        if qid in done: continue
        s = signals_of(img.convert("RGB"), q)
        if s is None: continue
        s.update({"qid": qid, "stratum": stratum})
        s = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}
        f.write(json.dumps(s) + "\n"); f.flush(); n += 1
        if n % 50 == 0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item", flush=True)
print(f"Done -> {OUT}", flush=True)
