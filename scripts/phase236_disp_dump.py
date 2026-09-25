"""
Phase 236a: dump the policy's single input statistic for the 4x4 grid cells.

  disp = ring-masked top-1 mass share of the block-mean attention map from the LOCALISE pass @300.

One forward per item -- the same localise pass phase 225 ran (its arms are already recorded on disk,
so the policy can be replayed offline). Layout handling copied from phase225 so the cell grid matches.
Usage: phase236_disp_dump.py <model-key> <bench> [n]
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
MODELS = {"qwen3_2b": "Qwen/Qwen3-VL-2B-Instruct", "qwen2_7b": "Qwen/Qwen2-VL-7B-Instruct",
          "llava_ov": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "internvl3_8b": "OpenGVLab/InternVL3-8B-hf"}
MK, BK = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL_ID = MODELS[MK]; OUT = f"{D}/data/phase236_disp_{MK}_{BK}.jsonl"; Image.MAX_IMAGE_PIXELS = None
LAYOUT = json.load(open(f"{D}/data/phase224_layout.json")) if os.path.exists(f"{D}/data/phase224_layout.json") else {}
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer
itid = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
layers = None
for f in (lambda m: m.model.language_model.layers, lambda m: m.language_model.model.layers,
          lambda m: m.model.text_model.layers, lambda m: m.model.layers):
    try: layers = f(model); break
    except Exception: pass
assert layers is not None
NL = len(layers); P = round(0.57 * NL); BLK = (P, NL - 1)
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
def grid_of(inp):
    if "image_grid_thw" in inp:
        g = inp["image_grid_thw"].tolist()[0]; return g[1] // 2, g[2] // 2, 0
    n = ntok(inp); L = LAYOUT.get(MK)
    if L:
        gh, gw, mode = L["gh"], L["gw"], L.get("mode", "suffix")
        off = n - gh * gw if mode == "suffix" else 0
        if off >= 0 and gh * gw <= n: return gh, gw, off
    s = int(round(n ** 0.5)); return (s, s, 0) if s * s == n else (0, 0, 0)


def disp_of(img, q):
    L = LAYOUT.get(MK) or {}
    src = img.resize((L["resize"], L["resize"]), Image.BICUBIC) if L.get("resize") else fit_budget(img, 300)[0]
    inp = build(src, q).to(model.device)
    gh, gw, off = grid_of(inp)
    if gh * gw == 0: return None
    pos = (inp["input_ids"][0] == itid).nonzero().flatten()
    if len(pos) < off + gh * gw: return None
    for l in layers: l.self_attn._capture = True
    with torch.no_grad(): model(**inp)
    Ph = np.stack([l.self_attn._lastrow for l in layers])
    for l in layers: l.self_attn._capture = False; l.self_attn._lastrow = None
    b = int(pos[0]) + off; Ph = Ph[:, :, b:b + gh * gw]
    del inp; torch.cuda.empty_cache()
    if Ph.shape[2] != gh * gw: return None
    A = Ph.mean(1); a = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    M = a[BLK[0]:BLK[1]].mean(0)
    rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
    return float(np.max(np.where(rm, M, -1e9))) / max(M.sum(), 1e-12)


items = B.load(BK, N or None)
print(f"{MK}/{BK}: {len(items)} items", flush=True)
done = set()
if os.path.exists(OUT): done = {json.loads(l)["qid"] for l in open(OUT)}
n = 0; t0 = time.time()
with open(OUT, "a") as f:
    for qid, img, q, gold, stratum, kind in items:
        if qid in done: continue
        d = disp_of(img.convert("RGB"), q)
        if d is None: continue
        f.write(json.dumps({"qid": qid, "stratum": stratum, "disp": float(d)}) + "\n"); f.flush(); n += 1
        if n % 50 == 0: print(f"  [{n}] {(time.time()-t0)/n:.1f}s/item", flush=True)
print(f"Done -> {OUT}", flush=True)
