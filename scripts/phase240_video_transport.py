"""Phase 240 (Track V1): does the transport boundary exist for VIDEO tokens?

The image protocol (phase176c) applied to synthetic videos built from V*Bench images:
video = the item's image repeated over 8 frames (temporal_patch_size=2 -> t=4 patches), question =
the item's own V* question. At selected layers we block every text row after the video span from
attending to the video-token columns, then measure the KL divergence and answer flips at the
output. Single-layer, prefix, suffix and all-layer schedules, exactly as on images.

Pre-registered (PREREG_CROSSMODAL.md H1): the boundary sits at ~0.57 of depth (L16 of 28); the
suffix mask is inert from there. If the boundary is elsewhere or absent, that is the result.

Usage: phase240_video_transport.py <tag> [n]      tag in {qwen3}
"""
import json, os, sys, time, importlib, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); Image.MAX_IMAGE_PIXELS = None
MODELS = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct"}
TAG = sys.argv[1]; NCAL = int(sys.argv[2]) if len(sys.argv) > 2 else 40
MID = MODELS[TAG]; OUT = f"{D}/data/phase240_video_transport_{TAG}.json"
VTOK_TARGET = 600
NFRAMES = 8

model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
pr = AutoProcessor.from_pretrained(MID); tok = pr.tokenizer
layers = None
for f in (lambda m: m.model.language_model.layers, lambda m: m.language_model.model.layers,
          lambda m: m.model.text_model.layers, lambda m: m.model.layers):
    try: layers = f(model); break
    except Exception: pass
assert layers is not None
NL = len(layers)
vid_id = getattr(model.config, "video_token_id", None)
modname = type(layers[0].self_attn).__module__; QM = importlib.import_module(modname)
assert hasattr(QM, "eager_attention_forward"), f"{modname} has no eager_attention_forward"


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
    return patched
QM.eager_attention_forward = make_patched(QM)
print(f"  NL={NL} attn={modname} video_token_id={vid_id}", flush=True)


def build(frames, text):
    content = [{"type": "video", "video": frames}, {"type": "text", "text": text}]
    chat = pr.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inp = pr(text=[chat], videos=[frames], do_sample_frames=False, return_tensors="pt")
    # Qwen3-VL INTERLEAVES a frame-timestamp text block between every per-frame token block, so
    # mm_token_type_ids has t separate video runs while the processor emits ONE grid row [[t,h,w]].
    # get_rope_index iterates the runs and calls next() on that 1-item iterator -> StopIteration.
    # Expand to one row per temporal patch. VALIDATED: video path vs image path on identical V*
    # content agree on 82.5% of items (acc 30.0 vs 27.5), i.e. the patched path is not garbage.
    g = inp["video_grid_thw"]; t, h, w = [int(x) for x in g[0]]
    if len(g) == 1 and t > 1:
        inp["video_grid_thw"] = torch.tensor([[1, h, w]] * t, dtype=g.dtype)
    return inp


def vspan(inp):
    """Returns (first, last+1, count, EXACT_COLUMNS). The video tokens are NOT contiguous -- the
    timestamp text sits between frame blocks -- so masking the span first..last would also blind
    the text to those timestamps and the measurement would not be about video transport."""
    if "mm_token_type_ids" in inp:
        pos = (inp["mm_token_type_ids"][0] == 2).nonzero().flatten()
        if len(pos): return int(pos[0]), int(pos[-1]) + 1, len(pos), pos
    if vid_id is not None:
        pos = (inp["input_ids"][0] == vid_id).nonzero().flatten()
        if len(pos): return int(pos[0]), int(pos[-1]) + 1, len(pos), pos
    return None


def fit_frames(img, target=VTOK_TARGET):
    """Scale the (single) frame so the 8-frame video lands near the token target."""
    W_, H_ = img.size; best = None; sc = 1.0
    for _ in range(6):
        fr = img.resize((max(64, int(W_ * sc)), max(64, int(H_ * sc))), Image.BICUBIC)
        inp = build([fr] * NFRAMES, "x"); sp = vspan(inp); n = sp[2] if sp else 0
        if best is None or abs(n - target) < abs(best[1] - target): best = (fr, n)
        if n == 0 or abs(n - target) / target <= 0.08: break
        sc *= (target / max(n, 1)) ** 0.5
    return best


state = {"layers": set(), "span": None, "cols": None, "rows": None}


def make_hook(l):
    def pre(mod, args, kwargs):
        if l not in state["layers"]: return None
        am = kwargs.get("attention_mask")
        if am is None: am = args[1] if len(args) > 1 else None
        if am is None or am.dtype == torch.bool:
            raise RuntimeError(f"unexpected attention_mask {None if am is None else am.dtype}")
        am = am.clone(); cols = state["cols"]; rows = state["rows"]
        # LEAK FIX: masking only rows AFTER the last frame (b:) leaves the interleaved TIMESTAMP
        # text rows, which sit INSIDE the video span, free to read earlier frames at every layer
        # and relay that content onward through unmasked columns. Mask every NON-VIDEO row from
        # the first video token onward. A weak kl_all/flip_all is the signature of this leak.
        am[:, :, rows.unsqueeze(1), cols.unsqueeze(0)] = torch.finfo(am.dtype).min
        kwargs["attention_mask"] = am; return (args, kwargs)
    return pre
hooks = [layers[l].self_attn.register_forward_pre_hook(make_hook(l), with_kwargs=True) for l in range(NL)]


def logits(inp):
    with torch.no_grad(): return model(**inp).logits[0, -1].float()


def kl(p, q):
    p = torch.softmax(p, -1); lq = torch.log_softmax(q, -1)
    return float((p * (torch.log(p + 1e-12) - lq)).sum())


items = B.load("vstar", None)
letters = [tok.encode(x, add_special_tokens=False)[0] for x in ["A", "B", "C", "D"]]
out = []; n = 0; t0 = time.time()
for qid, img, q, gold, stratum, kind in items:
    if n >= NCAL: break
    img = img.convert("RGB")
    fr, vt = fit_frames(img)
    frames = [fr] * NFRAMES
    inp = build(frames, q).to(model.device)
    sp = vspan(inp)
    if sp is None: continue
    state["span"] = (sp[0], sp[1]); state["cols"] = sp[3].to(model.device)
    _vset = set(int(x) for x in sp[3])
    state["rows"] = torch.tensor([i for i in range(sp[0], inp["input_ids"].shape[1]) if i not in _vset],
                                 device=model.device)
    state["layers"] = set(); base = logits(inp); bl = base[letters]
    state["layers"] = set(range(NL)); allab = logits(inp)
    rec = {"qid": qid, "stratum": stratum, "vtokens": sp[2], "kl_all": kl(base, allab),
           "flip_all": int(int(torch.argmax(bl)) != int(torch.argmax(allab[letters]))),
           "kl": [], "flip": [], "prefix": {}, "suffix": {}}
    for l in range(NL):
        state["layers"] = {l}; qq = logits(inp)
        rec["kl"].append(kl(base, qq)); rec["flip"].append(int(int(torch.argmax(bl)) != int(torch.argmax(qq[letters]))))
    for l in range(0, NL, 4):
        state["layers"] = set(range(0, l + 1)); qq = logits(inp)
        rec["prefix"][str(l)] = [kl(base, qq), int(int(torch.argmax(bl)) != int(torch.argmax(qq[letters])))]
        state["layers"] = set(range(l, NL)); qq = logits(inp)
        rec["suffix"][str(l)] = [kl(base, qq), int(int(torch.argmax(bl)) != int(torch.argmax(qq[letters])))]
    out.append(rec); n += 1
    if n % 5 == 0: print(f"  [{n}/{NCAL}] {(time.time()-t0)/n:.1f}s/item  vt={sp[2]}  kl_all={rec['kl_all']:.3f}", flush=True)
json.dump(out, open(OUT, "w"))
K = np.array([r["kl"] for r in out])
print(f"\nvideo tokens mean {np.mean([r['vtokens'] for r in out]):.0f}")
print(f"all-layer mask: KL {np.mean([r['kl_all'] for r in out]):.3f}  flips {np.mean([r['flip_all'] for r in out]):.2f}")
print("per-layer KL:", " ".join(f"L{l}:{K[:,l].mean():.4f}" for l in range(NL)))
for nm in ("prefix", "suffix"):
    ks = sorted(out[0][nm], key=int)
    print(f"  {nm}: " + "  ".join(f"L{k}: KL {np.mean([r[nm][k][0] for r in out]):.3f} flip {np.mean([r[nm][k][1] for r in out]):.2f}" for k in ks))
