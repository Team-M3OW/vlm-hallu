"""
Phase 205 (GPU): TSR's premise at its own operating point, at the VALUE level. PREREG_MECH_THEORY.md.

TSR prunes 90% of the 900-token visual set at L16 because those tokens are causally inert. Behaviourally
tested (§26C/§38: TSR ~= uniform@900 within 0.5pp). This run tests the premise on hidden VALUES, on the
UNPRUNED 900-token pass, with an in-run contrast:

  patch DROPPED positions at L16 with another image's hidden states  -> must be at the noise floor
  patch KEPT    positions at L16 with the same donor                  -> must be LARGE
  patch DROPPED positions at L8  with the same donor                  -> must be LARGE (transport window)
  patch DROPPED positions at L24 with the same donor                  -> second floor control

plus TSR itself (attention keep and random keep) against the unpruned base, same pass.

PRE-REGISTERED (both models required)
  P-T1  mean KL(patch dropped @L16) <= 0.01  AND  mean KL(patch kept @L16) >= 0.10
        AND mean KL(patch dropped @L8) >= 0.10
  P-T2  TSR (attention keep) vs unpruned base: mean KL <= 0.02, arg-max agreement >= 97%;
        |KL(attn keep) - KL(random keep)| <= 0.01
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
WHICH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 191
MODEL_ID = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct", "qwen2": "Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT = f"{D}/data/phase205_tsr_probe_{WHICH}.jsonl"
NL, P, E, K = 28, 16, 900, 0.10
Image.MAX_IMAGE_PIXELS = None


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
        b = getattr(module, "_prune_bias", None)
        if b is not None and b.shape[-1] == w.shape[-1]: w = w + b.to(w.dtype).view(1, 1, 1, -1)
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
    return patched


QM3.eager_attention_forward = make_patched(QM3)
QM2.eager_attention_forward = make_patched(QM2)


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
    layers = model.model.language_model.layers; assert len(layers) == NL
    opt = [sorted({tok(x, add_special_tokens=False)["input_ids"][-1] for x in [c, f" {c}"]}) for c in "ABCD"]

    def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
    def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
    def measure(i): return int(sum(g[1] * g[2] // 4 for g in build(i, "x")["image_grid_thw"].tolist()))
    def fit(img, target, refine=6, tol=0.06):
        W_, H_ = img.size; sc = (target / max(measure(img), 1)) ** 0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol: break
            sc *= (target / r) ** 0.5
        return best
    def probs_from(lg):
        return torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)

    def clear():
        for l in layers:
            if hasattr(l.self_attn, "_prune_bias"): del l.self_attn._prune_bias

    def run(inp, drop=None, from_layer=None, want_attn=False):
        clear()
        if drop is not None and len(drop):
            b = torch.zeros(inp["input_ids"].shape[1], device=model.device)
            b[torch.as_tensor(drop, device=model.device)] = -1e4
            for li in range(from_layer, NL): layers[li].self_attn._prune_bias = b
        with torch.no_grad(): out = model(**inp, output_attentions=want_attn)
        lg = out.logits[0, -1].float()
        p = probs_from(lg)
        A = None
        if want_attn: A = np.stack([out.attentions[L][0, :, -1, :].float().mean(0).cpu().numpy() for L in range(NL)])
        del out; torch.cuda.empty_cache(); clear()
        return p, A

    def hidden_states(inp, ls):
        with torch.no_grad(): out = model(**inp, output_hidden_states=True)
        return {l: out.hidden_states[l + 1][0].detach() for l in ls}

    def make_patch(donor, mask):
        def fn(mod, args, output):
            o = output[0] if isinstance(output, tuple) else output
            o = o.clone(); k = min(int(mask.sum()), donor.shape[0])
            idx = mask.nonzero().flatten()[:k]
            o[0, idx] = donor[:k].to(o.dtype)
            return (o,) + tuple(output[1:]) if isinstance(output, tuple) else o
        return fn

    def run_patch(inp, L, donor, mask):
        h = layers[L].register_forward_hook(make_patch(donor, mask))
        try:
            with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        finally:
            h.remove()
        return probs_from(lg)

    def kl(a, b):
        return float((a * (a.clamp_min(1e-9).log() - b.clamp_min(1e-9).log())).sum())

    lut = [(f"{e['category']}/{e['question_id']}", e) for e in ds]
    done = set()
    if os.path.exists(OUT): done = {json.loads(l)["question_id_full"] for l in open(OUT)}
    rng = random.Random(205); n = 0; t0 = time.time()
    with open(OUT, "a") as fout:
        for i, (qid, ex) in enumerate(lut):
            if qid in done or n >= N: continue
            ip = os.path.join(root, ex["image"])
            dq, dex = lut[(i + 37) % len(lut)]
            dp = os.path.join(root, dex["image"])
            if not (os.path.exists(ip) and os.path.exists(dp)): continue
            img = Image.open(ip).convert("RGB")
            lab = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            sm, _ = fit(img, E); inp = build(sm, ex["text"]).to(model.device)
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            base_i, nt = int(pos[0]), int(len(pos))
            base, A = run(inp, want_attn=True)
            Ai = A[:, base_i:base_i + nt]; Ai = Ai / np.maximum(Ai.sum(1, keepdims=True), 1e-12)
            s = Ai[max(0, P - 4):P + 1].mean(0)
            keep = max(1, int(round(K * nt))); order = np.argsort(-s)
            drop = (base_i + order[keep:]).tolist(); keep_idx = (base_i + order[:keep])
            keep_mask = torch.zeros_like(inp["input_ids"][0], dtype=torch.bool); keep_mask[keep_idx] = True
            drop_mask = torch.zeros_like(keep_mask); drop_mask[torch.as_tensor(drop, device=drop_mask.device)] = True
            p_tsr, _ = run(inp, drop=drop, from_layer=P + 1)
            rdrop = [base_i + j for j in rng.sample(range(nt), nt - keep)]
            p_rand, _ = run(inp, drop=rdrop, from_layer=P + 1)

            dimg = Image.open(dp).convert("RGB"); dsm, _ = fit(dimg, E)
            dinp = build(dsm, ex["text"]).to(model.device)
            dpos = (dinp["input_ids"][0] == itid).nonzero().flatten()
            hid = hidden_states(dinp, [16, 8, 24])
            d16 = hid[16][dpos]; d8 = hid[8][dpos]; d24 = hid[24][dpos]
            patch = {}
            for tag, L, donor, mask in [("drop_L16", 16, d16, drop_mask), ("keep_L16", 16, d16, keep_mask),
                                        ("drop_L8", 8, d8, drop_mask), ("drop_L24", 24, d24, drop_mask)]:
                p = run_patch(inp, L, donor, mask)
                patch[tag] = {"kl": round(kl(base, p), 6), "flip": int(int(p.argmax()) != int(base.argmax()))}
            del hid, d16, d8, d24, dinp; torch.cuda.empty_cache()
            rec = {"question_id_full": qid, "category": ex["category"], "label": lab,
                   "nt": nt, "keep": keep, "base": [round(float(v), 6) for v in base.tolist()],
                   "tsr": [round(float(v), 6) for v in p_tsr.tolist()],
                   "tsr_rand": [round(float(v), 6) for v in p_rand.tolist()],
                   "kl_tsr": round(kl(base, p_tsr), 6), "kl_rand": round(kl(base, p_rand), 6),
                   "patch": patch}
            fout.write(json.dumps(rec) + "\n"); fout.flush(); n += 1
            if n % 10 == 0: print(f"  [{n}/{N}] {(time.time()-t0)/n:.1f}s/item", flush=True)
    print(f"Done -> {OUT}", flush=True)


main()
