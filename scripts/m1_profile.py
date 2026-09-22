"""
M1 follow-up (GPU): did the alignment training change the CAUSAL transport profile, or only the attention map?

P2 of the M1 pre-registration: masking all text rows' image attention for prefix L0..16 should give a LARGER
output KL after training than before if the window's attention became causally load-bearing; the answer-row-only
mask must stay at the noise floor. Measured on the M1 held-out items (idx % 4 == 3), n=20.
"""
import json, os, sys, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
ADAPTER = f"{D}/models/m1_lora_qwen3"
OUT = f"{D}/data/m1_profile.json"
B0 = 300; Image.MAX_IMAGE_PIXELS = None
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
    layers = model.model.language_model.layers; NL = len(layers)
    opt = [sorted({tok(x, add_special_tokens=False)["input_ids"][-1] for x in [c, f" {c}"]}) for c in "ABCD"]
    state = {"layers": set(), "span": None}

    def mk(l):
        def pre(mod, args, kwargs):
            if l not in state["layers"]: return None
            am = kwargs.get("attention_mask")
            if am is None or am.dtype == torch.bool: return None
            am = am.clone(); a, b = state["span"]
            if state.get("rows") == "all": am[:, :, b:, a:b] = torch.finfo(am.dtype).min
            else: am[:, :, -1:, a:b] = torch.finfo(am.dtype).min
            kwargs["attention_mask"] = am; return (args, kwargs)
        return pre
    hooks = [layers[l].self_attn.register_forward_pre_hook(mk(l), with_kwargs=True) for l in range(NL)]

    def logits(inp):
        with torch.no_grad(): return model(**inp).logits[0, -1].float()

    def kl(p, q):
        p = torch.softmax(p, -1); lq = torch.log_softmax(q, -1)
        return float((p * (torch.log(p + 1e-12) - lq)).sum())

    def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
    def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
    def measure(i): return int(sum(g[1] * g[2] // 4 for g in build(i, "x")["image_grid_thw"].tolist()))
    def fit(img, target=B0, refine=6, tol=0.08):
        W_, H_ = img.size; sc = (target / max(measure(img), 1)) ** 0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol: break
            sc *= (target / r) ** 0.5
        return best[0]

    def evaluate(tag, n_items=20):
        out = []
        for idx, ex in enumerate(ds):
            if idx % 4 != 3 or len(out) >= n_items: continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img = fit(Image.open(ip).convert("RGB")); inp = build(img, ex["text"]).to(model.device)
            pos = (inp["input_ids"][0] == itid).nonzero().flatten(); state["span"] = (int(pos[0]), int(pos[-1]) + 1)
            state["layers"] = set(); base = logits(inp); bl = base[opt[0]] * 0  # placeholder
            letters = [torch.logsumexp(base[i], 0) for i in opt]
            bl = torch.stack(letters)
            rec = {"qid": f"{ex['category']}/{ex['question_id']}"}
            for nm, ls, rows in [("prefix_L16", set(range(0, 17)), "all"), ("suffix_L16", set(range(16, NL)), "all"),
                                 ("prefix_L12", set(range(0, 13)), "all"), ("answer_row", set(range(NL)), "row"),
                                 ("all_layers", set(range(NL)), "all")]:
                state["layers"] = ls; state["rows"] = rows
                q = logits(inp)
                rec[nm] = {"kl": kl(bl, torch.stack([torch.logsumexp(q[i], 0) for i in opt])),
                           "flip": int(int(torch.argmax(bl)) != int(torch.argmax(torch.stack([torch.logsumexp(q[i], 0) for i in opt]))))}
            out.append(rec)
        state["layers"] = set()
        for k in ("prefix_L16", "suffix_L16", "prefix_L12", "answer_row", "all_layers"):
            print(f"  [{tag}] {k:11s} KL {np.mean([r[k]['kl'] for r in out]):.4f}  flips {100*np.mean([r[k]['flip'] for r in out]):4.1f}%", flush=True)
        return out

    before = evaluate("base")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    after = evaluate("adapter")
    json.dump({"before": before, "after": after}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


main()
