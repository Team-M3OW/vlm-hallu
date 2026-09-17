"""
Phase 142 (Track 3): the phase-95 label-free divergence curve for the two LLaVA models.

Phase 140 found the only additive training-free component is the LAYER SET: max over the layers
where attention is question-conditioned (phase 95 divergence >= 0.5*max) beats the hand-set block
on both Qwen models. That gate could not be tested on LLaVA because no divergence curve existed.
Same protocol as phase 95 (B0=300, 40 images, 4 questions: the item's own + 3 borrowed from other
items by the same index rule), cosine divergence over the raw image-token positions (separators
included -- identical positions across questions for a fixed image, so they cancel). No boxes.
Phase 142b applies the resulting gate to phase 82's stored per-layer maps offline.
"""
import json, os, time
import numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
MODELS = [("onevision", "llava-hf/llava-onevision-qwen2-7b-ov-hf"), ("llavanext", "llava-hf/llava-v1.6-vicuna-7b-hf")]
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase142_llava_divergence.json"
B0, N_IMG, K_Q = 300, 40, 4
Image.MAX_IMAGE_PIXELS = None

def run(tag, mid, items):
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager"); model.eval()
    pr = AutoProcessor.from_pretrained(mid); itid = model.config.image_token_id
    nL = model.config.get_text_config().num_hidden_layers; print(f"  {tag}: {nL} layers", flush=True)
    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False, add_generation_prompt=True), return_tensors="pt")
    def ntok(img): return int((build(img, "x")["input_ids"][0] == itid).sum())
    def fit(img, target, refine=6, tol=0.10):
        W_, H_ = img.size; sc = (target/max(ntok(img), 1))**0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(56, int(W_*sc)), max(56, int(H_*sc))), Image.BICUBIC); r = ntok(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r)**0.5
        return best[0]
    div = [[] for _ in range(nL)]
    for n, (ip, qs) in enumerate(items):
        img = fit(Image.open(ip).convert("RGB"), B0); maps = []
        for q in qs:
            inp = build(img, q); pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            if len(pos) < 16: maps = []; break
            base, nt = int(pos[0].item()), int(len(pos))
            with torch.no_grad(): o = model(**inp.to(model.device), output_attentions=True)
            A = np.stack([o.attentions[L][0, :, -1, base:base+nt].float().mean(0).cpu().numpy() for L in range(nL)])
            del o; torch.cuda.empty_cache(); maps.append(A/np.maximum(A.sum(1, keepdims=True), 1e-12))
        if len(maps) < 2: continue
        m = min(x.shape[1] for x in maps)
        for L in range(nL):
            ds = []
            for i in range(len(maps)):
                for j in range(i+1, len(maps)):
                    a, b = maps[i][L][:m], maps[j][L][:m]; ds.append(1.0-float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)))
            div[L].append(float(np.mean(ds)))
        if (n+1) % 10 == 0: print(f"    [{n+1}/{len(items)}]", flush=True)
    del model; torch.cuda.empty_cache()
    return [float(np.mean(d)) if d else float("nan") for d in div]

def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset"); ds = load_dataset("craigwu/vstar_bench")["test"]
    rows = [e for e in ds if os.path.exists(os.path.join(root, e["image"]))]; qs_all = [e["text"] for e in rows]
    items = [(os.path.join(root, e["image"]), [e["text"]]+[qs_all[(i*37+j*91) % len(qs_all)] for j in range(1, K_Q)]) for i, e in enumerate(rows[:N_IMG])]
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, mid in MODELS:
        if tag in out: print(f"  {tag}: cached"); continue
        out[tag] = run(tag, mid, items); json.dump(out, open(OUT, "w"), indent=1)
        d = out[tag]; print(f"  {tag}: " + " ".join(f"L{i}:{v:.3f}" for i, v in enumerate(d)), flush=True)
    print("wrote", OUT)
main()
