"""
M1 follow-up (GPU): after the alignment training, where does the read-out depth curve peak?

phase 199 measured the base model's end-task accuracy when cropping at each layer's arg-max: it peaks at L17
(70.2%) and every layer inside the transport window is below no-crop. If the training moved the localization
signal into L10-16, the curve should now have its usable band inside the window. Measured on the M1 held-out
items (idx % 4 == 3, n=47); the base curve for the same items is read from phase199_layersweep_qwen3.jsonl.
"""
import json, os, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
ADAPTER = f"{D}/models/m1_lora_qwen3"; OUT = f"{D}/data/m1_sweep.json"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"; W = 0.25; B0 = 300; NL = 28
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
    opt = [sorted({tok(x, add_special_tokens=False)["input_ids"][-1] for x in [c, f" {c}"]}) for c in "ABCD"]

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
    def win(img, cx, cy, w):
        iw, ih = img.size; x0 = min(max(0, (cx - w / 2) * iw), iw - w * iw); y0 = min(max(0, (cy - w / 2) * ih), ih - w * ih)
        return img.crop((int(x0), int(y0), int(x0 + w * iw), int(y0 + w * ih)))
    def answer(im, t):
        inp = build(fit(im, B0), t).to(model.device)
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)
        del inp; torch.cuda.empty_cache(); return [round(float(v), 6) for v in p.tolist()]

    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    recs = []
    for idx, ex in enumerate(ds):
        if idx % 4 != 3: continue
        ip = os.path.join(root, ex["image"])
        if not os.path.exists(ip): continue
        img = Image.open(ip).convert("RGB"); sm = fit(img)
        inp = build(sm, ex["text"]).to(model.device)
        with torch.no_grad(): out = model(**inp, output_attentions=True)
        pos = (inp["input_ids"][0] == itid).nonzero().flatten(); base, nt = int(pos[0]), int(len(pos))
        g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1] // 2, g[2] // 2
        rm = np.zeros((gh, gw), bool); rm[1:-1, 1:-1] = True; rm = rm.ravel()
        rec = {"qid": f"{ex['category']}/{ex['question_id']}", "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]), "probs": {}}
        for l in range(NL):
            A = out.attentions[l][0, :, -1, base:base + nt].float().mean(0).cpu().numpy()
            A = A / max(A.sum(), 1e-12)
            j = int(np.argmax(np.where(rm, A, -1e9)))
            cx, cy = (j % gw + .5) / gw, (j // gw + .5) / gh
            rec["probs"][f"L{l}"] = answer(win(img, cx, cy, W), ex["text"])
        del out, inp; torch.cuda.empty_cache()
        recs.append(rec)
        if len(recs) % 10 == 0: print(f"  [{len(recs)}] ", flush=True)
    json.dump(recs, open(OUT, "w"), indent=1)
    # compare with the base curve on the same items
    base = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(f"{D}/data/phase199_layersweep_qwen3.jsonl")}
    qs = [r["qid"] for r in recs if r["qid"] in base]
    for tag, src in [("base", base), ("adapter", {r["qid"]: r for r in recs})]:
        acc = []
        for l in range(NL):
            a = [int(np.argmax(src[q]["probs"][f"L{l}"]) == src[q]["label"]) for q in qs]
            acc.append(100 * np.mean(a))
        b = int(np.argmax(acc))
        print(f"  {tag}: best L{b} = {acc[b]:.1f}%; window max {max(acc[:16]):.1f}; after max {max(acc[16:]):.1f}")
        print(f"    curve: " + " ".join(f"{v:.0f}" for v in acc), flush=True)
    print(f"wrote {OUT}", flush=True)


main()
