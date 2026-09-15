"""
Phase 62: does the L24 read-out compose with multi-crop? Both are free; are their gains additive?

§11A: multi-crop adds +7.9pp over single-crop at equal cost (V*Bench) and +8.0pp at 4K.
§12B: reading at L24 instead of the final layer adds +5.1pp on sub-token items at 11% less compute.
They act on different things -- one supplies information, the other stops it being discarded -- so
they should compose. "Should" is not evidence; this measures it.

Captures per-LAYER ABCD distributions for uniform@300, top1@0.15 and multi4 on the same items, so
every (arm x read-out layer) cell is comparable and the interaction can be read directly.
"""
import json
import math
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase62_combine.jsonl"
BLOCK = list(range(16, 27))
B0, W, MIN_SEP = 300, 0.15, 0.20
CONN = "Here is another zoomed-in crop from the same image."
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    lm = model.get_output_embeddings()
    norm = model.model.language_model.norm
    nL = model.config.get_text_config().num_hidden_layers

    def chat(k, text):
        c = [{"type": "image"}]
        for _ in range(k - 1):
            c += [{"type": "text", "text": CONN}, {"type": "image"}]
        c.append({"type": "text", "text": text})
        return pr.apply_chat_template([{"role": "user", "content": c}],
                                      tokenize=False, add_generation_prompt=True)

    def build(imgs, text):
        return pr(images=imgs, text=chat(len(imgs), text), return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build([img], "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=6, tol=0.06):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    def lens(imgs, text):
        i = build(imgs, text).to(model.device)
        with torch.no_grad():
            o = model(**i, output_hidden_states=True)
            out = []
            for li in range(1, nL + 1):
                h = o.hidden_states[li][0, -1, :]
                lg = lm(norm(h)).float()
                p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
                out.append([round(float(v), 5) for v in p.tolist()])
        del o
        torch.cuda.empty_cache()
        return out

    def candidates(img, text, k):
        small = fit(img, B0)
        inp = build([small], text)
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1] // 2, g[2] // 2
        n_img = gh * gw
        base = int((inp["input_ids"][0] == itid).nonzero().flatten()[0].item())
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, dtype=torch.float32, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
            acc += a / (a.sum() + 1e-12)
        del out
        torch.cuda.empty_cache()
        acc = (acc / len(BLOCK)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2:
            m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else:
            m = acc.clone()
        f = m.flatten()
        pts = []
        for i in torch.argsort(f, descending=True).tolist():
            if f[i].item() < 0:
                break
            px, py = ((i % gw) + .5) / gw, ((i // gw) + .5) / gh
            if all(math.hypot(px - x, py - y) >= MIN_SEP for x, y in pts):
                pts.append((px, py))
            if len(pts) == k:
                break
        while len(pts) < k:
            pts.append(pts[0])
        return pts

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}; {nL} layers", flush=True)

    n, t0 = 0, time.time()
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"
            ip = os.path.join(root, ex["image"])
            ap = os.path.splitext(ip)[0] + ".json"
            if not os.path.exists(ap) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB")
            IW, IH = img.size
            area = ((max(b[0]+b[2] for b in ann["bbox"]) - min(b[0] for b in ann["bbox"])) *
                    (max(b[1]+b[3] for b in ann["bbox"]) - min(b[1] for b in ann["bbox"]))) / (IW*IH)
            text = ex["text"]
            pts = candidates(img, text, 4)
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str)
                   else int(ex["label"]),
                   "tokens_on_target": area * B0,
                   "uniform": lens([fit(img, B0)], text),
                   "top1": lens([fit(window(img, *pts[0], W), B0)], text),
                   "multi4": lens([fit(window(img, *pts[i], W), B0 // 4) for i in range(4)], text)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
