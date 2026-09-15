"""
Phase 60: LOGIT LENS. Is the answer ABSENT under uniform, or present but not read out?

THE GAP THIS FILLS
------------------
§10A-C concluded that when the target is sub-token the information is ABSENT from the
representation, not merely mis-weighted -- suppressing the sink does nothing, amplifying the target
recovers 16% of the crop, and the transferable residual direction gives +0.0pp. But all three are
INTERVENTIONS: they change behaviour and infer what must be encoded. None READS the representation.

The logit lens reads it directly, and it can falsify §10:

    if the correct option is decodable at some intermediate layer under `uniform` but lost by the
    output layer, the deficit is a READ-OUT failure, not an information failure. §10's conclusion
    would be wrong, and a best-layer read-out would be a free method (one pass, no crop).

    if the correct option is never decodable at any layer under `uniform`, while it IS decodable
    under the oracle crop, §10's conclusion is confirmed by direct measurement rather than by
    inference from nulls -- which is the stronger form of the same claim.

METHOD
------
For each item, run the SAME prompt twice at 300 tokens -- uniform image, and oracle crop (93%) --
and capture the LAST TOKEN's residual at EVERY layer. Project each through the model's own final
norm and unembedding, restrict to the A/B/C/D token ids, and score. This yields an accuracy-by-layer
curve for both conditions.

    uniform_best_layer - uniform_final   > 0  -> recoverable signal is being discarded late
    oracle curve rises where uniform's does not -> the crop ADDS information, as §10 claims

The sub-token stratum is reported separately, since that is where §10's claim is specifically about.
No hyper-parameter is fitted; the layer curve is simply read off.
"""
import json
import os
import statistics as st
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"
B0, PAD = 300, 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    lm = model.get_output_embeddings()
    norm = model.model.language_model.norm
    nL = model.config.get_text_config().num_hidden_layers
    print(f"{nL} layers; logit lens = final norm + unembedding applied to each layer's output",
          flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    def lens(img, text):
        """ABCD distribution at EVERY layer, from the last token's residual."""
        i = build(img, text).to(model.device)
        with torch.no_grad():
            o = model(**i, output_hidden_states=True)
            out = []
            for li in range(1, nL + 1):                 # hidden_states[i+1] = output of layer i
                h = o.hidden_states[li][0, -1, :]
                lg = lm(norm(h)).float()
                p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
                out.append([round(float(v), 5) for v in p.tolist()])
        del o
        torch.cuda.empty_cache()
        return out

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

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
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            dx, dy = (gx1 - gx0) * PAD, (gy1 - gy0) * PAD
            ob = (max(0, gx0 - dx), max(0, gy0 - dy), min(IW, gx1 + dx), min(IH, gy1 + dy))
            text = ex["text"]
            area = ((gx1 - gx0) * (gy1 - gy0)) / (IW * IH)
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str)
                   else int(ex["label"]),
                   "gt_area_frac": area, "tokens_on_target": area * B0,
                   "uniform": lens(fit(img, B0), text),
                   "oracle": lens(fit(img.crop(tuple(int(v) for v in ob)), B0), text)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
