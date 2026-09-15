"""
Phase 64: LOGIT-LENS-GUIDED COMPONENT ABLATION. Which part of the last layers destroys the answer?

THE MOTIVATION
--------------
§12A/§12B: decoding at L24 beats the final layer by ~+4pp on sub-token items, because the last
layers trade correct visual evidence for a default answer when evidence is weak. But truncation is
BLUNT -- it discards all of L25-27, including the part that HELPS: on the oracle crop with resolvable
targets the final layers IMPROVE accuracy by 1.8pp. A method that truncates would give that up.

Each decoder layer makes TWO residual updates:
        h <- h + attn(norm1(h));   h <- h + mlp(norm2(h))
So the lens says WHERE the damage happens (L25-27) and component ablation says WHAT does it. Zeroing
one sublayer's output removes exactly that update and leaves everything else intact.

    one component accounts for the damage -> ablate only it: keep the helpful part of the late
        layers, recover the harmful part, and beat truncation on BOTH strata.
    the damage is distributed -> truncation is already the right blunt instrument and nothing
        finer is available. Report that.

PRIOR ART, checked: Anchored Answers (2405.03205) localises MLP value vectors and attention heads
causing 'A'-bias in GPT-2 MCQ -- text-only. DeCo (2410.11779) ADDS an anchor layer's logits rather
than ablating a component. Component ablation in a VLM, conditioned on visual-evidence strength, is
the part not covered by either.

ARMS (all one pass, identical tokens -- ablation changes no compute)
    baseline                      no intervention
    attn@L{25,26,27}              zero that layer's attention update
    mlp@L{25,26,27}               zero that layer's MLP update
    attn@all_late / mlp@all_late  zero all three
    trunc_L24                     the §12B baseline, for reference
Run on BOTH the uniform image and the oracle crop, so the evidence-dependence can be read directly:
a component that is harmful only under weak evidence is the one we want.
"""
import json
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase64_component_ablation.jsonl"
B0, PAD = 300, 0.25
LATE = [25, 26, 27]
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
    layers = model.model.language_model.layers
    lm, norm = model.get_output_embeddings(), model.model.language_model.norm
    nL = len(layers)
    OFF = {"attn": set(), "mlp": set()}

    def mk(kind, idx):
        def hook(mod, args, out):
            if idx not in OFF[kind]:
                return out
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + tuple(out[1:])
            return torch.zeros_like(out)
        return hook

    for i, l in enumerate(layers):
        l.self_attn.register_forward_hook(mk("attn", i))
        l.mlp.register_forward_hook(mk("mlp", i))
    print(f"{nL} layers hooked (attn + mlp); ablating within {LATE}", flush=True)

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

    def run(inp, a_off=(), m_off=(), read=-1):
        OFF["attn"], OFF["mlp"] = set(a_off), set(m_off)
        try:
            with torch.no_grad():
                o = model(**inp, output_hidden_states=(read != -1))
                h = o.hidden_states[read + 1][0, -1, :] if read != -1 else None
                lg = (lm(norm(h)).float() if read != -1 else o.logits[0, -1].float())
        finally:
            OFF["attn"], OFF["mlp"] = set(), set()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt_ids]), 0)
        return [round(float(v), 5) for v in p.tolist()]

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
                   "tokens_on_target": area * B0, "probs": {}}
            for cond, im in (("uniform", fit(img, B0)),
                             ("oracle", fit(img.crop(tuple(int(v) for v in ob)), B0))):
                inp = build(im, text).to(model.device)
                d = {"baseline": run(inp)}
                for L in LATE:
                    d[f"attn@L{L}"] = run(inp, a_off=(L,))
                    d[f"mlp@L{L}"] = run(inp, m_off=(L,))
                d["attn@late"] = run(inp, a_off=tuple(LATE))
                d["mlp@late"] = run(inp, m_off=tuple(LATE))
                d["both@late"] = run(inp, a_off=tuple(LATE), m_off=tuple(LATE))
                d["trunc_L24"] = run(inp, read=24)
                rec["probs"][cond] = d
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  [{n}/191] eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
