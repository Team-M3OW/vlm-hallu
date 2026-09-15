"""
Phase 29: reconciling our V*Bench numbers with ViRGo's (BLOCKING item B1).

THE PROBLEM
-----------
ViRGo (2606.21968) reports, on **Qwen3-VL-2B, V*Bench, all 191 items** -- our exact setup:
    global perception (baseline)   64.2%
    RAP (patch retrieval)          78.9%
We report `uniform@2400` 75.4% and `crop_only@300` **93.7%**. A reviewer who knows ViRGo will read
93.7% as inflated, and they would be right to ask. The gap has (at least) TWO components and they
must be separated before either number is written:

  1. **Oracle premium.** Their RAP is a training-free PROPOSER; our crop is the benchmark's own
     ground-truth box. This is real and is exactly why section 9's oracle caveat is load-bearing.
  2. **Readout offset.** We score 4-way MCQ by argmax over the pooled {A,B,C,D} logits at the final
     position. ViRGo follows each method's original protocol, i.e. **generating** an answer and
     matching it. Constrained argmax cannot produce an unparseable answer; free generation can.
     That alone inflates every arm, baseline included.

THIS SCRIPT MEASURES (2) SO (1) CAN BE STATED HONESTLY.
Same items, same images, same budgets, same model -- only the readout changes:
    logit_argmax   pooled logits over {A,B,C,D} (what every result of ours uses)
    generate_em    greedy generation, then exact-match of the first option letter found;
                   **unparseable output counts as WRONG**, which is the whole point
Arms: `uniform@2400` (≈ the native operating point) and `crop_only@300`.

If the offset is large, every V*Bench number in the paper needs the readout stated next to it, and
the comparison to ViRGo must be made on the generation readout, not ours.
"""
import json, os, re, sys, time
import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase17_budget_allocation import fit_to_budget
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase29_readout_results.jsonl"
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    import random
    rng = random.Random(107)
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}).eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]

    def chat(text):
        return pr.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)

    def measure(imgs, text="x"):
        i = pr(images=imgs, text=chat(text), return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    def both_readouts(img, text):
        i = pr(images=[img], text=chat(text), return_tensors="pt").to(model.device)
        rz = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        argmax_pred = int(p.argmax().item())
        with torch.no_grad():
            g = model.generate(**i, max_new_tokens=12, do_sample=False)
        txt = tok.decode(g[0][i["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r"\b([ABCD])\b", txt)
        gen_pred = "ABCD".index(m.group(1)) if m else -1      # -1 = unparseable = WRONG
        return argmax_pred, gen_pred, txt, rz

    done = set()
    try:
        for l in open(OUT):
            done.add(json.loads(l)["qid"])
    except FileNotFoundError:
        pass

    t0, n = time.time(), 0
    with open(OUT, "a") as fo:
        for ex in ds:
            rel = ex["image"]; qid = f"{rel}::{ex['question_id']}"
            ip = os.path.join(root, rel); ap = os.path.splitext(ip)[0] + ".json"
            if not (os.path.exists(ip) and os.path.exists(ap)) or qid in done:
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox"):
                continue
            img = Image.open(ip).convert("RGB"); W, H = img.size
            x0 = min(b[0] for b in ann["bbox"]); y0 = min(b[1] for b in ann["bbox"])
            x1 = max(b[0]+b[2] for b in ann["bbox"]); y1 = max(b[1]+b[3] for b in ann["bbox"])
            pw, ph = (x1-x0)*PAD, (y1-y0)*PAD
            cb = (max(0, x0-pw), max(0, y0-ph), min(W, x1+pw), min(H, y1+ph))
            if cb[2]-cb[0] < 8 or cb[3]-cb[1] < 8:
                continue
            crop = img.crop(tuple(int(v) for v in cb))
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]
            rec = {"qid": qid, "category": ex["category"], "label": label, "arms": {}}
            for nm, src, B in [("uniform", img, 2400), ("crop_only", crop, 300)]:
                f, _ = fit_to_budget(lambda ims: measure(ims, text), [src], B)
                a, gp, raw, rz = both_readouts(f[0], text)
                rec["arms"][nm] = {"logit_argmax": a, "generate_em": gp,
                                   "raw": raw[:40], "tokens": rz}
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            n += 1
            if n % 25 == 0:
                print(f"[{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
