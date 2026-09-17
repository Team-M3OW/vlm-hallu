"""
Phase 115: MEASURE THE BUDGET-AXIS SLOPE, so benchmark choice stops being a judgement call.

Allocation methods re-spend a fixed token budget. Whether that can beat simply spending more tokens
depends on a property of the BENCHMARK, not of the method: how much accuracy a doubling of the visual
budget buys. Where the curve is steep, no placement policy can compete; where it has gone flat,
placement is the only lever left.

We already have both extremes by accident:
    V*Bench      doubling buys little -- our method wins here
    HR-Bench 4K  +18.2pp for 4x tokens (SS9D/phase 59) -- every allocation method loses here,
                 ours by 6.2pp on single-region, Zoom Eye by 15.3pp
    POPE         uniform@300 is already ~100% above the cliff -- saturated AND no magnification
                 headroom (COCO 640x480 is ~1.3x a 300-token view)

This measures the slope directly on every benchmark we can reach, with NO method involved and no
boxes required, so it also applies to benchmarks that ship no annotations.

REPORTED PER BENCHMARK
    accuracy at uniform@{150,300,600,1200}
    slope 300->600 in pp per doubling   <- the doubling our method is charged for
    magnification headroom  = median image area / the area a 300-token view resolves (~0.235 MP)

PRE-REGISTERED READING
    slope near zero  -> allocation CAN win; a null there is about the method
    slope large      -> allocation CANNOT win; a loss there is about the benchmark
This makes "which benchmark" a measurement, and predicts where the whole family of
attention-reweighting and cropping methods can help -- something none of VEA, Zoom Eye, ViCrop or
AttWarp reports.
"""
import json, os, random, sys, time
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ["HF_DATASETS_OFFLINE"] = "1"
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
OUT = f"{D}/data/phase115_budget_slope.jsonl"
BUDGETS = [150, 300, 600, 1200]
NPB = 150
Image.MAX_IMAGE_PIXELS = None


def load_benches():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    B = []
    try:
        root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
        ds = load_dataset("craigwu/vstar_bench")["test"]
        rows = [{"img": os.path.join(root, e["image"]), "text": e["text"],
                 "label": "ABCD".index(e["label"]) if isinstance(e["label"], str) else int(e["label"]),
                 "kind": "mcq4"} for e in ds]
        B.append(("VStar", rows))
    except Exception as e: print("VStar skip:", str(e)[:80], flush=True)
    try:
        ds = load_dataset("DreamMr/HR-Bench", "hrbench_4k")["test"]
        rows = []
        for e in ds:
            opts = [e.get(f"{c}") for c in "ABCD"]
            q = e["question"] + "\n" + "\n".join(f"({c}) {o}" for c, o in zip("ABCD", opts)) + \
                "\nAnswer with the option's letter from the given choices directly."
            rows.append({"pil": e["image"], "text": q, "label": "ABCD".index(str(e["answer"]).strip()[0]),
                         "kind": "mcq4"})
        B.append(("HRBench4k", rows))
    except Exception as e: print("HRBench skip:", str(e)[:80], flush=True)
    try:
        ds = load_dataset("lmms-lab/POPE", "default", split="test")
        rows = [{"pil": e["image"], "text": e["question"],
                 "label": int(str(e["answer"]).lower().startswith("y")), "kind": "yn"} for e in ds]
        B.append(("POPE", rows))
    except Exception as e: print("POPE skip:", str(e)[:80], flush=True)
    try:
        ds = load_dataset("lmms-lab/MMBench_EN", "default", split="dev")
        rows = []
        for e in ds:
            opts = [e.get(c) for c in "ABCD"]
            if any(o is None for o in opts): continue
            q = e["question"] + "\n" + "\n".join(f"({c}) {o}" for c, o in zip("ABCD", opts)) + \
                "\nAnswer with the option's letter from the given choices directly."
            rows.append({"pil": e["image"], "text": q, "label": "ABCD".index(str(e["answer"]).strip()[0]),
                         "kind": "mcq4"})
        B.append(("MMBench", rows))
    except Exception as e: print("MMBench skip:", str(e)[:80], flush=True)
    return B


def main():
    benches = load_benches()
    print("benchmarks:", [(n, len(r)) for n, r in benches], flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}); model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [c, f" {c}"]}) for c in "ABCD"]
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in ["Yes"," Yes","yes"," yes"]})
    no_ids  = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in ["No"," No","no"," no"]})

    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
                  return_tensors="pt")
    def measure(img): return int(sum(g[1]*g[2]//4 for g in build(img, "x")["image_grid_thw"].tolist()))
    def fit(img, target, refine=5, tol=0.06):
        W_, H_ = img.size; sc = (target/max(measure(img), 1))**0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r)**0.5
        return best
    def ans(img, text, kind):
        inp = build(img, text)
        rz = int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist()))
        inp = inp.to(model.device)
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        ids = opt_ids if kind == "mcq4" else [yes_ids, no_ids]
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in ids]), 0)
        return int(p.argmax()), rz

    with open(OUT, "w") as fout:
        for name, rows in benches:
            rng = random.Random(115); sel = rng.sample(rows, min(NPB, len(rows)))
            t0 = time.time()
            for k, r in enumerate(sel):
                img = r["pil"].convert("RGB") if "pil" in r else Image.open(r["img"]).convert("RGB")
                rec = {"bench": name, "kind": r["kind"], "label": r["label"],
                       "mp": round(img.size[0]*img.size[1]/1e6, 4), "acc": {}, "tok": {}}
                for b in BUDGETS:
                    im, _ = fit(img, b)
                    a, rz = ans(im, r["text"], r["kind"])
                    # for yes/no, index 0 = yes; label 1 = yes
                    ok = int(a == r["label"]) if r["kind"] == "mcq4" else int((a == 0) == (r["label"] == 1))
                    rec["acc"][str(b)] = ok; rec["tok"][str(b)] = rz
                fout.write(json.dumps(rec)+"\n"); fout.flush()
                if (k+1) % 25 == 0:
                    el = time.time()-t0
                    print(f"  {name} [{k+1}/{len(sel)}] {(k+1)/el:.2f} it/s "
                          f"eta={(len(sel)-k-1)/max((k+1)/el,1e-9)/60:.1f}min", flush=True)
                torch.cuda.empty_cache()
            print(f"{name} done", flush=True)
    print("Done ->", OUT, flush=True)


main()
