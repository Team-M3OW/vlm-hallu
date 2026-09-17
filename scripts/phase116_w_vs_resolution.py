"""
Phase 116: does the optimal crop width scale INVERSELY with image resolution?

Phase 115 refuted my own diagnostic. V*Bench's budget axis is NOT flat -- it is close to linear
(+8.0, +7.3, +8.7 per doubling from 150 to 1200 tokens) -- so "allocation wins where the budget axis
has saturated" is wrong, and wrong about the very benchmark it was meant to explain. V*Bench
(+7.3/doubling) and HR-Bench 4K (~+9/doubling) have SIMILAR slopes, yet we win on one and lose by
12.4pp on the other. Slope cannot be the discriminator.

THE REPLACEMENT ACCOUNT, and the reason it is testable rather than another story: a crop only
delivers magnification if it is re-encoded at (or below) the resolution the token budget can carry.

    a 300-token view resolves roughly 485 x 485 px
    V*Bench   ~1800px image, W=0.25 -> 450px crop  -> fully resolved      (and W*=0.25 is what
                                                                          phase 90 measured)
    HR-Bench  4032px image, W=0.25 -> 1008px crop  -> still 2x undersampled

    PREDICTION:  W* ~= (resolution the budget resolves) / image_side
                 V*Bench   485/1800 ~= 0.27   -- measured W* = 0.25  ✔ (already on the record)
                 HR-Bench  485/4032 ~= 0.12   -- PRE-REGISTERED HERE, never measured

PRE-REGISTERED READING
    argmax@W peaks near W ~= 0.12 on HR-Bench   -> the resolution account holds, and one inequality
                                                   explains V*Bench, HR-Bench and POPE together
    peak at W = 0.25, or no peak                -> the account is refuted and the 4K failure needs a
                                                   different explanation
    a peak elsewhere                            -> report the measured W* against the predicted one;
                                                   the formula is falsified but the trend may survive

HR-Bench ships no boxes, so there is no oracle arm and no coverage metric here: end-task accuracy
against the equal-compute bar is the only outcome, which is exactly the quantity in dispute.
Also completes phase 115's slope table, which HR-Bench was skipped from on a cache-config mismatch.
"""
import base64, glob, json, os, random, time
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
import io

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
OUT = f"{D}/data/phase116_w_resolution_hrbench.jsonl"
WS = [0.06, 0.09, 0.12, 0.15, 0.25]
B0, NSEL = 300, 150
BLOCK = list(range(16, 27))
Image.MAX_IMAGE_PIXELS = None


def main():
    f = glob.glob(f"{os.environ['HF_HUB_CACHE']}/datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet")[0]
    t = pq.read_table(f)
    rows = []
    for i in range(t.num_rows):
        r = {c: t.column(c)[i].as_py() for c in t.column_names}
        q = r["question"] + "\n" + "\n".join(f"({c}) {r[c]}" for c in "ABCD") + \
            "\nAnswer with the option's letter from the given choices directly."
        rows.append({"q": q, "label": "ABCD".index(str(r["answer"]).strip()[0]),
                     "cat": r["category"], "img": r["image"], "idx": r["index"]})
    rng = random.Random(116); sel = rng.sample(rows, min(NSEL, len(rows)))
    print(f"HR-Bench 4k: {len(rows)} rows, sampled {len(sel)}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager"); model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in [c, f" {c}"]}) for c in "ABCD"]
    img_tok_id = model.config.image_token_id

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
    def window(img, cx, cy, w):
        iw, ih = img.size
        x0 = min(max(0, (cx-w/2)*iw), iw-w*iw); y0 = min(max(0, (cy-w/2)*ih), ih-w*ih)
        return img.crop((int(x0), int(y0), int(x0+w*iw), int(y0+w*ih)))
    def ans(img, text):
        inp = build(img, text); rz = int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist()))
        inp = inp.to(model.device)
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt_ids]), 0)
        return int(p.argmax()), rz
    def localize(img, text):
        small, _ = fit(img, B0); inp = build(small, text)
        g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1]//2, g[2]//2; n = gh*gw
        pos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten(); base = int(pos[0].item())
        inp = inp.to(model.device)
        with torch.no_grad(): out = model(**inp, output_attentions=True)
        acc = torch.zeros(n, device=model.device)
        for L in BLOCK:
            a = out.attentions[L][0, :, -1, base:base+n].float().mean(0); acc += a/a.sum().clamp_min(1e-12)
        del out
        acc = (acc/len(BLOCK)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else: m = acc.clone()
        i = int(m.argmax().item())
        return ((i % gw)+.5)/gw, ((i//gw)+.5)/gh

    t0 = time.time()
    with open(OUT, "w") as fout:
        for k, r in enumerate(sel):
            raw = r["img"]
            if isinstance(raw, dict): raw = raw["bytes"]
            if isinstance(raw, str): raw = base64.b64decode(raw)   # HR-Bench ships base64 strings
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            cx, cy = localize(img, r["q"])
            rec = {"idx": r["idx"], "cat": r["cat"], "label": r["label"],
                   "wh": list(img.size), "mp": round(img.size[0]*img.size[1]/1e6, 3),
                   "cell": [cx, cy], "pred": {}, "tok": {}}
            arms = {f"uniform@{b}": fit(img, b)[0] for b in [150, 300, 600, 1200]}
            for w in WS: arms[f"argmax@{w}"] = fit(window(img, cx, cy, w), B0)[0]
            rr = random.Random(hash(str(r["idx"])) % 99999)
            arms["rand@0.12"] = fit(window(img, rr.uniform(.1,.9), rr.uniform(.1,.9), 0.12), B0)[0]
            for nm, im in arms.items():
                a, rz = ans(im, r["q"]); rec["pred"][nm] = a; rec["tok"][nm] = rz
            fout.write(json.dumps(rec)+"\n"); fout.flush()
            if (k+1) % 20 == 0:
                el = time.time()-t0
                print(f"  [{k+1}/{len(sel)}] {(k+1)/el:.2f} it/s eta={(len(sel)-k-1)/max((k+1)/el,1e-9)/60:.1f}min", flush=True)
            torch.cuda.empty_cache()
    print("Done ->", OUT, flush=True)


main()
