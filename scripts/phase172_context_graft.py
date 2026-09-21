"""
Phase 172 (training-free import #3): give the crop its CONTEXT back.

WHY
---
DPR crops and throws the surrounding scene away. Two 2026 results say that is exactly what breaks
relational questions:
  * "Grounding Isn't Knowing" (arXiv 2608.23074, MBZUAI, 2026-08-24): ablating object-interior tokens
    costs up to 64.9 IoU of localisation but AT MOST 1.32 points of spatial-relation accuracy, while
    dilating the mask into the SURROUNDING context is what degrades relations. Relations run on
    coarse layout, not object detail.
  * "Thinking Once Is Enough" (arXiv 2607.27830, 2026-07-30): keeps question-relevant tokens at full
    resolution PLUS a pooled summary of everything else; removing that background summary costs them
    3.3 points, and their gains concentrate on relation-sensitive metrics (+4.2 spatial on V*Bench).
Our own numbers agree that this is the failing stratum: DPR is -3.9 / +0.0 on relational vs the bar.

THE GRAFT
---------
Not their intermediate-layer routing (that is their method, not a graft). The budget-legal analogue
is TWO IMAGES in the answer pass: the crop, plus a downscaled full scene, TOTALLING the same 300
tokens the crop-only arm spends. Bug #21's 64-merged-token floor per sub-image makes exactly two
sub-images affordable at 300 where `anyres@300`'s five were not.

    ctx64    crop fitted to 236 + scene fitted to  64   (scene at the floor)
    ctx128   crop fitted to 172 + scene fitted to 128
    total answer-pass budget 300 in both, so method = localise@300 + answer@300 = 600 = the bar.

Related repo work, cited not repeated: phase 21 (SS4I) swept a 25/75 crop:scene split and found it beat
50/50 in the scarcest stratum -- but at ORACLE placement, on RePOPE strata, before the head existed.
This is that knob at the head's own OOF proposal, on V*Bench, against the equal-compute bar.

PRE-REGISTERED
    PRIMARY  ctx - bar on RELATIONAL questions, CI clear of zero on BOTH Qwen models.
             (This is the cell DPR loses today and the one the skip in SS18E can only draw.)
    GUARD    ctx - head@0.25 on SINGLE-OBJECT must not be significantly negative: the context tokens
             are paid for out of the crop's resolution, and SS13B says that is where the method lives.
    NOTE     the oracle@0.25 - bar = +9.2 [-5.3,+23.7] bound on relational does NOT apply here. That
             oracle is a TIGHT crop; adding context back is a different intervention, not a better
             placement of the same one.
    Image order (crop, then scene) and the connector sentence are FIXED and unswept.
Proposals are read from phase71a/phase80a (OOF, grouped by item). No fitting happens in this file.
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/phase80a_qwen2vl_proposals.json")}
MODEL_ID, PROP = CFG[WHICH]
OUT = f"{D}/phase172_context_{WHICH}.jsonl"
B0, W = 300, 0.25
CONN = "Here is the full image that the crop above was taken from."
SPLITS = {"ctx64": (236, 64), "ctx128": (172, 128)}
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt = [sorted({tok(c, add_special_tokens=False)["input_ids"][-1] for c in [C, f" {C}"]})
           for C in "ABCD"]

    def chat(k, text):
        c = [{"type": "image"}]
        for _ in range(k - 1):
            c += [{"type": "text", "text": CONN}, {"type": "image"}]
        c.append({"type": "text", "text": text})
        return pr.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                      add_generation_prompt=True)

    def build(imgs, text):
        return pr(images=imgs, text=chat(len(imgs), text), return_tensors="pt")

    def measure(img):
        return int(sum(g[1]*g[2]//4 for g in build([img], "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=6, tol=0.06):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r) ** 0.5
        return best[0]

    def window(img, cx, cy, w):
        iw, ih = img.size
        x0 = min(max(0, (cx - w/2)*iw), iw - w*iw); y0 = min(max(0, (cy - w/2)*ih), ih - w*ih)
        return img.crop((int(x0), int(y0), int(x0 + w*iw), int(y0 + w*ih)))

    def answer(imgs, text):
        i = build(imgs, text)
        rz = int(sum(g[1]*g[2]//4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        assert torch.isfinite(lg).all(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    byq = {f"{e['category']}/{e['question_id']}": e for e in ds}
    t0, done = time.time(), 0
    with open(OUT, "w") as fo:
        for qid, pinfo in props.items():
            ex = byq.get(qid)
            if ex is None: continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            cx, cy = pinfo["head"]          # OOF head proposal, [cx, cy] in image fractions
            crop = window(img, cx, cy, W)
            probs, tokens = {}, {}
            for nm, im in [("uniform@300", [fit(img, B0)]), ("uniform@600", [fit(img, 2*B0)]),
                           ("head@0.25", [fit(crop, B0)])]:
                probs[nm], tokens[nm] = answer(im, ex["text"])
            for nm, (bc, bs) in SPLITS.items():
                probs[nm], tokens[nm] = answer([fit(crop, bc), fit(img, bs)], ex["text"])
            lab = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            fo.write(json.dumps({"question_id_full": qid, "category": ex["category"], "label": lab,
                                 "probs": probs, "realized_tokens": tokens}) + "\n")
            done += 1
            if done % 25 == 0:
                print(f"  [{done}] {done/(time.time()-t0):.2f} it/s", flush=True)
    print(f"wrote {OUT}  n={done}")


if __name__ == "__main__":
    main()
