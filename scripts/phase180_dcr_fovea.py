"""
Phase 180: DPR-FOVEA (the repo name; DCR was the earlier name for the same method) -- training-free. Let the ROI decide WHERE resolution goes instead of deleting
the rest of the image.

    current DCR :  (I,q) -> ROI -> crop only        -> re-encode -> answer
    DCR-Fovea   :  (I,q) -> ROI -> coarse global scene + high-res ROI -> answer

ARCHITECTURE NOTE (inspected, not assumed)
------------------------------------------
Qwen2-VL / Qwen3-VL take multiple images natively in one turn: the chat template emits one
placeholder per image and the processor returns one `image_grid_thw` row per image; MRoPE assigns
each image its own 2-D position block. So the clean architecture-compatible form is two IMAGES, not
concatenated internal vision tokens and no projector surgery:

    Image 1 = full scene, deterministically downscaled  (global context, uniform spatial coverage)
    Image 2 = the DCR ROI at high resolution            (the fovea)

HARD CONSTRAINT -- bug #21: every image costs >= 64 merged tokens regardless of pixel size. With a
300-token answer budget the requested global/local splits clamp to the closest valid pair:

    requested 83/17 -> 236/64   (17% of 300 = 51 < 64 floor; 64 is the closest valid)
    requested 75/25 -> 225/75
    requested 67/33 -> 200/100
    requested 50/50 -> 150/150

BUDGET. Method = localise@300 (DCR's existing first pass) + answer@300 = 600 = the `uniform@600`
equal-compute bar the repo already uses. Token counts are MEASURED from `image_grid_thw`, summed over
both images, never computed; >10% drift voids a contrast.

TRAINING-FREE, and nothing here is fitted: weights frozen, no LoRA, no learned fusion, no learned
pruning, no DCR retraining. The ROI is read from the EXISTING out-of-fold proposals (phase 71a /
80a). Global token reduction is a deterministic bicubic resize of the whole scene calibrated to a
measured token count -- uniform spatial coverage, no token selection of any kind.

ARMS
    uniform@300                    the unaided model
    uniform@600                    EQUAL-COMPUTE BAR
    dcr_crop                       the existing DCR arm (crop only, W=0.25 @300)
    fovea_{83_17,75_25,67_33,50_50}  scene + ROI at the four splits
    oracle_crop, oracle_fovea_*    the same with the GT-box centre instead of the DCR ROI
                                   -> separates "localisation is the bottleneck" from "representation is"

usage: phase180_dcr_fovea.py <qwen3|qwen2> [--limit N] [--viz]
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
LIMIT = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else None
VIZ = "--viz" in sys.argv
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/data/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/data/phase80a_qwen2vl_proposals.json")}
MODEL_ID, PROP = CFG[WHICH]
TAG = "sanity" if LIMIT else "full"
OUT = f"{D}/data/phase180_dprfovea_{WHICH}_{TAG}.jsonl"
VIZDIR = f"{D}/figs/fovea_examples"
B0, W = 300, 0.25
# TRIMMED to the gap left by phase 178, which already runs global/local 200/100, 150/150, 100/200
# and oracle_glocal_150_150. Here: the two global-heavy splits the 64-token floor clamps, plus the
# oracle sweep 178 lacks. uniform@600 and dpr_crop are kept as MERGE ANCHORS -- if they disagree
# with 178 beyond the noise floor, the two implementations are not comparable and the merge is void.
SPLITS = {"83_17": (236, 64), "75_25": (225, 75)}
ORACLE_SPLITS = {"83_17": (236, 64), "75_25": (225, 75), "67_33": (200, 100)}
CONN = "Here is a high-resolution view of the region of that image most relevant to the question."
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
        """Deterministic bicubic resize calibrated to a MEASURED merged-token count."""
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

    def win_box(img, cx, cy, w):
        iw, ih = img.size
        x0 = min(max(0, (cx - w/2)*iw), iw - w*iw); y0 = min(max(0, (cy - w/2)*ih), ih - w*ih)
        return (int(x0), int(y0), int(x0 + w*iw), int(y0 + w*ih))

    def answer(imgs, text):
        i = build(imgs, text)
        rz = int(sum(g[1]*g[2]//4 for g in i["image_grid_thw"].tolist()))
        i = i.to(model.device)
        t0 = time.time()
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        torch.cuda.synchronize()
        dt = time.time() - t0
        assert torch.isfinite(lg).all(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz, round(dt, 4)

    byq = {f"{e['category']}/{e['question_id']}": e for e in ds}
    if VIZ: os.makedirs(VIZDIR, exist_ok=True)
    items = [q for q in props if q in byq]
    if LIMIT: items = items[:LIMIT//2] + items[-(LIMIT - LIMIT//2):]
    t0, done, nviz = time.time(), 0, 0
    with open(OUT, "w") as fo:
        for qid in items:
            ex = byq[qid]
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            p = props[qid]
            cx, cy = p["head"]
            gt = p["gt_box_frac"]
            gcx, gcy = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
            roi_box = win_box(img, cx, cy, W)
            orc_box = win_box(img, gcx, gcy, W)
            roi, orc = img.crop(roi_box), img.crop(orc_box)

            probs, tokens, lat = {}, {}, {}
            def rec(nm, imgs):
                probs[nm], tokens[nm], lat[nm] = answer(imgs, ex["text"])
            rec("uniform@600", [fit(img, 2*B0)])          # merge anchor (the bar)
            rec("dpr_crop",    [fit(roi, B0)])             # merge anchor (= 178's head@0.25)
            for nm, (bg, bl) in SPLITS.items():
                rec(f"fovea_{nm}", [fit(img, bg), fit(roi, bl)])
            for nm, (bg, bl) in ORACLE_SPLITS.items():
                rec(f"oracle_fovea_{nm}", [fit(img, bg), fit(orc, bl)])

            lab = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            fo.write(json.dumps({
                "question_id_full": qid, "category": ex["category"], "label": lab,
                "gt_tokens": p.get("gt_tokens"), "head_cov": p.get("head_cov"),
                "gt_box_frac": gt, "roi_centre": [cx, cy], "img_wh": list(img.size),
                "probs": probs, "realized_tokens": tokens, "latency_s": lat}) + "\n")

            if VIZ and nviz < 4:
                sc = fit(img, SPLITS["67_33"][0]); hi = fit(roi, SPLITS["67_33"][1])
                ann = img.copy(); dr = ImageDraw.Draw(ann)
                dr.rectangle(roi_box, outline=(255, 40, 40), width=max(3, img.size[0]//220))
                dr.rectangle(orc_box, outline=(40, 200, 60), width=max(3, img.size[0]//300))
                panels = [ann, sc, hi]
                hgt = 360
                panels = [q.resize((max(1, int(q.size[0]*hgt/q.size[1])), hgt)) for q in panels]
                cw = sum(q.size[0] for q in panels) + 20
                sheet = Image.new("RGB", (cw, hgt), (255, 255, 255))
                x = 0
                for q in panels:
                    sheet.paste(q, (x, 0)); x += q.size[0] + 10
                sheet.save(f"{VIZDIR}/{WHICH}_{qid.replace('/', '_')}.png")
                nviz += 1
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(items)}] {done/(time.time()-t0):.2f} it/s", flush=True)
    print(f"wrote {OUT}  n={done}" + (f"  viz -> {VIZDIR}" if VIZ else ""))


if __name__ == "__main__":
    main()
