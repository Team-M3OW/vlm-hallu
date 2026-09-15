"""
Phase 19 (E2b): does budget-matched allocation help FINE-GRAINED discrimination?

THE QUESTION
------------
Phase 18 found CUB's failure is NOT ours: birds are large (53.8 tokens on object) and the model
OVER-ACCEPTS confusable species -- 70.7% false-positive rate on same-genus negatives, AUROC 0.672.
Token starvation cannot explain that; the capacity is there.

But the bird being large does not mean the DIAGNOSTIC evidence is large. Species separation within a
genus lives in the beak, crown, eye-ring and throat -- structures that may themselves be sub-token
even when the bird fills the frame. So:

    Does allocating the SAME token budget to the DIAGNOSTIC PARTS lift same-genus discrimination
    above the 0.672 baseline?

  * YES -> the allocation claim generalizes from OBJECT SIZE to DIAGNOSTIC DETAIL. Materially
    stronger, and it makes Phase 18's counterexample into supporting evidence.
  * NO  -> the claim is specifically about object size. That is the honest scope, and Phase 18
    stands as a boundary on the thesis.

METRIC (different from Phase 17 -- note carefully)
--------------------------------------------------
Phase 17 measured RECOVERY (flipping a confident denial to "yes"). Here the failure is the opposite
polarity, so the metric is DISCRIMINATION: AUROC of P(yes) between the TRUE species and a
SAME-GENUS wrong species, asked of the identical image under the identical arm. Higher = better.
Raising P(yes) on everything does nothing to AUROC, so this metric is intrinsically immune to the
bias-shift trap that Phase 16's `rand_last` fell into.

ARMS (all matched on REALIZED tokens, via the Phase 17 calibration loop)
------------------------------------------------------------------------
    uniform@B        whole image downsampled to B
    alloc_bbox@B     low-res full (B/2) + bird bbox crop (B/2)
    alloc_head@B     low-res full (B/2) + DIAGNOSTIC-PART region (B/2)   <-- the parts arm
    alloc_random@B   low-res full (B/2) + random region SIZE-MATCHED TO THE HEAD BOX  <-- DECIDER
    crop_bbox@B      bird bbox crop alone at B

Diagnostic region = padded bounding box of the VISIBLE head parts
{beak, crown, forehead, left eye, right eye, nape, throat} from CUB `parts/part_locs.txt`
(points, not boxes -- so they are unioned and padded).

B in {150, 300}. CUB test split only.
"""
import json
import random
import sys
import time
from collections import defaultdict

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase17_budget_allocation import fit_to_budget
from phase18_cub_tail import pretty, genus
from phase7_vision_zoom import CONNECTOR_TEXT

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ROOT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/cub/CUB_200_2011"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase19_cub_parts_results.jsonl"
ANSWER_SUFFIX = " Please answer this question with yes or no."
BUDGETS = [150, 300]
HEAD_PARTS = {2, 5, 6, 7, 10, 11, 15}   # beak, crown, forehead, l-eye, nape, r-eye, throat
N_ITEMS = 420
PAD = 0.6


def main():
    rng = random.Random(101)
    classes = {int(a): b for a, b in (l.split() for l in open(f"{ROOT}/classes.txt"))}
    images = {int(a): b for a, b in (l.split() for l in open(f"{ROOT}/images.txt"))}
    labels = {int(a): int(b) for a, b in (l.split() for l in open(f"{ROOT}/image_class_labels.txt"))}
    boxes = {int(p[0]): [float(v) for v in p[1:]] for p in (l.split() for l in open(f"{ROOT}/bounding_boxes.txt"))}
    is_test = {int(a): b == "0" for a, b in (l.split() for l in open(f"{ROOT}/train_test_split.txt"))}
    parts = defaultdict(list)
    for l in open(f"{ROOT}/parts/part_locs.txt"):
        iid, pid, x, y, vis = l.split()
        if int(vis) == 1 and int(pid) in HEAD_PARTS:
            parts[int(iid)].append((float(x), float(y)))

    by_genus = defaultdict(list)
    for cid, nm in classes.items():
        by_genus[genus(nm)].append(cid)

    ids = [i for i in images if is_test.get(i) and len(parts.get(i, [])) >= 2]
    rng.shuffle(ids)
    ids = ids[:N_ITEMS]
    print(f"CUB test images with >=2 visible head parts: using {len(ids)}")

    done = set()
    try:
        for l in open(OUT_PATH):
            done.add(json.loads(l)["image_id"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass
    ids = [i for i in ids if i not in done]

    print("Loading Qwen3-VL-2B (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                      for s in ["yes", "Yes", " yes", " Yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                     for s in ["no", "No", " no", " No"]})

    def _chat(imgs, q, conn):
        if len(imgs) == 1:
            content = [{"type": "image"}]
        else:
            content = ([{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                        {"type": "image"}] if conn else [{"type": "image"}, {"type": "image"}])
        content.append({"type": "text", "text": q + ANSWER_SUFFIX})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    def measure(imgs, q="x", conn=True):
        i = processor(images=imgs, text=_chat(imgs, q, conn), return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    def run(imgs, q, conn):
        i = processor(images=imgs, text=_chat(imgs, q, conn), return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                       torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()
        return p, int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for iid in ids:
            cid = labels[iid]
            g = genus(classes[cid])
            same = [c for c in by_genus[g] if c != cid]
            if not same:
                continue
            hard = classes[rng.choice(same)]
            img = Image.open(f"{ROOT}/images/{images[iid]}").convert("RGB")
            W, H = img.size
            bx, by, bw, bh = boxes[iid]
            bbox_crop = img.crop((int(bx), int(by), int(bx + bw), int(by + bh)))

            pts = parts[iid]
            px0, py0 = min(p[0] for p in pts), min(p[1] for p in pts)
            px1, py1 = max(p[0] for p in pts), max(p[1] for p in pts)
            pw, ph = max(px1 - px0, bw * .12), max(py1 - py0, bh * .12)
            hx0 = max(0, px0 - PAD * pw); hy0 = max(0, py0 - PAD * ph)
            hx1 = min(W, px1 + PAD * pw); hy1 = min(H, py1 + PAD * ph)
            if hx1 - hx0 < 8 or hy1 - hy0 < 8:
                continue
            head_crop = img.crop((int(hx0), int(hy0), int(hx1), int(hy1)))
            hw, hh = hx1 - hx0, hy1 - hy0
            rx = rng.uniform(0, max(1, W - hw)); ry = rng.uniform(0, max(1, H - hh))
            rnd_crop = img.crop((int(rx), int(ry), int(rx + hw), int(ry + hh)))

            rec = {"image_id": iid, "class": classes[cid], "genus": g,
                   "species": pretty(classes[cid]), "hard_species": pretty(hard),
                   "bbox_area_frac": (bw * bh) / (W * H),
                   "head_area_frac": (hw * hh) / (W * H),
                   "n_head_parts": len(pts), "arms": {}, "realized_tokens": {}}
            for B in BUDGETS:
                cells = {
                    "uniform":      ([img], False),
                    "crop_bbox":    ([bbox_crop], False),
                    "alloc_bbox":   ([img, bbox_crop], True),
                    "alloc_head":   ([img, head_crop], True),
                    "alloc_random": ([img, rnd_crop], True),
                }
                for name, (imgs, conn) in cells.items():
                    fitted, _ = fit_to_budget(lambda ims: measure(ims, "x", conn), imgs, B)
                    for qlab, sp in [("true", pretty(classes[cid])), ("hard", pretty(hard))]:
                        p, realized = run(fitted, f"Is there a {sp} in the image?", conn)
                        rec["arms"][f"{name}@{B}:{qlab}"] = p
                        rec["realized_tokens"][f"{name}@{B}"] = realized
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(ids)}] {n/el:.2f} it/s eta={(len(ids)-n)/(n/el)/60:.1f}min",
                      flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
