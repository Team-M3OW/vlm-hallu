"""
Phase 20 (E1-expanded): budget-matched query-conditional allocation on V*Bench.

WHY THIS BENCHMARK, AND WHY IT FIXES THE PAPER'S WEAKEST POINT
--------------------------------------------------------------
Phase 17 established the allocation effect on POPE, but the phenomenon there is only **1.6% of
positives** -- thin enough that a reviewer can call POPE saturated (P(yes) AUROC 0.9725), and
CUB (Phase 18) failed to rescue it (2.2%, and a different failure mode entirely).

V*Bench is the regime where sub-token targets are the DESIGN, not a tail:
  * 191 items, source images ~2246x1582 (SA-1B), deliberately crowded / small-detail
  * ships per-item target BBOXES (`{split}/{img}.json`: target_object, bbox [x,y,w,h], question)
  * example target: 29x81 px in 2246x1582 => area fraction 0.00066 -- SMALLER than our POPE
    confident denials (~0.0015)
⇒ every item is the condition we care about, so the effect base goes from 1.6% to ~100%.

READOUT MAPPING -- DECIDED BEFORE WRITING ANY CODE (the CUB lesson)
-------------------------------------------------------------------
V*Bench is 4-way multiple choice, NOT yes/no presence. None of this project's apparatus (pooled
yes/no logits, `P(yes)<0.01` cohorts, recovery rates, false-positive arms) transfers. The mapping:

  score      softmax over the logits of tokens {A,B,C,D} at the final position (the prompt already
             ends with "Answer with the option's letter from the given choices directly.")
  correct    argmax over those four == the gold label
  metric     ACCURACY per arm at matched budget (not "recovery")
  cohort     ALL 191 items -- no confidence-based selection, because the benchmark is already the
             hard regime. Accuracy on the natively-WRONG subset is reported separately as the
             closest analogue to Phase 17's recovery number.
  FP arm     NOT NEEDED, and this is a genuine advantage of the MCQ metric: 4-way argmax is
             intrinsically immune to the bias-shift trap that Phase 16's `rand_last` fell into
             (uniformly inflating one option's logit cannot improve argmax accuracy). The decider
             remains `alloc_random`.

ARMS (matched on REALIZED tokens via the Phase 17 calibration loop)
-------------------------------------------------------------------
    uniform@B       whole image downsampled to B
    alloc_query@B   low-res full (B/2) + GT target crop (B/2)
    alloc_random@B  IDENTICAL layout, size-matched WRONG region        <-- THE DECIDER
    crop_only@B     target crop alone at B

B in {300, 600, 1200}. Oracle region (from the benchmark's own annotation), stated as such --
same framing as Phase 7/17. NOTE: V*Bench was built by the V*/SEAL authors to motivate their crop
method, so "cropping helps here" is already their result; the matched-budget framing is what makes
this a different claim rather than a replication of theirs.
"""
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase17_budget_allocation import fit_to_budget
from phase7_vision_zoom import CONNECTOR_TEXT

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase20_vstar_results.jsonl"
BUDGETS = [300, 600, 1200]
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    rng = random.Random(107)
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    print("Fetching V*Bench...")
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    print(f"  {len(ds)} items, root={root}")

    print("Loading Qwen3-VL-2B (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]
    print("option token ids:", opt_ids)

    def _chat(imgs, text, conn):
        if len(imgs) == 1:
            content = [{"type": "image"}]
        else:
            content = ([{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                        {"type": "image"}] if conn else [{"type": "image"}, {"type": "image"}])
        content.append({"type": "text", "text": text})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    def measure(imgs, text="x", conn=True):
        i = processor(images=imgs, text=_chat(imgs, text, conn), return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    def run(imgs, text, conn):
        i = processor(images=imgs, text=_chat(imgs, text, conn), return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        pooled = torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids])
        probs = torch.softmax(pooled, 0)
        realized = int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
        return probs.tolist(), int(probs.argmax().item()), realized

    done = set()
    try:
        for l in open(OUT_PATH):
            done.add(json.loads(l)["question_id_full"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for ex in ds:
            rel = ex["image"]
            qid_full = f"{rel}::{ex['question_id']}"
            if qid_full in done:
                continue
            img_path = os.path.join(root, rel)
            ann_path = os.path.splitext(img_path)[0] + ".json"
            if not (os.path.exists(img_path) and os.path.exists(ann_path)):
                continue
            ann = json.load(open(ann_path))
            if not ann.get("bbox"):
                continue
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            # union over all annotated target boxes (COCO-style x,y,w,h)
            x0 = min(b[0] for b in ann["bbox"]); y0 = min(b[1] for b in ann["bbox"])
            x1 = max(b[0] + b[2] for b in ann["bbox"]); y1 = max(b[1] + b[3] for b in ann["bbox"])
            pw, ph = (x1 - x0) * PAD, (y1 - y0) * PAD
            cx0, cy0 = max(0, x0 - pw), max(0, y0 - ph)
            cx1, cy1 = min(W, x1 + pw), min(H, y1 + ph)
            if cx1 - cx0 < 8 or cy1 - cy0 < 8:
                continue
            gt_crop = img.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
            gw, gh = cx1 - cx0, cy1 - cy0
            rx = rng.uniform(0, max(1, W - gw)); ry = rng.uniform(0, max(1, H - gh))
            rnd_crop = img.crop((int(rx), int(ry), int(rx + gw), int(ry + gh)))

            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            rec = {"question_id_full": qid_full, "image": rel, "category": ex["category"],
                   "label": label, "target_object": ann.get("target_object"),
                   "img_wh": [W, H], "bbox_area_frac": ((x1-x0)*(y1-y0))/(W*H),
                   "arms": {}, "pred": {}, "realized_tokens": {}}
            text = ex["text"]
            for B in BUDGETS:
                cells = {"uniform": ([img], False), "crop_only": ([gt_crop], False),
                         "alloc_query": ([img, gt_crop], True),
                         "alloc_random": ([img, rnd_crop], True)}
                for name, (imgs, conn) in cells.items():
                    fitted, _ = fit_to_budget(lambda ims: measure(ims, text, conn), imgs, B)
                    probs, pred, realized = run(fitted, text, conn)
                    rec["arms"][f"{name}@{B}"] = probs
                    rec["pred"][f"{name}@{B}"] = pred
                    rec["realized_tokens"][f"{name}@{B}"] = realized
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[{n}] {n/el:.2f} it/s", flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
