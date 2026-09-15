"""
The killer experiment (per advisor, 2026-09-03): if center-crop preprocessing is why LLaVA's
small+peripheral cell collapses to 0.54 accuracy, then re-running those SAME items with
pad-instead-of-crop preprocessing (so no content is discarded, aspect ratio preserved via letterbox
padding to a square before the standard resize) should recover accuracy toward the small+central
level (0.80), without touching the model weights at all -- a pure preprocessing fix.

Reuses the pre-quantized LLaVA checkpoint and the same forced single-token P(yes) forward pass as
phase1_eval.py. Only the image preprocessing changes: pad each image to a square (letterbox, black
fill) before handing it to the processor, so CLIP's resize-shortest-edge-then-center-crop becomes a
no-op crop (already square) instead of discarding the longer axis's edges.

"Before" accuracy is read directly from the already-computed data/phase1_results.jsonl (identical
items, identical model, default/cropping preprocessing) rather than re-run, to isolate the
preprocessing change as the only variable and save compute.
"""
import json
import re
import sys
import time
import argparse
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase3_crop_check import build_survival_by_key
from phase3_centrality import build_centrality_by_key, tertile_edges, tertile_of
from phase1_eval import build_items

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
LOCAL_4BIT_DIR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/models/llava-1.5-7b-4bit"
OUT_PATH = f"{DATA}/phase3_fix_pad_results.jsonl"


def pad_to_square(img):
    w, h = img.size
    side = max(w, h)
    return ImageOps.pad(img, (side, side), color=(0, 0, 0), centering=(0.5, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("Identifying the small+peripheral cell (same tertile definition as phase3_crop_check.py) ...")
    with open(f"{DATA}/phase1_results.jsonl") as f:
        recs = [json.loads(l) for l in f]
    positives = [r for r in recs if r["label"] == "yes" and r.get("patch_token_frac") is not None]

    centrality_by_key = build_centrality_by_key()
    survival_by_key = build_survival_by_key()

    triples = []
    for r in positives:
        key = (r["split"], str(r["question_id"]))
        c = centrality_by_key.get(key)
        s = survival_by_key.get(key)
        if c is not None and s is not None:
            triples.append((r, c, s))

    areas = [t[0]["patch_token_frac"] for t in triples]
    cents = [t[1] for t in triples]
    a_edges = tertile_edges(areas)
    c_edges = tertile_edges(cents)

    small_peripheral = [t for t in triples
                         if tertile_of(t[0]["patch_token_frac"], a_edges) == 0
                         and tertile_of(t[1], c_edges) == 2]
    print(f"small+peripheral cell: n={len(small_peripheral)}")
    before_acc = sum(1 for t in small_peripheral if t[0]["p_yes_real"] > 0.5) / len(small_peripheral)
    print(f"before (cropped, cached) accuracy on this cell = {before_acc:.4f}")

    fully_cropped = [t for t in small_peripheral if t[2] == 0.0]
    partially_cropped = [t for t in small_peripheral if 0.0 < t[2] < 0.95]
    intact = [t for t in small_peripheral if t[2] >= 0.95]
    print(f"  breakdown: fully_cropped(survival==0) n={len(fully_cropped)}, "
          f"partially_cropped n={len(partially_cropped)}, intact(survival>=0.95) n={len(intact)}")

    items = small_peripheral
    if args.limit:
        items = items[:args.limit]

    done_uids = set()
    if args.resume:
        try:
            with open(OUT_PATH) as f:
                for line in f:
                    done_uids.add(json.loads(line)["uid"])
            print(f"Resuming: {len(done_uids)} already done")
        except FileNotFoundError:
            pass

    target_uids = {f"pos_{r['split']}_{r['question_id']}" for r, _, _ in items if f"pos_{r['split']}_{r['question_id']}" not in done_uids}
    print(f"Loading full item list via build_items() (validated join logic, reused from phase1_eval.py) "
          f"to fetch images for {len(target_uids)} target items ...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)} target uids to images")

    print("Loading pre-quantized LLaVA checkpoint ...")
    processor = AutoProcessor.from_pretrained(LOCAL_4BIT_DIR)
    import gc
    model = None
    for attempt in range(5):
        try:
            model = LlavaForConditionalGeneration.from_pretrained(LOCAL_4BIT_DIR, device_map={"": 0})
            break
        except torch.OutOfMemoryError:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"OOM on load attempt {attempt+1}/5, retrying in 15s")
            time.sleep(15)
    if model is None:
        raise RuntimeError("Could not load model.")
    model.eval()

    tok = processor.tokenizer
    yes_variants = ["yes", "Yes", " yes", " Yes"]
    no_variants = ["no", "No", " no", " No"]
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in yes_variants})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in no_variants})

    def p_yes(image, question):
        prompt = f"USER: <image>\n{question} Please answer this question with yes or no. ASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
        no_logit = torch.logsumexp(logits[no_ids], dim=0)
        probs = torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)
        return probs[0].item()

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r, cent, surv in items:
            uid = f"pos_{r['split']}_{r['question_id']}"
            if uid in done_uids:
                continue
            item = items_by_uid.get(uid)
            if item is None:
                print(f"WARNING: no image found for {uid}, skipping")
                continue
            img = item["image"].convert("RGB")
            padded = pad_to_square(img)
            try:
                py_padded = p_yes(padded, item["question"])
            except Exception as e:
                print(f"ERROR on {uid}: {e}")
                continue
            rec = {"uid": uid, "split": r["split"], "question_id": r["question_id"],
                   "category": r["category"], "patch_token_frac": r["patch_token_frac"],
                   "centrality": cent, "crop_survival": surv,
                   "p_yes_real_cropped": r["p_yes_real"], "p_yes_real_padded": py_padded}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                print(f"[{n_done}/{len(items)-len(done_uids)}] rate={n_done/elapsed:.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
