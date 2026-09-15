"""
Phase 1: forced single-token P(yes)/P(no) eval of LLaVA-1.5-7B (4-bit) on POPE(coco) positives
(area-joined) + a matched negative pool, both on the real image and a blank/black image
(prior-solvability gate). One forward pass per (item, image-variant) -- no sampling, no
autoregression.

Writes incremental results to data/phase1_results.jsonl so a crash/interrupt loses at most the
in-flight batch.
"""
import json
import random
import time
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LOCAL_4BIT_DIR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/models/llava-1.5-7b-4bit"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase1_results.jsonl"

random.seed(0)


def build_items(limit=None, neg_per_split=500):
    with open(f"{DATA}/pope_coco_area_joined.json") as f:
        positives = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("lmms-lab/pope", split="test")
    idx = {}
    for row in ds:
        key = (row["category"], row["question_id"])
        idx.setdefault(key, []).append(row)

    items = []
    for p in positives:
        key = (p["split"], str(p["question_id"]))
        rows = idx.get(key)
        if not rows:
            continue
        row = rows[0]
        items.append({
            "uid": f"pos_{p['split']}_{p['question_id']}",
            "label": "yes",
            "split": p["split"],
            "question_id": p["question_id"],
            "category": p["category"],
            "pixel_area_frac": p["pixel_area_frac"],
            "patch_token_frac": p["patch_token_frac"],
            "question": row["question"],
            "image": row["image"],
        })

    negs_by_split = {"random": [], "popular": [], "adversarial": []}
    for row in ds:
        if row["answer"] == "no":
            negs_by_split[row["category"]].append(row)
    neg_items = []
    for split, rows in negs_by_split.items():
        random.shuffle(rows)
        for row in rows[:neg_per_split]:
            neg_items.append({
                "uid": f"neg_{split}_{row['question_id']}",
                "label": "no",
                "split": split,
                "question_id": row["question_id"],
                "category": None,
                "pixel_area_frac": None,
                "patch_token_frac": None,
                "question": row["question"],
                "image": row["image"],
            })

    all_items = items + neg_items
    random.shuffle(all_items)
    if limit:
        all_items = all_items[:limit]
    return all_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    items = build_items(limit=args.limit)
    print(f"Total items to evaluate: {len(items)} "
          f"(pos={sum(1 for i in items if i['label']=='yes')}, "
          f"neg={sum(1 for i in items if i['label']=='no')})")

    done_uids = set()
    if args.resume:
        try:
            with open(OUT_PATH) as f:
                for line in f:
                    done_uids.add(json.loads(line)["uid"])
            print(f"Resuming: {len(done_uids)} already done")
        except FileNotFoundError:
            pass

    print("Loading pre-quantized 4-bit checkpoint from", LOCAL_4BIT_DIR)
    # Pre-quantized on CPU offline (scripts/prequantize_llava.py) specifically to avoid bnb's
    # load-time fp16-shard-then-quantize transient spike, which was OOM'ing against the ~39GB
    # used by another job sharing GPU0 even though the ~4.5GB *resident* 4-bit footprint fits
    # comfortably in the ~8GB steady-state headroom. Loading pre-quantized shards keeps peak
    # ~= resident, no such spike.
    processor = AutoProcessor.from_pretrained(LOCAL_4BIT_DIR)
    import gc
    model = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                LOCAL_4BIT_DIR, device_map={"": 0},
            )
            break
        except torch.OutOfMemoryError:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info(0)[0] / 1e6
            print(f"OOM on load attempt {attempt+1}/{max_attempts} (GPU0 free={free_mb:.0f}MB), "
                  f"retrying in 15s")
            time.sleep(15)
    if model is None:
        raise RuntimeError(f"Could not load model after {max_attempts} attempts, GPU too busy.")
    model.eval()

    tok = processor.tokenizer
    # Pool case variants via logsumexp rather than betting on one spelling -- confirmed necessary
    # empirically on Qwen3-VL (lowercase 'yes'/'no' outranked 'Yes'/'No' there); verify per-model
    # via the top-5 debug dump below rather than assuming llama's tokenizer behaves differently.
    yes_variants = ["yes", "Yes", " yes", " Yes"]
    no_variants = ["no", "No", " no", " No"]
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in yes_variants})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in no_variants})
    print(f"yes_ids={yes_ids} ({[tok.decode([i]) for i in yes_ids]}), "
          f"no_ids={no_ids} ({[tok.decode([i]) for i in no_ids]})")

    black_img_cache = {}

    def get_black_image(size):
        if size not in black_img_cache:
            black_img_cache[size] = Image.new("RGB", size, (0, 0, 0))
        return black_img_cache[size]

    def p_yes(image, question, debug_argmax=False):
        # standard POPE prompt format (question + explicit yes/no instruction), not a custom one
        prompt = f"USER: <image>\n{question} Please answer this question with yes or no. ASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        if debug_argmax:
            top5 = torch.topk(logits, 5)
            print("  top5 next-token:", [(tok.decode([i]), round(v.item(), 2))
                                          for v, i in zip(top5.values, top5.indices)])
        yes_logit = torch.logsumexp(logits[yes_ids], dim=0)
        no_logit = torch.logsumexp(logits[no_ids], dim=0)
        two = torch.stack([yes_logit, no_logit])
        probs = torch.softmax(two, dim=0)
        return probs[0].item()

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for i, item in enumerate(items):
            if item["uid"] in done_uids:
                continue
            img = item["image"].convert("RGB")
            t_item = time.time()
            try:
                py_real = p_yes(img, item["question"], debug_argmax=(n_done < 5))
                py_blank = p_yes(get_black_image(img.size), item["question"])
            except Exception as e:
                print(f"ERROR on {item['uid']}: {e}")
                continue
            if n_done < 5:
                print(f"  item={item['uid']} label={item['label']} p_yes_real={py_real:.3f} "
                      f"p_yes_blank={py_blank:.3f} took={time.time()-t_item:.2f}s")
            rec = {k: v for k, v in item.items() if k != "image"}
            rec["p_yes_real"] = py_real
            rec["p_yes_blank"] = py_blank
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = len(items) - len(done_uids) - n_done
                eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                print(f"[{n_done}/{len(items)-len(done_uids)}] rate={rate:.2f}/s ETA={eta_min:.1f}min "
                      f"GPU_mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

    print(f"Done. Wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
