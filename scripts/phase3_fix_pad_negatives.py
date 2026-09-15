"""
Confirmatory check (per advisor, 2026-09-03): does pad-instead-of-crop preprocessing degrade
negative-class accuracy? If padding were a pure yes-ward bias shift, negatives should get MORE
false "yes" answers under padding. Run a sample of negatives through the same padded preprocessing
and compare to their cached cropped P(yes) from phase1_results.jsonl.
"""
import json
import time
import random
import torch
from PIL import ImageOps
from transformers import AutoProcessor, LlavaForConditionalGeneration

import sys
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
LOCAL_4BIT_DIR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/models/llava-1.5-7b-4bit"
OUT_PATH = f"{DATA}/phase3_fix_pad_negatives_results.jsonl"
N_SAMPLE = 300


def pad_to_square(img):
    w, h = img.size
    side = max(w, h)
    return ImageOps.pad(img, (side, side), color=(0, 0, 0), centering=(0.5, 0.5))


def main():
    with open(f"{DATA}/phase1_results.jsonl") as f:
        recs = [json.loads(l) for l in f]
    negatives = [r for r in recs if r["label"] == "no"]
    # Use a local RNG, not random.seed()/random.sample() on the global module -- build_items()
    # (called below) also uses the global `random` module internally with its own seed(0) set at
    # import time, and polluting that global state here caused build_items() to reshuffle its
    # negative-split sampling differently, silently producing a DIFFERENT 1500-negative population
    # than the one actually used in phase1_results.jsonl (caught: only 88/300 resolved on the first
    # run, instead of 300/300).
    sample = random.Random(1).sample(negatives, N_SAMPLE)
    target_uids = {f"neg_{r['split']}_{r['question_id']}" for r in sample}

    print(f"Loading full item list to fetch images for {len(target_uids)} sampled negatives ...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)}")

    done_uids = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done_uids)} already done")
    except FileNotFoundError:
        pass

    print("Loading pre-quantized LLaVA checkpoint ...")
    processor = AutoProcessor.from_pretrained(LOCAL_4BIT_DIR)
    model = LlavaForConditionalGeneration.from_pretrained(LOCAL_4BIT_DIR, device_map={"": 0})
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
        for r in sample:
            uid = f"neg_{r['split']}_{r['question_id']}"
            if uid in done_uids:
                continue
            item = items_by_uid.get(uid)
            if item is None:
                continue
            img = item["image"].convert("RGB")
            padded = pad_to_square(img)
            py_padded = p_yes(padded, item["question"])
            rec = {"uid": uid, "split": r["split"], "question_id": r["question_id"],
                   "p_yes_real_cropped": r["p_yes_real"], "p_yes_real_padded": py_padded}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
