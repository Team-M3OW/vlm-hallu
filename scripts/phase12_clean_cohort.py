"""
Phase 12: rebuild the confident-denial cohort RePOPE-clean and in fp16.

WHY THIS IS A PREREQUISITE, NOT A CONVENIENCE
---------------------------------------------
Every cohort in this project so far has two defects, and Phase 11 quantified the first:

  1. LABEL NOISE. The `confident_denial` cohort is 30.9% mislabeled + 27.2% ambiguous = 58.1% not
     clean (crop cohort: 61.7%). Cleaning it roughly DOUBLED the context-interference effect
     (40.4% -> 72.2% crop-alone recovery), because mislabeled items -- where the object genuinely
     is not there -- can never be "recovered" by any arm and were being scored as model failures.
  2. PRECISION MISMATCH (FINDINGS bug #6). The cached `p_yes_real` in the phase1 files came from a
     4-bit model. All interpretability work is fp16. One item measured 0.0067 (4-bit) vs 0.169
     (fp16) -- crossing the <0.01 cohort threshold. Cohort membership must be established in the
     same precision it will be used in.

So this recomputes P(yes) in fp16 over ALL RePOPE-clean POPE items (positives AND negatives) and
writes a clean baseline other phases select from. Positives with fp16 P(yes) < 0.01 become the
rebuilt cohort; it also fixes the n=54 thinness that currently caps every clean-data CI.

Clean = RePOPE kept the item (not pruned as ambiguous) AND RePOPE's label agrees with POPE's.
Method invariants held identical to §7 of FINDINGS (prompt, yes/no token pooling, fp16).
"""
import json
import sys
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase11_repope_audit import load_repope, verdict

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase12_clean_pyes.jsonl"
ANSWER_SUFFIX = " Please answer this question with yes or no."


def main():
    rp = load_repope()

    print("Loading items...")
    items = build_items()
    kept, drop = [], {"AMBIGUOUS": 0, "FLIPPED": 0, "MISSING": 0}
    for it in items:
        v = verdict(rp, (it["split"], str(it["question_id"])))
        if v == "CLEAN":
            kept.append(it)
        else:
            drop[v] += 1
    npos = sum(1 for it in kept if it["label"] == "yes")
    print(f"  RePOPE-CLEAN items: {len(kept)}  ({npos} positives, {len(kept)-npos} negatives)")
    print(f"  dropped: {drop}")

    done = set()
    try:
        for line in open(OUT_PATH):
            done.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done)} done")
    except FileNotFoundError:
        pass
    todo = [it for it in kept if it["uid"] not in done]
    if not todo:
        print("Nothing to do.")
        return

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

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for it in todo:
            img = it["image"].convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": it["question"] + ANSWER_SUFFIX}]}]
            chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=img, text=chat, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            yl = torch.logsumexp(logits[yes_ids], dim=0)
            nl = torch.logsumexp(logits[no_ids], dim=0)
            p = torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()
            fout.write(json.dumps({
                "uid": it["uid"], "label": it["label"], "split": it["split"],
                "question_id": it["question_id"], "category": it["category"],
                "pixel_area_frac": it["pixel_area_frac"],
                "patch_token_frac": it["patch_token_frac"],
                "question": it["question"], "p_yes_fp16": p,
                "image_wh": list(img.size),
                "image_grid_thw": inputs["image_grid_thw"][0].tolist(),
            }) + "\n")
            fout.flush()
            n += 1
            if n % 100 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] {n/el:.2f} it/s eta={(len(todo)-n)/(n/el)/60:.1f}min",
                      flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
