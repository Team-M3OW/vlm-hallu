"""
Phase 5 validation (per advisor, 2026-09-03): two questions that decide whether the layer-23
collapse finding supports a real, usable fix, or is a narrower interpretability curiosity.

1. Does layer 15 (before the collapse band) meaningfully track VISUAL EVIDENCE, or is it a
   content-independent default that says "yes" to almost everything? Tested by comparing layer-15
   P(yes) on true NEGATIVES (object genuinely absent) against the confident_denial group's layer-15
   values (already known to be ~0.99 from phase5_logit_lens.py). If negatives ALSO sit near 0.99 at
   layer 15, it's a generic default, not evidence tracking.

2. If layer 15 does track evidence: does reading the answer off an intermediate layer actually beat
   the final layer on POSITIVES-VS-NEGATIVES AUROC (not just recall on cherry-picked denials, which
   would repeat the exact bias-shift trap already caught once in this project with the padding
   experiment)? Computed on a broad, unbiased sample (all smallest-area-bin positives + a matched
   negative sample), not the two extreme cohorts used to find the effect -- avoiding circularity.
"""
import json
import sys
import time
import random
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase5_layer_auroc_results.jsonl"
N_POS_SAMPLE = 500
N_NEG_SAMPLE = 500
CANDIDATE_LAYERS = [15, 18, 20, 22, 25, 28]


def auroc(pos_scores, neg_scores):
    labeled = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    labeled.sort(key=lambda x: x[0])
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    n = len(labeled)
    while i < n:
        j = i
        while j < n and labeled[j][0] == labeled[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if labeled[k][1] == 1:
                rank_sum += avg_rank
        i = j
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def main():
    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs = [json.loads(l) for l in f]
    small_positives = [r for r in recs if r["label"] == "yes" and r.get("pixel_area_frac") is not None
                        and r["pixel_area_frac"] < 0.02]
    negatives = [r for r in recs if r["label"] == "no"]

    rng = random.Random(7)
    pos_sample = rng.sample(small_positives, min(N_POS_SAMPLE, len(small_positives)))
    neg_sample = rng.sample(negatives, min(N_NEG_SAMPLE, len(negatives)))
    print(f"positive sample (smallest-area-bin, unbiased): n={len(pos_sample)}")
    print(f"negative sample (unbiased): n={len(neg_sample)}")

    targets = [(r, "positive") for r in pos_sample] + [(r, "negative") for r in neg_sample]
    target_uids = {r["uid"] for r, _ in targets}

    print("Loading items...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)}")

    print("Loading Qwen3-VL-2B (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    tok = processor.tokenizer
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in ["yes", "Yes", " yes", " Yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                      for s in ["no", "No", " no", " No"]})

    lm_head = model.lm_head
    final_norm = model.model.language_model.norm

    def p_yes_from_hidden(h):
        with torch.no_grad():
            normed = final_norm(h.unsqueeze(0))
            logits = lm_head(normed)[0]
        yl = torch.logsumexp(logits[yes_ids], dim=0)
        nl = torch.logsumexp(logits[no_ids], dim=0)
        return torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()

    def build_prompt(question):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question} Please answer this question with yes or no."},
        ]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def trajectory_at_layers(image, question, layers):
        prompt = build_prompt(question)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return {l: p_yes_from_hidden(out.hidden_states[l][0, -1]) for l in layers}

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r, group in targets:
            item = items_by_uid.get(r["uid"])
            if item is None:
                continue
            img = item["image"].convert("RGB")
            p_at_layers = trajectory_at_layers(img, item["question"], CANDIDATE_LAYERS)
            rec = {"uid": r["uid"], "group": group, "p_at_layers": p_at_layers}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"[{n_done}/{len(targets)}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
