"""
Phase 8: answer-by-grounding as an UNBIASED presence detector (the §4.1 headline experiment).

The dissociation (Phase 4) showed the localization channel stays correct on items where the yes/no
channel is confidently wrong. Phase 8 asks the question that turns that observation into a usable
result: on a BROAD, UNSELECTED positives-vs-negatives sample, does a grounding-derived presence
score beat the standard forced-choice P(yes) readout -- at matched compute?

Why the Phase 4 number (0.749 balanced accuracy) cannot be the headline: that cohort was *selected*
for p_yes_real < 0.01, i.e. selected for the yes/no channel failing. Reporting it as a detector
result would repeat exactly the bias-shift trap this project already caught twice (bugs #7, #10).

DESIGN
------
Sample: 500 of ALL 4053 positives + 500 of ALL 1500 negatives, random.Random(7). This deliberately
DEVIATES from phase5_layer_auroc_validation.py, which sampled only `pixel_area_frac < 0.02`
positives: a §4.1 headline needs a broad population, not the small-object stratum where the effect
was discovered. Same seed, different population -- do NOT describe this as "the phase5 sample".
pixel_area_frac is recorded per item so the small-object stratum is recoverable post-hoc (and gives
the dose-response link to the §2.1 scaling law for free).

THE MATCHED-COST SCORE (p_ground)
---------------------------------
Nothing in "Locate the {obj}..." forces a binary, so we do not assume one. Empirically, on all 486
Phase 4 grounding generations, the FIRST generated token has exactly two-valued support under the
Qwen tokenizer:
    id 73594 '```'   (341/486) -> begins a ```json [{"bbox_2d": ...}] block  = ASSERTS PRESENCE
    id  3862 'There' (145/486) -> begins "There are none." / "There is no X" = ABSTAINS
So p_ground = softmax over the two pooled logits at the first generation position -- ONE forward
pass, the same cost as P(yes). That is the compute-matched comparison. We additionally log the
top-10 tokens and the captured probability mass at that position so tail/coverage failures are
diagnosable post-hoc without a re-run.

CONTROLS (all recorded in the same loop, one GPU run)
----------------------------------------------------
1. BLANK-IMAGE NULL. POPE asks a *different question* for positives vs negatives, so a detector can
   score AUROC > 0.5 from category-frequency priors alone with no perception at all. We recompute
   p_ground on a black image of the same size; p_yes_blank is already cached from phase1. If the
   grounding advantage survives on blank images, it is a language prior, not a perception channel.
2. PER-SPLIT AUROC. POPE-adversarial negatives are co-occurring objects. A detector that only wins
   on `random` is exploiting a co-occurrence prior. `split` is recorded per item.
3. FULL GENERATION (separate, UNMATCHED compute tier -- never folded into the matched-cost claim).
   ~100 decode steps giving n_pred_boxes, best IoU vs COCO GT, and raw text, for the richer
   grounding scores of §4.1 bullet 2.

Precision: fp16 throughout. The cached `p_yes_real` in phase1 files came from a 4-bit model
(bug #6), so P(yes) is RECOMPUTED here in fp16 on the identical items -- the comparison must be
within-precision or it is meaningless.

RNG discipline (bug #4): never touch the global `random` module before build_items(), which shuffles
with it. Only random.Random(seed) instances are used here.
"""
import json
import re
import sys
import time
import argparse
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase4_localize_vs_answer import extract_boxes, iou

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase8_grounding_detector_results.jsonl"

N_POS_SAMPLE = 500
N_NEG_SAMPLE = 500
SEED = 7

GROUND_PROMPT = "Locate the {obj} in the image and output its bounding box coordinates."
ANSWER_SUFFIX = " Please answer this question with yes or no."

# First-token vocabulary of the grounding channel, established empirically over all 486 Phase 4
# generations (see module docstring). Kept to the observed support rather than a guessed superset;
# coverage is verified per item via `ground_mass_captured` and `top10`.
BOX_START_STRS = ["```"]
ABSTAIN_STRS = ["There"]


def build_bbox_lookup():
    """(split, question_id) -> list of COCO GT boxes [x1,y1,x2,y2] in raw pixels, all instances of
    the queried category (union-of-instances, per bug #8)."""
    with open(f"{DATA}/pope_coco_area_joined.json") as f:
        joined = json.load(f)
    with open(f"{DATA}/coco_ann/instances_val2014.json") as f:
        coco = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    anns_by_image_cat = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_cat[(a["image_id"], a["category_id"])].append(a)

    lookup = {}
    for j in joined:
        img_id = int(re.search(r"(\d+)\.jpg$", j["image"]).group(1))
        cat_id = cat_name_to_id.get(j["category"])
        if cat_id is None:
            continue
        lookup[(j["split"], str(j["question_id"]))] = [
            [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
            for a in anns_by_image_cat.get((img_id, cat_id), [])
        ]
    return lookup


def extract_category(question):
    m = re.search(r"Is there an?\s+(.+?)\s+in the image\?", question, re.I)
    return m.group(1).strip().lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-generate", action="store_true",
                    help="skip the unmatched-compute full generation tier")
    args = ap.parse_args()

    import random as _random
    rng = _random.Random(SEED)

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs = [json.loads(l) for l in f]

    positives = [r for r in recs if r["label"] == "yes"]
    negatives = [r for r in recs if r["label"] == "no"]
    # negatives carry category=None in the phase1 schema; recover the queried object from the
    # question text (same regex phase0_area_join.py / phase4 use).
    for r in negatives:
        r["category"] = extract_category(r["question"])
    negatives = [r for r in negatives if r["category"] is not None]

    pos_sample = rng.sample(positives, min(N_POS_SAMPLE, len(positives)))
    neg_sample = rng.sample(negatives, min(N_NEG_SAMPLE, len(negatives)))
    print(f"BROAD positive sample (all positives, not area-restricted): n={len(pos_sample)}")
    print(f"BROAD negative sample: n={len(neg_sample)}")

    targets = [(r, "positive") for r in pos_sample] + [(r, "negative") for r in neg_sample]
    if args.limit:
        targets = targets[:args.limit]

    done_uids = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done_uids)} already done")
    except FileNotFoundError:
        pass
    targets = [(r, g) for r, g in targets if r["uid"] not in done_uids]
    if not targets:
        print("Nothing to do.")
        return

    bbox_lookup = build_bbox_lookup()

    print(f"Loading item images for {len(targets)} targets...")
    target_uids = {r["uid"] for r, _ in targets}
    items_by_uid = {it["uid"]: it for it in build_items() if it["uid"] in target_uids}
    print(f"  resolved {len(items_by_uid)}/{len(target_uids)}")

    print("Loading Qwen3-VL-2B (fp16)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer

    def first_ids(strs):
        return sorted({tok(s, add_special_tokens=False)["input_ids"][0] for s in strs})

    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                      for s in ["yes", "Yes", " yes", " Yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                     for s in ["no", "No", " no", " No"]})
    box_ids = first_ids(BOX_START_STRS)
    abstain_ids = first_ids(ABSTAIN_STRS)
    print(f"yes_ids={yes_ids} no_ids={no_ids} box_ids={box_ids} abstain_ids={abstain_ids}")
    assert box_ids == [73594] and abstain_ids == [3862], \
        f"token ids drifted from the empirically verified support: {box_ids}, {abstain_ids}"

    def build_prompt(text):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def last_logits(image, text):
        inputs = processor(images=image, text=build_prompt(text), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        return out.logits[0, -1].float()

    def two_way(logits, a_ids, b_ids):
        al = torch.logsumexp(logits[a_ids], dim=0)
        bl = torch.logsumexp(logits[b_ids], dim=0)
        p = torch.softmax(torch.stack([al, bl]), dim=0)[0].item()
        return p, al.item(), bl.item()

    def ground_readout(image, category):
        """One forward pass -> p_ground plus the raw components needed to diagnose coverage."""
        logits = last_logits(image, GROUND_PROMPT.format(obj=category))
        p, box_l, abst_l = two_way(logits, box_ids, abstain_ids)
        probs = torch.softmax(logits, dim=0)
        top_p, top_i = torch.topk(probs, 10)
        return {
            "p_ground": p,
            "box_logit": box_l,
            "abstain_logit": abst_l,
            # pre-normalization mass on the two pooled sets: if this is far below 1.0 the two-way
            # score is mis-specified for this item.
            "ground_mass_captured": (probs[box_ids].sum() + probs[abstain_ids].sum()).item(),
            "argmax_id": int(top_i[0].item()),
            "argmax_tok": tok.decode([int(top_i[0].item())]),
            "top10": [[int(i), tok.decode([int(i)]), float(pv)]
                      for i, pv in zip(top_i.tolist(), top_p.tolist())],
        }

    black_cache = {}

    def black_image(size):
        if size not in black_cache:
            black_cache[size] = Image.new("RGB", size, (0, 0, 0))
        return black_cache[size]

    t0, n_done = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for r, group in targets:
            item = items_by_uid.get(r["uid"])
            if item is None:
                continue
            img = item["image"].convert("RGB")
            img_w, img_h = img.size
            cat = r["category"]

            # --- compute tier A: one forward pass each, the matched-cost comparison ---
            p_yes_fp16, _, _ = two_way(last_logits(img, item["question"] + ANSWER_SUFFIX),
                                       yes_ids, no_ids)
            g = ground_readout(img, cat)
            g_blank = ground_readout(black_image(img.size), cat)

            rec = {
                "uid": r["uid"], "group": group, "split": r["split"],
                "question_id": r["question_id"], "category": cat,
                "pixel_area_frac": r.get("pixel_area_frac"),
                "patch_token_frac": r.get("patch_token_frac"),
                "p_yes_fp16": p_yes_fp16,
                "p_yes_4bit_cached": r["p_yes_real"],
                "p_yes_blank_cached": r.get("p_yes_blank"),
                "p_ground": g["p_ground"],
                "p_ground_blank": g_blank["p_ground"],
                "ground": g,
                "ground_blank": {k: g_blank[k] for k in
                                 ("box_logit", "abstain_logit", "ground_mass_captured", "argmax_tok")},
            }

            # --- compute tier B: full generation (~100 decode steps). NOT compute-matched. ---
            if not args.no_generate:
                inputs = processor(images=img, text=build_prompt(GROUND_PROMPT.format(obj=cat)),
                                   return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                gen = processor.decode(new_tokens, skip_special_tokens=True)
                pred = [[b[0] * img_w / 1000, b[1] * img_h / 1000,
                         b[2] * img_w / 1000, b[3] * img_h / 1000] for b in extract_boxes(gen)]
                gt = bbox_lookup.get((r["split"], str(r["question_id"])), [])
                rec.update({
                    "n_pred_boxes": len(pred),
                    "n_gt_boxes": len(gt),
                    # NOTE: circular for negatives (no GT box => IoU==0 by construction). The
                    # non-circular grounding signals are p_ground and box emission.
                    "best_iou": max([iou(p, q) for p in pred for q in gt], default=0.0),
                    "n_gen_tokens": int(new_tokens.shape[0]),
                    "raw_gen": gen[:300],
                })

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"[{n_done}/{len(targets)}] {n_done/el:.2f} it/s  eta={(len(targets)-n_done)/(n_done/el)/60:.1f}min",
                      flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
