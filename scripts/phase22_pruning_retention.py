"""
Phase 22 (E2): WHY attention-guided token pruning fails on small objects.

THE PUBLISHED, UNEXPLAINED RESULT WE ARE EXPLAINING
----------------------------------------------------
"Token Pruning in MLLMs: Are We Solving the Right Problem?" (ACL Findings 2025, arXiv 2502.11501)
reports two things it does not explain:
  (a) attention-guided pruning (FastV-style) is often BEATEN by random / uniform selection, and
  (b) pruning methods "catastrophically fail on fine-grained localization".

Phase 6c already contains the mechanism: when the model is about to deny a small object, attention
to that object's own tokens is statistically indistinguishable from attention to an object nobody
asked about. If attention does not mark the target, an attention-guided selector cannot keep it.

THE MEASUREMENT (retention, not accuracy)
------------------------------------------
Downstream accuracy is what the ACL paper already reports. The number it is MISSING -- and the one
that is the mechanism -- is **whether the target's tokens survive selection at all**:

    retention = |selected tokens  ∩  target's tokens| / |target's tokens|

Under RANDOM or UNIFORM selection, expected retention == the keep-rate r, by construction. That
makes r the exact null hypothesis, with no fitting required:

    PREDICTION (pre-registered):
      * attention-guided retention  <  r   in the sub-token strata  -> attention selects AGAINST
        the small target, which is why uniform beats it.
      * attention-guided retention  >= r   for large objects        -> where attention is informative
                                                                       and the ACL finding reverses.
      * REFUTED if attention retention >= r at every stratum: then attention does mark small
        targets and the ACL failure needs a different explanation (report as such).

Selector = mean attention over heads from the FINAL prompt position to the image tokens, at an
early layer (FastV uses layer 2; we log 2/4/8 since the choice is a free parameter of that family).

Cohort: the Phase 21 stratified sample (RePOPE-clean positives, 120 per `tokens_on_object`
stratum, range 0.08-290) so retention is read along the same size axis as the allocation law.
Memory: lean_loader (build_items peaks at 17GB -- bug #19).
"""
import json
import random
import sys
import time

import torch

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from lean_loader import LeanItems
from phase6_attention_entanglement import object_token_indices, load_bbox_lookup, PATCH, MERGE

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase22_retention_results.jsonl"
LAYERS = [2, 4, 8]
KEEP_RATES = [0.10, 0.25, 0.50]
ANSWER_SUFFIX = " Please answer this question with yes or no."


def main():
    rng = random.Random(127)
    bb, isz = load_bbox_lookup()

    src = [json.loads(l) for l in open(f"{DATA}/phase21_crossing_results.jsonl")]
    seen, pos = set(), []
    for x in src:
        if x["group"] == "positive" and x["uid"] not in seen:
            seen.add(x["uid"]); pos.append(x)
    print(f"Phase 21 stratified positives: {len(pos)}")

    done = set()
    try:
        for l in open(OUT_PATH):
            done.add(json.loads(l)["uid"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass
    pos = [x for x in pos if x["uid"] not in done]
    if not pos:
        print("Nothing to do."); return

    print("Indexing POPE (lazy)...")
    lean = LeanItems()
    pos = [x for x in pos if lean.has(x["uid"])]
    print(f"  resolvable: {len(pos)}")

    print("Loading Qwen3-VL-2B (fp16, eager attention)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for x in pos:
            it = lean.get(x["uid"])
            if it is None:
                continue
            img = it["image"].convert("RGB")
            key = (x["split"], str(x["question_id"]))
            bxs, sz = bb.get(key), isz.get(key)
            if not bxs or not sz:
                continue
            W, H = sz
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": it["question"] + ANSWER_SUFFIX}]}]
            chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=img, text=chat, return_tensors="pt").to(model.device)
            g = inputs["image_grid_thw"][0].tolist()
            gh, gw = g[1] // MERGE, g[2] // MERGE
            rz_h, rz_w = g[1] * PATCH, g[2] * PATCH
            img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) < gh * gw:
                continue
            base = img_pos[0].item()
            n_img = gh * gw

            tgt = sorted(object_token_indices(bxs, W, H, rz_w, rz_h, gh, gw))
            if not tgt:
                continue

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            rec = {"uid": x["uid"], "stratum": x["stratum"],
                   "tokens_on_object": x["tokens_on_object"],
                   "n_img_tokens": n_img, "n_target_tokens": len(tgt),
                   "target_frac": len(tgt) / n_img, "retention": {}, "attn_mass": {}}

            for L in LAYERS:
                # mean over heads of attention from the FINAL prompt position to image tokens
                a = out.attentions[L][0, :, -1, base:base + n_img].float().mean(0)
                order = torch.argsort(a, descending=True).tolist()
                # attention mass concentrated on the target, vs its share of the grid (=chance)
                rec["attn_mass"][f"L{L}"] = float(a[tgt].sum() / (a.sum() + 1e-9))
                for r in KEEP_RATES:
                    k = max(1, int(round(r * n_img)))
                    sel_attn = set(order[:k])
                    sel_unif = set(range(0, n_img, max(1, n_img // k)))
                    sel_rand = set(rng.sample(range(n_img), k))
                    for nm, sel in [("attn", sel_attn), ("uniform", sel_unif), ("random", sel_rand)]:
                        rec["retention"][f"{nm}@L{L}@{r}"] = len(sel & set(tgt)) / len(tgt)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(pos)}] {n/el:.2f} it/s eta={(len(pos)-n)/(n/el)/60:.1f}min",
                      flush=True)
    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
