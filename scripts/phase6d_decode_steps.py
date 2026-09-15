"""
Phase 6d: autoregressive-decoding-depth intervention (user's hypothesis, 2026-09-06). Reframes
Phase 4's localize-vs-answer dissociation and Phase 6's attention-routing signature under a single
mechanism: the forced yes/no answer is read off a SINGLE forward pass (one shot, no autoregressive
steps), while grounding requires generating ~15-20 coordinate tokens, each a fresh forward pass that
gets to re-attend to the image conditioned on more context (including its own partial output). If
one-shot decoding is itself the bottleneck -- not a fixed defect in yes/no attention specifically --
then forcing the model through a few EXTRA autoregressive steps before the yes/no token should
recover accuracy on the confident-denial cohort, and should be visible as increased attention
enrichment on the object's tokens at the (now later) answer position. This is a genuine causal
intervention, not another correlational signature like Phase 5/6 -- if it moves P(yes), that is an
actual fix, not just an explanation.

Three conditions per item, all on the SAME confident_denial cohort used in Phase 6
(data/phase6_attention_results.jsonl, group=="confident_denial"):
  1. baseline    -- the original single-shot forced-choice pass (reuses phase1/phase6's exact
                     prompt and p_yes computation).
  2. filler      -- SAME NUMBER of extra tokens before the answer, but content-free ("Hmm, let me
                     think about this.", repeated to roughly match the cot condition's token count
                     for that item) -- controls for "any extra tokens/compute help for some generic
                     reason (e.g. distributional shift away from a terse prompt)" rather than actual
                     re-attention to image content.
  3. cot         -- the model FREELY GENERATES (do_sample=False, ~32 tokens) a one-sentence
                     description of the region where the queried object would be, THEN the yes/no
                     question is re-asked with that self-generated description as context.

If cot recovers P(yes) toward correct while filler does not, that is evidence the extra
autoregressive steps help because they let the model re-attend to the image, not merely because
more tokens precede the answer. Attention enrichment (reusing phase6's object-token mapping and
enrichment_from_attention) is also measured for all 3 conditions to connect this directly to the
Phase 6 attention-routing signature: does enrichment on the object's own tokens increase from
baseline -> filler -> cot, and does any such increase track improved P(yes)?
"""
import json
import sys
import time
import torch

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import (
    Ctx, grid_from_image, object_token_indices, enrichment_from_attention, load_bbox_lookup,
    N_LAST_LAYERS,
)

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase6d_decode_steps_results.jsonl"
FILLER_UNIT = "Hmm, let me think about this. "
N_COT_TOKENS = 32


def build_base_prompt(ctx, question):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{question} Please answer this question with yes or no."},
    ]}]
    return ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_cot_prompt(ctx):
    # Deliberately does NOT name the queried category or invite a presence/absence verdict --
    # smoke-testing the category-naming version showed the model just answers "No, there is no
    # {category}" during the "description," which then anchors the final answer via self-consistency
    # rather than testing whether extra autoregressive steps help it re-attend to the image.
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "First, describe in one sentence what objects and scene elements "
                                  "you notice in this image."},
    ]}]
    return ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def p_yes_and_enrichment(ctx, image, full_text, obj_indices):
    """One forward pass over the given (already-built) text, output_attentions=True. Returns
    (p_yes, enrichment_dict_or_None) from the LAST token position -- the position where the
    yes/no answer is about to be generated, whatever context precedes it."""
    inputs = ctx.processor(images=image, text=full_text, return_tensors="pt").to(ctx.model.device)
    with torch.no_grad():
        out = ctx.model(**inputs, output_attentions=True)
    logits = out.logits[0, -1]
    yl = torch.logsumexp(logits[ctx.yes_ids], dim=0)
    nl = torch.logsumexp(logits[ctx.no_ids], dim=0)
    p_yes = torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()
    image_mask = (inputs["input_ids"][0] == ctx.image_token_id)
    rows = [layer_attn[0, :, -1, :].mean(dim=0) for layer_attn in out.attentions[-N_LAST_LAYERS:]]
    attn_from = torch.stack(rows)
    enrich = enrichment_from_attention(attn_from, image_mask, obj_indices)
    return p_yes, enrich


def generate_cot(ctx, image, cot_prompt):
    inputs = ctx.processor(images=image, text=cot_prompt, return_tensors="pt").to(ctx.model.device)
    with torch.no_grad():
        out = ctx.model.generate(**inputs, max_new_tokens=N_COT_TOKENS, do_sample=False)
    gen = ctx.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


def matched_filler(ctx, target_text):
    """Content-free filler, length-matched (in tokens) to target_text, so the filler condition
    controls for token COUNT while the cot condition additionally has image-grounded content."""
    n_target = len(ctx.tok(target_text, add_special_tokens=False)["input_ids"])
    filler = ""
    while len(ctx.tok(filler, add_special_tokens=False)["input_ids"]) < n_target:
        filler += FILLER_UNIT
    return filler.strip()


def main():
    with open(f"{DATA}/phase6_attention_results.jsonl") as f:
        phase6_recs = [json.loads(l) for l in f]
    denials = [r for r in phase6_recs if r["group"] == "confident_denial"]
    print(f"confident_denial cohort: {len(denials)}")

    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs_by_uid = {(d := json.loads(l))["uid"]: d for l in f}

    bbox_by_key, imgsize_by_key = load_bbox_lookup()

    target_uids = {r["uid"] for r in denials}
    print("Loading items...")
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

    print("Loading Qwen3-VL-2B (fp16, eager attention)...")
    ctx = Ctx()

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a") as fout:
        for r in denials:
            uid = r["uid"]
            if uid in done_uids:
                continue
            item = items_by_uid.get(uid)
            if item is None:
                continue
            rfull = recs_by_uid.get(uid)
            key = (rfull["split"], str(rfull["question_id"]))
            bboxes = bbox_by_key.get(key)
            imgsize = imgsize_by_key.get(key)
            if not bboxes or imgsize is None:
                continue
            img = item["image"].convert("RGB")
            img_w, img_h = imgsize
            grid_h, grid_w, resized_h, resized_w = grid_from_image(ctx, img)
            obj_indices = object_token_indices(bboxes, img_w, img_h, resized_w, resized_h, grid_h, grid_w)
            if not obj_indices:
                continue

            base_prompt = build_base_prompt(ctx, item["question"])
            base_py, base_enrich = p_yes_and_enrichment(ctx, img, base_prompt, obj_indices)

            cot_prompt = build_cot_prompt(ctx)
            cot_text = generate_cot(ctx, img, cot_prompt)
            cot_followup = (cot_prompt + cot_text +
                             f"\nNow answer the original question: {item['question']} "
                             f"Please answer this question with yes or no.")
            cot_py, cot_enrich = p_yes_and_enrichment(ctx, img, cot_followup, obj_indices)

            filler_text = matched_filler(ctx, cot_text)
            filler_followup = (cot_prompt + filler_text +
                                f"\nNow answer the original question: {item['question']} "
                                f"Please answer this question with yes or no.")
            filler_py, filler_enrich = p_yes_and_enrichment(ctx, img, filler_followup, obj_indices)

            rec = {"uid": uid, "category": r["category"], "pixel_area_frac": r["pixel_area_frac"],
                   "n_obj_tokens": len(obj_indices), "n_image_tokens": grid_h * grid_w,
                   "cot_text": cot_text[:200],
                   "baseline_p_yes": base_py, "baseline_attn": base_enrich,
                   "cot_p_yes": cot_py, "cot_attn": cot_enrich,
                   "filler_p_yes": filler_py, "filler_attn": filler_enrich}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
