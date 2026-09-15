"""
Phase 6: image/text token entanglement test (user's hypothesis, 2026-09-03). Operationalization:
does the model's attention correctly route to the queried object's image tokens when answering the
yes/no presence question, even on items it gets WRONG -- given Phase 4 already showed the model CAN
localize the same object accurately when asked to ground it? If attention during yes/no answering is
diffuse/misdirected while attention during grounding concentrates correctly on the object's tokens,
that is direct evidence that the failure is a TASK-DEPENDENT ATTENTION-ROUTING problem: the visual
information is present and reachable (grounding proves it), but the yes/no task's text-conditioned
attention pattern fails to pull it into the answer decision. This is a different, more specific
culprit hypothesis than the (falsified) logit-lens "detected-then-suppressed" story -- it is about
WHERE attention looks, not what an intermediate layer's projected output says.

Method:
1. Map each item's COCO ground-truth bbox to Qwen3-VL's actual image-token index set, using
   `image_grid_thw` from the processor (pre-merge patch grid) and the known merge_size=2, patch=14
   -- verified empirically (phase6 smoke test) that the mapping math is correct, and that image
   tokens form a contiguous block in the input sequence, arranged row-major on a
   (grid_h/2) x (grid_w/2) token grid, each token covering a 28x28px region of the RESIZED image.
2. YES/NO attention: one forced-choice forward pass (as in phase1), output_attentions=True,
   extract attention FROM the last input token position (where "yes"/"no" is about to be generated)
   TO every image-token position, averaged over heads and over the last few layers (post-hoc
   decision layers, per phase5's finding that the decision crystallizes late).
3. GROUNDING attention: teacher-forced forward pass over [grounding prompt + the model's own saved
   generated text from phase4, up to and including the first bbox_2d array], output_attentions=True,
   attention FROM the token positions spanning the 4 coordinate numbers TO every image-token
   position, same averaging.
4. Enrichment score = (attention mass on object's token set / number of object tokens) /
   (attention mass on all image tokens / total number of image tokens) -- i.e. attention density on
   the object's own tokens relative to the image-wide average density. 1.0 = no preference, >1 =
   object tokens get more attention than average (correct focus), <1 = object tokens are
   under-attended relative to average.

Compares two groups: confident_denial (yes/no wrong, grounding already shown accurate in phase4),
confident_correct_small (yes/no right, same area bin), on BOTH the yes/no-attention and (for
confident_denial only, reusing phase4's saved generations) grounding-attention enrichment scores.
"""
import json
import re
import sys
import time
from collections import defaultdict
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase6_attention_results.jsonl"
PATCH = 14
MERGE = 2
N_LAST_LAYERS = 4
N_PER_GROUP = 150


class Ctx:
    def __init__(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model.eval()
        self.tok = self.processor.tokenizer
        self.yes_ids = sorted({self.tok(s, add_special_tokens=False)["input_ids"][-1]
                                for s in ["yes", "Yes", " yes", " Yes"]})
        self.no_ids = sorted({self.tok(s, add_special_tokens=False)["input_ids"][-1]
                               for s in ["no", "No", " no", " No"]})
        self.image_token_id = self.tok.convert_tokens_to_ids("<|image_pad|>")


def _single_bbox_token_indices(bbox_orig, img_w, img_h, resized_w, resized_h, grid_h, grid_w):
    """Map ONE COCO bbox (in ORIGINAL image pixel coords) to the set of merged-token indices
    (row-major on a grid_h x grid_w token grid) it overlaps, in the RESIZED image frame Qwen
    actually processes. Verified empirically: bbox [100,100,50,50] in a 640x480->560x420 resize
    with a 15x20 token grid maps to indices {63,64,83,84} (rows 3-4, cols 3-4), matching hand
    computation exactly."""
    sx, sy = resized_w / img_w, resized_h / img_h
    x0, y0 = bbox_orig[0] * sx, bbox_orig[1] * sy
    x1, y1 = (bbox_orig[0] + bbox_orig[2]) * sx, (bbox_orig[1] + bbox_orig[3]) * sy
    tok_size = PATCH * MERGE
    col0, col1 = max(0, int(x0 // tok_size)), min(grid_w - 1, int((x1 - 1) // tok_size))
    row0, row1 = max(0, int(y0 // tok_size)), min(grid_h - 1, int((y1 - 1) // tok_size))
    return {r * grid_w + c for r in range(row0, row1 + 1) for c in range(col0, col1 + 1)}


def object_token_indices(bboxes_orig, img_w, img_h, resized_w, resized_h, grid_h, grid_w):
    """Union of token indices over ALL instances of the queried category in the image (bboxes_orig
    is a list), not just the single largest instance. Matches how the rest of this project defines
    object "evidence" (patch_token_frac / pixel_area_frac are sums over all instances) and matches
    Phase 4's IoU check, which scored a prediction against ANY ground-truth instance -- using only
    the biggest instance here would silently test attention to a different region than what Phase 4
    verified the model could localize, for any category with >1 instance in an image."""
    out = set()
    for bbox in bboxes_orig:
        out |= _single_bbox_token_indices(bbox, img_w, img_h, resized_w, resized_h, grid_h, grid_w)
    return out


def grid_from_image(ctx, image):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
    probe_text = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    probe_inputs = ctx.processor(images=image, text=probe_text, return_tensors="pt")
    grid_thw = probe_inputs["image_grid_thw"][0].tolist()
    grid_h, grid_w = grid_thw[1] // MERGE, grid_thw[2] // MERGE
    resized_h, resized_w = grid_thw[1] * PATCH, grid_thw[2] * PATCH
    return grid_h, grid_w, resized_h, resized_w


def p_yes_final(ctx, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{question} Please answer this question with yes or no."},
    ]}]
    text = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ctx.processor(images=image, text=text, return_tensors="pt").to(ctx.model.device)
    with torch.no_grad():
        out = ctx.model(**inputs)
    logits = out.logits[0, -1]
    yl = torch.logsumexp(logits[ctx.yes_ids], dim=0)
    nl = torch.logsumexp(logits[ctx.no_ids], dim=0)
    return torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()


def enrichment_from_attention(attn_from_positions, image_mask, object_indices_in_image_order):
    """attn_from_positions: [n_query_positions, seq_len] averaged attention rows.
    image_mask: bool tensor [seq_len], True at image-token positions (in sequence order).
    object_indices_in_image_order: set of indices INTO the image-token subsequence (0-based,
    row-major on the token grid) that the object occupies.

    Returns raw (obj_attn_mass, n_obj_tokens, total_img_attn_mass, n_img_tokens) rather than a
    single ratio -- for cohorts where objects map to only 1-4 tokens (the norm in the smallest
    area bin), a per-item density RATIO is dominated by single-token noise. Saving raw components
    lets analysis use a POOLED ratio (sum of numerators / sum of denominators across many items),
    which is far more robust than averaging noisy per-item ratios -- caught in the phase6 smoke
    test before running at scale on a metric that would have been mostly noise."""
    img_positions = image_mask.nonzero().flatten()
    n_img = len(img_positions)
    obj_seq_positions = [img_positions[i].item() for i in object_indices_in_image_order if i < n_img]
    if not obj_seq_positions:
        return None
    mean_row = attn_from_positions.mean(dim=0)
    total_img_attn = mean_row[img_positions].sum().item()
    obj_attn = mean_row[obj_seq_positions].sum().item()
    n_obj = len(obj_seq_positions)
    return {"obj_attn_mass": obj_attn, "n_obj_tokens": n_obj,
            "total_img_attn_mass": total_img_attn, "n_img_tokens": n_img}


def yes_no_enrichment(ctx, image, question, obj_indices):
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{question} Please answer this question with yes or no."},
    ]}]
    text = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ctx.processor(images=image, text=text, return_tensors="pt").to(ctx.model.device)
    with torch.no_grad():
        out = ctx.model(**inputs, output_attentions=True)
    image_mask = (inputs["input_ids"][0] == ctx.image_token_id)
    rows = [layer_attn[0, :, -1, :].mean(dim=0) for layer_attn in out.attentions[-N_LAST_LAYERS:]]
    attn_from = torch.stack(rows)
    return enrichment_from_attention(attn_from, image_mask, obj_indices)


def grounding_enrichment(ctx, image, category, obj_indices, saved_gen):
    m = re.search(r'"bbox_2d"\s*:\s*\[[^\]]*\]', saved_gen)
    if not m:
        return None
    prefix_text = saved_gen[:m.end()]
    prompt_q = f"Locate the {category} in the image and output its bounding box coordinates."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_q}]}]
    prompt = ctx.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt + prefix_text
    inputs = ctx.processor(images=image, text=full_text, return_tensors="pt").to(ctx.model.device)
    prompt_only = ctx.processor(images=image, text=prompt, return_tensors="pt").to(ctx.model.device)
    n_prompt_tokens = prompt_only["input_ids"].shape[1]
    with torch.no_grad():
        out = ctx.model(**inputs, output_attentions=True)
    image_mask = (inputs["input_ids"][0] == ctx.image_token_id)
    rows = []
    for layer_attn in out.attentions[-N_LAST_LAYERS:]:
        seg = layer_attn[0, :, n_prompt_tokens:, :].mean(dim=1)
        rows.append(seg.mean(dim=0))
    attn_from = torch.stack(rows)
    return enrichment_from_attention(attn_from, image_mask, obj_indices)


def load_bbox_lookup():
    with open(f"{DATA}/pope_coco_area_joined.json") as f:
        joined = json.load(f)
    with open(f"{DATA}/coco_ann/instances_val2014.json") as f:
        coco = json.load(f)
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    img_id_to_info = {im["id"]: im for im in coco["images"]}
    anns_by_image_cat = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image_cat[(a["image_id"], a["category_id"])].append(a)

    def coco_image_id_from_filename(fn):
        return int(re.search(r"(\d+)\.jpg$", fn).group(1))

    bbox_by_key, imgsize_by_key = {}, {}
    for j in joined:
        key = (j["split"], str(j["question_id"]))
        img_id = coco_image_id_from_filename(j["image"])
        cat_id = cat_name_to_id.get(j["category"])
        info = img_id_to_info.get(img_id)
        if cat_id is None or info is None:
            continue
        instances = anns_by_image_cat.get((img_id, cat_id), [])
        if not instances:
            continue
        bbox_by_key[key] = [a["bbox"] for a in instances]
        imgsize_by_key[key] = (info["width"], info["height"])
    return bbox_by_key, imgsize_by_key


def main():
    with open(f"{DATA}/phase1_results_qwen_dedup.jsonl") as f:
        recs = [json.loads(l) for l in f]
    positives = [r for r in recs if r["label"] == "yes" and r.get("pixel_area_frac") is not None]

    bbox_by_key, imgsize_by_key = load_bbox_lookup()

    phase4_by_uid = {}
    try:
        with open(f"{DATA}/phase4_localize_results.jsonl") as f:
            for line in f:
                r4 = json.loads(line)
                phase4_by_uid[r4["uid"]] = r4
    except FileNotFoundError:
        pass

    candidate_denials_4bit = [r for r in positives if r["p_yes_real"] < 0.05]
    # Prioritize denial candidates that HAVE a saved phase4 grounding generation -- without one,
    # grounding_enrichment is always None and the item is useless for the yes/no-vs-grounding
    # comparison that is the whole point of this test.
    candidate_denials_4bit.sort(key=lambda r: r["uid"] not in phase4_by_uid)
    small_bin = [r for r in positives if r["pixel_area_frac"] < 0.02]
    candidate_correct_small_4bit = [r for r in small_bin if r["p_yes_real"] > 0.90]

    candidate_uids = {r["uid"] for r in candidate_denials_4bit[:N_PER_GROUP * 2]} | \
                      {r["uid"] for r in candidate_correct_small_4bit[:N_PER_GROUP * 2]}
    print(f"candidate pool (2x oversample per group, will stop at N_PER_GROUP={N_PER_GROUP} "
          f"post fp16-reverification): {len(candidate_uids)} unique uids")

    print("Loading items...")
    all_items = build_items()
    items_by_uid = {it["uid"]: it for it in all_items if it["uid"] in candidate_uids}
    print(f"  resolved {len(items_by_uid)}/{len(candidate_uids)}")

    print("Loading Qwen3-VL-2B (fp16, eager attention)...")
    ctx = Ctx()

    done_uids, group_counts = set(), defaultdict(int)
    try:
        with open(OUT_PATH) as f:
            for line in f:
                d = json.loads(line)
                done_uids.add(d["uid"])
                group_counts[d["group"]] += 1
        print(f"Resuming: {len(done_uids)} already done "
              f"(confident_denial={group_counts['confident_denial']}, "
              f"confident_correct_small={group_counts['confident_correct_small']})")
    except FileNotFoundError:
        pass

    t0 = time.time()
    n_done = 0
    loop_pool = candidate_denials_4bit[:N_PER_GROUP * 2] + candidate_correct_small_4bit[:N_PER_GROUP * 2]
    with open(OUT_PATH, "a") as fout:
        for r in loop_pool:
            if group_counts["confident_denial"] >= N_PER_GROUP and \
               group_counts["confident_correct_small"] >= N_PER_GROUP:
                break
            if r["uid"] in done_uids:
                continue
            item = items_by_uid.get(r["uid"])
            if item is None:
                continue
            key = (r["split"], str(r["question_id"]))
            bboxes = bbox_by_key.get(key)
            imgsize = imgsize_by_key.get(key)
            if not bboxes or imgsize is None:
                continue
            img = item["image"].convert("RGB")

            fp16_py = p_yes_final(ctx, img, item["question"])
            is_denial_candidate = r["p_yes_real"] < 0.05
            is_correct_candidate = r in candidate_correct_small_4bit
            if is_denial_candidate and not is_correct_candidate and fp16_py >= 0.01:
                continue
            if is_correct_candidate and fp16_py <= 0.95:
                continue
            group = "confident_denial" if fp16_py < 0.01 else "confident_correct_small"
            if group_counts[group] >= N_PER_GROUP:
                continue

            grid_h, grid_w, resized_h, resized_w = grid_from_image(ctx, img)
            img_w, img_h = imgsize
            obj_indices = object_token_indices(bboxes, img_w, img_h, resized_w, resized_h, grid_h, grid_w)
            if not obj_indices:
                continue

            yn_enrich = yes_no_enrichment(ctx, img, item["question"], obj_indices)

            ground_enrich = None
            saved = phase4_by_uid.get(r["uid"])
            if saved is not None and saved.get("raw_gen"):
                try:
                    ground_enrich = grounding_enrichment(ctx, img, r["category"], obj_indices, saved["raw_gen"])
                except Exception as e:
                    print(f"  grounding-attention error on {r['uid']}: {e}")

            rec = {"uid": r["uid"], "group": group, "category": r["category"],
                   "pixel_area_frac": r["pixel_area_frac"], "fp16_p_yes": fp16_py,
                   "n_obj_tokens": len(obj_indices), "n_image_tokens": grid_h * grid_w,
                   "yesno_attn": yn_enrich, "grounding_attn": ground_enrich}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
            group_counts[group] += 1
            if n_done % 20 == 0:
                print(f"[{n_done}] rate={n_done/(time.time()-t0):.2f}/s")

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
