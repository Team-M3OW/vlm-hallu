"""
Phase 14: the GATE for every internal (activation / attention / gradient) intervention.

THE QUESTION
------------
Is object presence linearly decodable from the model's OWN internal states at the object's own
visual-token positions? Everything downstream hinges on the answer:

  * If YES -> the information is encoded but the yes/no readout fails to use it. Steering /
    attention reweighting have something to amplify, and are worth building.
  * If NO  -> nothing is encoded at that location, and NO inference-time internal intervention can
    work. That is a REPRESENTATIONAL CAPACITY FLOOR, and it explains, in one mechanism, four
    otherwise-separate results: Phase 6c (attention to the denied object looks like attention to an
    object nobody asked about), Phase 6e (attention patching: null), Phase 13 `redbox` (5.6%) and
    `prompt_coords`/`prompt_focus` (0.0% -- pointing recovers nothing).

Phase 13 measured the median confident-denial object at **0.4 merged visual tokens**. So the prior
here is strongly "NO", and this script is designed to make that answer *interpretable* rather than
to rescue a positive.

DESIGN (the controls are the whole point)
-----------------------------------------
For every item, ONE forward pass on the standard yes/no prompt, from which we pool hidden states at
three locations x several layers:

  obj_tokens   mean over merged visual tokens covering the object's COCO bbox (positives) or a
               size-matched random region (negatives). THE LOAD-BEARING PROBE.
  rand_tokens  mean over merged visual tokens covering a random region, SAME item, same size.
               Paired control: if obj ~= rand, the object's location carries no extra presence
               information and any apparent signal is global/scene/co-occurrence context.
  last_pos     final input position (where yes/no is emitted). CONFOUND CHECK ONLY -- this carries
               the question text and scene priors, so it can score high by learning "kitchens
               contain forks" with zero access to the object's pixels.

Plus a SECOND forward pass on a BLACK image of the same size, giving a language-prior floor at the
same positions (the same control that caught Phase 8's grounding channel, whose blank-image AUROC
floors were 0.407/0.356).

STRATIFICATION (pre-specified, not post-hoc): every record stores `tokens_on_object`, computed from
`image_grid_thw` + bbox exactly as in Phase 13. The interesting result is not a single AUROC but
whether decodability tracks token budget the way RECOVERY did (0.4 tok -> 0%, 1.8 -> 28%,
4.1 -> 50%). Chance-level in the sub-token stratum + rising with budget = the capacity floor with a
dose-response curve attached.

ZERO-COVERAGE RULE (decided before running): at 0.4 median tokens, the bbox often covers NO merged
token. Rule: if the covering set is empty, fall back to the single token nearest the bbox centre,
and set `zero_coverage=True`. **The fraction of confident denials with zero coverage is itself a
result** -- if a meaningful share of them have no visual token substantially covering the object,
the capacity floor is definitional, not merely empirical.

Data: RePOPE-CLEAN items only (Phase 11: the confident-denial cohort is 58.1% not-clean).
"""
import json
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import object_token_indices, load_bbox_lookup, PATCH, MERGE
from phase11_repope_audit import load_repope, verdict

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_NPZ = f"{DATA}/phase14_probe_feats.npz"
OUT_META = f"{DATA}/phase14_probe_meta.jsonl"
LAYERS = [10, 16, 22, 27]
ANSWER_SUFFIX = " Please answer this question with yes or no."
N_POS, N_NEG = 700, 700


def main():
    rng = random.Random(53)
    rp = load_repope()
    bb, isz = load_bbox_lookup()

    print("Loading items...")
    items = [it for it in build_items()
             if verdict(rp, (it["split"], str(it["question_id"]))) == "CLEAN"]
    pos = [it for it in items if it["label"] == "yes"]
    neg = [it for it in items if it["label"] == "no"]
    pos = rng.sample(pos, min(N_POS, len(pos)))
    neg = rng.sample(neg, min(N_NEG, len(neg)))
    targets = [(it, "positive") for it in pos] + [(it, "negative") for it in neg]
    print(f"  RePOPE-clean: {len(pos)} positives, {len(neg)} negatives")

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
    image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    feats, metas = [], []
    t0 = 0
    start = time.time()

    for it, group in targets:
        img = it["image"].convert("RGB")
        key = (it["split"], str(it["question_id"]))
        W, H = isz.get(key, img.size)

        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": it["question"] + ANSWER_SUFFIX}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=chat, return_tensors="pt").to(model.device)

        gthw = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = gthw[1] // MERGE, gthw[2] // MERGE
        rz_h, rz_w = gthw[1] * PATCH, gthw[2] * PATCH

        # region of interest: the true bbox for positives, a size-matched random box for negatives
        if group == "positive":
            boxes = bb.get(key)
            if not boxes:
                continue
            roi = [[b[0], b[1], b[2], b[3]] for b in boxes]  # x,y,w,h (COCO)
        else:
            bw, bh = W * rng.uniform(.05, .3), H * rng.uniform(.05, .3)
            roi = [[rng.uniform(0, max(1, W - bw)), rng.uniform(0, max(1, H - bh)), bw, bh]]
        rw, rh = W * rng.uniform(.05, .3), H * rng.uniform(.05, .3)
        rand_roi = [[rng.uniform(0, max(1, W - rw)), rng.uniform(0, max(1, H - rh)), rw, rh]]

        def idx_for(rois):
            s = object_token_indices(rois, W, H, rz_w, rz_h, grid_h, grid_w)
            zero = len(s) == 0
            if zero:  # pre-declared fallback: nearest token to the ROI centre
                cx = (rois[0][0] + rois[0][2] / 2) / W * grid_w
                cy = (rois[0][1] + rois[0][3] / 2) / H * grid_h
                c = min(max(int(cy), 0), grid_h - 1) * grid_w + min(max(int(cx), 0), grid_w - 1)
                s = {c}
            return sorted(s), zero

        obj_idx, obj_zero = idx_for(roi)
        rnd_idx, _ = idx_for(rand_roi)

        img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(img_pos) < grid_h * grid_w:
            continue
        base = img_pos[0].item()

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        logits = out.logits[0, -1].float()
        yl = torch.logsumexp(logits[yes_ids], dim=0); nl = torch.logsumexp(logits[no_ids], dim=0)
        p_yes = torch.softmax(torch.stack([yl, nl]), dim=0)[0].item()

        black = Image.new("RGB", img.size, (0, 0, 0))
        binputs = processor(images=black, text=chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            bout = model(**binputs, output_hidden_states=True)

        vec = []
        for L in LAYERS:
            h = out.hidden_states[L][0].float()
            vec.append(h[[base + i for i in obj_idx]].mean(0).cpu().numpy())
            vec.append(h[[base + i for i in rnd_idx]].mean(0).cpu().numpy())
            vec.append(h[-1].cpu().numpy())
            vec.append(bout.hidden_states[L][0, -1].float().cpu().numpy())
        feats.append(np.concatenate(vec).astype(np.float16))

        # tokens on object, same formula as Phase 13
        x0 = min(b[0] for b in roi); y0 = min(b[1] for b in roi)
        x1 = max(b[0] + b[2] for b in roi); y1 = max(b[1] + b[3] for b in roi)
        toks = max(0., (x1-x0) * (rz_w/W) / (PATCH*MERGE)) * max(0., (y1-y0) * (rz_h/H) / (PATCH*MERGE))
        metas.append({"uid": it["uid"], "group": group, "split": it["split"],
                      "question_id": it["question_id"], "category": it["category"],
                      "pixel_area_frac": it["pixel_area_frac"], "p_yes_fp16": p_yes,
                      "tokens_on_object": toks, "n_obj_tokens": len(obj_idx),
                      "zero_coverage": obj_zero, "grid": [grid_h, grid_w]})
        t0 += 1
        if t0 % 100 == 0:
            el = time.time() - start
            print(f"[{t0}/{len(targets)}] {t0/el:.2f} it/s eta={(len(targets)-t0)/(t0/el)/60:.1f}min",
                  flush=True)

    np.savez_compressed(OUT_NPZ, feats=np.stack(feats), layers=np.array(LAYERS))
    with open(OUT_META, "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"Done. {len(feats)} items -> {OUT_NPZ} (+ meta). "
          f"blocks per layer = [obj, rand, last, last_blank]")


if __name__ == "__main__":
    main()
