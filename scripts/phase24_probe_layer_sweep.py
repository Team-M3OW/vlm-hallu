"""
Phase 24: WHERE DOES THE SIGNAL DIE? -- a full-depth probe sweep (the C2 mechanism experiment).

WHY THIS IS NOW THE PAPER'S CENTRE
----------------------------------
The 2026-09-09 prior-art check demoted C1: Q-CueGraph (2608.04452) already has the area-matched
anti-region AND shuffled-question controls on V*Bench n=191, and RUTA (2608.04132) already has the
random-region control. The behavioural allocation result is taken.

**C2 is not.** Neither concurrent paper touches internals, and three of our phases already box it in:
    Phase 14  a linear probe recovers the target from hidden states   -> it is ENCODED
    Phase 22  attention rank keeps the target +20-35pp over chance    -> it is LOCATED
    Phase 16  L16 steering gives no real recovery (rand_last control) -> one layer cannot FIX it
"Encoded, located, still unusable" is a mechanistic statement nobody in this literature has made.
But right now it is an ASSERTION backed by four nulls at scattered layers. This phase converts it
into a LOCALIZED claim: at which depth does the information stop being available?

THE MEASUREMENT
---------------
One forward pass per item with `output_hidden_states=True`, pooled at FOUR positions x ALL 29 layer
outputs (embeddings + 28 blocks) -- Phase 14 sampled only [10,16,22,27], which is why it could say
"encoded" but not "encoded until layer k":

  obj         mean over merged visual tokens covering the object's bbox        IS IT THERE?
  rand        mean over a size-matched random region, SAME item                paired control
  last        final input position, where yes/no is emitted                    WAS IT ROUTED OUT?
  last_blank  same position, BLACK image of the same size                      language-prior floor

**The two curves are the experiment.** `obj` says whether the object's own tokens carry presence
information; `last` says whether it ever reaches the position that answers. Their divergence as a
function of depth is what "irreversible" means mechanically:

  obj high across depth, last flat at the blank floor  -> NEVER ROUTED OUT of the visual tokens
  obj high early then collapsing                       -> OVERWRITTEN in a specific layer band
  both high, answer still wrong                        -> the readout has it and ignores it; C2's
                                                          framing is wrong and must be rewritten

CONTROLS ARE NOT OPTIONAL (Phase 16 produced a 55.6% "recovery" that was pure bias shift)
-----------------------------------------------------------------------------------------
No layer is called informative unless it beats BOTH `rand` (same item, same size, different place)
and `last_blank` (same prompt, no image). A probe AUROC on its own is not a finding.

THE COHORT THAT MATTERS
-----------------------
`correct` is recorded per item so the analyzer can restrict to items the model ANSWERS WRONG. On
items it already gets right, "the information is present" is unsurprising. C2 lives on the failures:
if the object is still decodable at `obj` positions on items the model confidently denies, that is
the mechanism -- not a restatement of it. Phase 14 pooled both and could not make this split.

Stratified by `tokens_on_object` throughout, since the whole domain (C3) is the sub-token regime.
Data: RePOPE-CLEAN only (Phase 11: the confident-denial cohort is 58.1% not-clean).
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase6_attention_entanglement import object_token_indices, load_bbox_lookup, PATCH, MERGE
from phase11_repope_audit import load_repope, verdict
from phase7_vision_zoom import union_bbox
from lean_loader import LeanItems

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_NPZ = f"{DATA}/phase24_probe_sweep_feats.npz"
OUT_META = f"{DATA}/phase24_probe_sweep_meta.jsonl"
N_LAYERS = 29          # hidden_states: embeddings + 28 decoder blocks
BLOCKS = ["obj", "rand", "last", "last_blank"]
ANSWER_SUFFIX = " Please answer this question with yes or no."
N_POS, N_NEG = 700, 700
NEED_GB = 9.0


def free_gb():
    return torch.cuda.mem_get_info()[0] / 1024 ** 3


def wait_for_gpu(need=NEED_GB, every=120):
    while free_gb() < need:
        print(f"  waiting for GPU: {free_gb():.1f}GB free, need {need:.1f}GB", flush=True)
        time.sleep(every)
    print(f"  GPU has {free_gb():.1f}GB free, proceeding", flush=True)


def main():
    rng = random.Random(53)      # same seed as Phase 14 -> comparable item sample
    rp = load_repope()
    bb, isz = load_bbox_lookup()

    # Phase 14 used build_items(), which peaks at 17GB RSS by decoding all 5553 POPE images.
    # LeanItems decodes one at a time (see lean_loader docstring, bug #16).
    print("Indexing POPE (lazy image access)...")
    lean = LeanItems()
    clean = [json.loads(l) for l in open(f"{DATA}/phase12_clean_pyes.jsonl")]
    clean = [x for x in clean
             if verdict(rp, (x["split"], str(x["question_id"]))) == "CLEAN" and lean.has(x["uid"])]
    # ------------------------------------------------------------------------------------------
    # FAILURE-ENRICHED SAMPLING (decided from an offline count, before spending any GPU).
    # The model is right on 95.5% of clean positives and 97.0% of clean negatives, so a uniform
    # 700/700 sample yields only ~61 wrong items -- useless for a 2048-dim probe. The ENTIRE clean
    # pool contains just 153 wrong positives and 42 wrong negatives, so we take ALL of them and
    # spend the remaining budget on CORRECT items matched to the wrong ones on `tokens_on_object`.
    #
    # The matching is not cosmetic. Error rate runs 32.8% at <0.5 tokens down to 1.0% above 32
    # (measured offline), and small objects are simultaneously harder to DECODE. An unmatched
    # correct-vs-wrong comparison would therefore confound "the model failed here" with "the object
    # is small here", and would manufacture the very result C2 predicts.
    # ------------------------------------------------------------------------------------------
    for x in clean:
        k = (x["split"], str(x["question_id"]))
        bxs, sz = bb.get(k), isz.get(k)
        if x["label"] == "yes" and bxs and sz:
            W0, H0 = sz
            g = x["image_grid_thw"]
            a, b, c2, d2 = union_bbox(bxs)
            x["tok"] = (max(0., (c2 - a) * ((g[2] * PATCH) / W0) / (PATCH * MERGE))
                        * max(0., (d2 - b) * ((g[1] * PATCH) / H0) / (PATCH * MERGE)))
        else:
            x["tok"] = None
        x["prior_correct"] = (x["p_yes_fp16"] > 0.5) == (x["label"] == "yes")

    pos = [x for x in clean if x["label"] == "yes" and x["tok"] is not None]
    neg = [x for x in clean if x["label"] == "no"]
    wrong_pos = [x for x in pos if not x["prior_correct"]]
    wrong_neg = [x for x in neg if not x["prior_correct"]]
    ok_pos = [x for x in pos if x["prior_correct"]]
    ok_neg = [x for x in neg if x["prior_correct"]]

    # size-matched correct positives: MATCH_RATIO nearest-in-`tok` partners per wrong positive
    MATCH_RATIO = 3
    pool = sorted(ok_pos, key=lambda z: z["tok"])
    used, matched = set(), []
    for w in sorted(wrong_pos, key=lambda z: z["tok"]):
        cands = sorted((z for z in pool if id(z) not in used), key=lambda z: abs(z["tok"] - w["tok"]))
        for z in cands[:MATCH_RATIO]:
            used.add(id(z)); matched.append(z)
    sel_pos = wrong_pos + matched
    sel_neg = wrong_neg + rng.sample(ok_neg, min(len(matched), len(ok_neg)))
    targets = [(x, "positive") for x in sel_pos] + [(x, "negative") for x in sel_neg]
    rng.shuffle(targets)
    import statistics as _st
    print(f"  RePOPE-clean pool: {len(pos)} pos, {len(neg)} neg")
    print(f"  FAILURE COHORT (all of it): {len(wrong_pos)} wrong positives, {len(wrong_neg)} wrong negatives")
    print(f"  size-matched correct positives: {len(matched)} "
          f"(median tok wrong={_st.median([z['tok'] for z in wrong_pos]):.2f} vs "
          f"matched={_st.median([z['tok'] for z in matched]):.2f})")
    print(f"  total forward passes: {len(targets)}")

    wait_for_gpu()
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
    start, n = time.time(), 0
    for x, group in targets:
        rec = lean.get(x["uid"])
        if rec is None:
            continue
        img = rec["image"].convert("RGB")
        key = (x["split"], str(x["question_id"]))
        W, H = isz.get(key, img.size)

        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": rec["question"] + ANSWER_SUFFIX}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=chat, return_tensors="pt").to(model.device)

        gthw = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = gthw[1] // MERGE, gthw[2] // MERGE
        rz_h, rz_w = gthw[1] * PATCH, gthw[2] * PATCH

        if group == "positive":
            boxes = bb.get(key)
            if not boxes:
                continue
            roi = [[b[0], b[1], b[2], b[3]] for b in boxes]
        else:
            bw, bh = W * rng.uniform(.05, .3), H * rng.uniform(.05, .3)
            roi = [[rng.uniform(0, max(1, W - bw)), rng.uniform(0, max(1, H - bh)), bw, bh]]
        rw, rh = W * rng.uniform(.05, .3), H * rng.uniform(.05, .3)
        rand_roi = [[rng.uniform(0, max(1, W - rw)), rng.uniform(0, max(1, H - rh)), rw, rh]]

        def idx_for(rois):
            s = object_token_indices(rois, W, H, rz_w, rz_h, grid_h, grid_w)
            zero = len(s) == 0
            if zero:   # pre-declared fallback, identical to Phase 14: nearest token to ROI centre
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
        lg = out.logits[0, -1].float()
        p_yes = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                           torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()
        black = Image.new("RGB", img.size, (0, 0, 0))
        binputs = processor(images=black, text=chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            bout = model(**binputs, output_hidden_states=True)

        # layer-major layout: [L0 obj, L0 rand, L0 last, L0 last_blank, L1 obj, ...]
        vec = []
        for L in range(N_LAYERS):
            h = out.hidden_states[L][0].float()
            vec.append(h[[base + i for i in obj_idx]].mean(0).cpu().numpy())
            vec.append(h[[base + i for i in rnd_idx]].mean(0).cpu().numpy())
            vec.append(h[-1].cpu().numpy())
            vec.append(bout.hidden_states[L][0, -1].float().cpu().numpy())
        feats.append(np.concatenate(vec).astype(np.float16))
        del out, bout
        torch.cuda.empty_cache()

        x0 = min(b[0] for b in roi); y0 = min(b[1] for b in roi)
        x1 = max(b[0] + b[2] for b in roi); y1 = max(b[1] + b[3] for b in roi)
        toks = (max(0., (x1 - x0) * (rz_w / W) / (PATCH * MERGE))
                * max(0., (y1 - y0) * (rz_h / H) / (PATCH * MERGE)))
        pred_yes = p_yes > 0.5
        metas.append({"uid": x["uid"], "group": group, "split": x["split"],
                      "question_id": x["question_id"], "p_yes_fp16": p_yes,
                      # THE COHORT SPLIT C2 NEEDS: is the model right on this item?
                      "correct": bool(pred_yes == (group == "positive")),
                      "prior_correct": x["prior_correct"],
                      "tok_offline": x["tok"],
                      "tokens_on_object": toks, "n_obj_tokens": len(obj_idx),
                      "zero_coverage": obj_zero, "grid": [grid_h, grid_w]})
        n += 1
        if n % 100 == 0:
            el = time.time() - start
            print(f"[{n}/{len(targets)}] {n/el:.2f} it/s "
                  f"eta={(len(targets)-n)/(n/el)/60:.1f}min", flush=True)

    np.savez_compressed(OUT_NPZ, feats=np.stack(feats), layers=np.arange(N_LAYERS))
    with open(OUT_META, "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"Done. {len(feats)} items -> {OUT_NPZ}. blocks per layer = {BLOCKS}")


if __name__ == "__main__":
    main()
