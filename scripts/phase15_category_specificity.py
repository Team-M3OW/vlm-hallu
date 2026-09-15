"""
Phase 15: the category-specificity control for Phase 14.

THE CONFOUND
------------
Phase 14 found presence linearly decodable from the object's own visual tokens (obj AUROC 0.88 in
the sub-token stratum vs 0.59 for a paired random region). But the comparison was:
    positives -> features at the TRUE bbox
    negatives -> features at a RANDOM region
so a probe could score high by detecting OBJECTNESS ("something is here") rather than the QUERIED
CATEGORY ("a fork is here"). The random-region control rules out global/scene context; it does not
rule this out.

THE CONTROL
-----------
Entirely WITHIN positives, and PAIRED within the same image and the same forward pass:
    region A = the queried category's bbox        (label 1)
    region B = a DIFFERENT COCO category's bbox   (label 0)
Same image, same prompt, same global context, same scene statistics -- the only difference is which
object sits at the probed location. If a probe separates A from B, the encoding at those tokens is
CATEGORY-SPECIFIC and a steering direction has a real target. If it collapses to chance, Phase 14
was reading objectness and the readout-gap story weakens a lot.

SIZE MATCHING (essential, or the control is worthless)
------------------------------------------------------
The queried objects in this cohort are tiny (median 0.16 merged tokens for confident denials). If
region B were an arbitrary other object it would usually be much LARGER, and the probe could
separate A from B on size/token-count alone -- reproducing the confound in a new form. So B is
chosen as the other-category instance whose token area is CLOSEST to A's, and both areas are logged
so the residual size gap is checkable rather than assumed.

Reported overall and stratified by tokens_on_object, the same pre-specified bins as Phase 14.
RePOPE-clean items only.
"""
import json
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase1_eval import build_items
from phase6_attention_entanglement import object_token_indices, PATCH, MERGE
from phase11_repope_audit import load_repope, verdict

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_NPZ = f"{DATA}/phase15_catspec_feats.npz"
OUT_META = f"{DATA}/phase15_catspec_meta.jsonl"
LAYERS = [10, 16, 22, 27]
ANSWER_SUFFIX = " Please answer this question with yes or no."
N_POS = 900


def build_full_lookup():
    """(split, qid) -> (image_wh, {category_id: [COCO bboxes]}) for ALL categories in that image."""
    joined = json.load(open(f"{DATA}/pope_coco_area_joined.json"))
    coco = json.load(open(f"{DATA}/coco_ann/instances_val2014.json"))
    cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    img_info = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    out = {}
    for j in joined:
        iid = int(re.search(r"(\d+)\.jpg$", j["image"]).group(1))
        info = img_info.get(iid)
        if info is None:
            continue
        by_cat = defaultdict(list)
        for a in anns.get(iid, []):
            by_cat[a["category_id"]].append(a["bbox"])
        out[(j["split"], str(j["question_id"]))] = (
            (info["width"], info["height"]), dict(by_cat), cat_name_to_id.get(j["category"]))
    return out


def main():
    rng = random.Random(67)
    rp = load_repope()
    lut = build_full_lookup()

    print("Loading items...")
    pos = [it for it in build_items()
           if it["label"] == "yes"
           and verdict(rp, (it["split"], str(it["question_id"]))) == "CLEAN"]
    pos = rng.sample(pos, min(N_POS, len(pos)))
    print(f"  RePOPE-clean positives: {len(pos)}")

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
    for it in pos:
        key = (it["split"], str(it["question_id"]))
        ent = lut.get(key)
        if ent is None:
            continue
        (W, H), by_cat, qcat = ent
        if qcat is None or qcat not in by_cat:
            continue
        others = {c: b for c, b in by_cat.items() if c != qcat}
        if not others:
            continue

        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": it["question"] + ANSWER_SUFFIX}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = it["image"].convert("RGB")
        inputs = processor(images=img, text=chat, return_tensors="pt").to(model.device)
        gthw = inputs["image_grid_thw"][0].tolist()
        grid_h, grid_w = gthw[1] // MERGE, gthw[2] // MERGE
        rz_h, rz_w = gthw[1] * PATCH, gthw[2] * PATCH

        def tokarea(bxs):
            x0 = min(b[0] for b in bxs); y0 = min(b[1] for b in bxs)
            x1 = max(b[0] + b[2] for b in bxs); y1 = max(b[1] + b[3] for b in bxs)
            return (max(0., (x1-x0) * (rz_w/W) / (PATCH*MERGE))
                    * max(0., (y1-y0) * (rz_h/H) / (PATCH*MERGE)))

        qbxs = by_cat[qcat]
        qa = tokarea(qbxs)
        # size-matched distractor: the other-category instance closest in token area
        cand = [([b], abs(tokarea([b]) - qa)) for bxs in others.values() for b in bxs]
        obxs = min(cand, key=lambda x: x[1])[0]
        oa = tokarea(obxs)

        def idx_for(bxs):
            s = object_token_indices(bxs, W, H, rz_w, rz_h, grid_h, grid_w)
            if not s:
                x0 = min(b[0] for b in bxs); y0 = min(b[1] for b in bxs)
                x1 = max(b[0]+b[2] for b in bxs); y1 = max(b[1]+b[3] for b in bxs)
                cx = (x0+x1)/2/W*grid_w; cy = (y0+y1)/2/H*grid_h
                s = {min(max(int(cy),0),grid_h-1)*grid_w + min(max(int(cx),0),grid_w-1)}
            return sorted(s)

        qi, oi = idx_for(qbxs), idx_for(obxs)
        img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(img_pos) < grid_h * grid_w:
            continue
        base = img_pos[0].item()

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        lg = out.logits[0, -1].float()
        p_yes = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                           torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()
        vec = []
        for L in LAYERS:
            h = out.hidden_states[L][0].float()
            vec.append(h[[base + i for i in qi]].mean(0).cpu().numpy())   # queried category
            vec.append(h[[base + i for i in oi]].mean(0).cpu().numpy())   # other category
        feats.append(np.concatenate(vec).astype(np.float16))
        metas.append({"uid": it["uid"], "split": it["split"], "question_id": it["question_id"],
                      "category": it["category"], "p_yes_fp16": p_yes,
                      "tokens_on_object": qa, "tokens_on_other": oa,
                      "n_q_tokens": len(qi), "n_o_tokens": len(oi)})
        n += 1
        if n % 100 == 0:
            el = time.time() - start
            print(f"[{n}/{len(pos)}] {n/el:.2f} it/s eta={(len(pos)-n)/(n/el)/60:.1f}min", flush=True)

    np.savez_compressed(OUT_NPZ, feats=np.stack(feats), layers=np.array(LAYERS))
    with open(OUT_META, "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"Done. {len(feats)} paired items. blocks per layer = [queried, other]")


if __name__ == "__main__":
    main()
