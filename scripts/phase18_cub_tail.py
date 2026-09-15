"""
Phase 18 (E2a): does the confident-error tail exist on FINE-GRAINED discrimination?

WHY THIS IS LOAD-BEARING, NOT JUST "GENERALIZATION"
---------------------------------------------------
On RePOPE-clean data the phenomenon is **1.6% of positives** (54/3387), and `P(yes)` scores 0.9725
AUROC on POPE overall. A reviewer will say POPE is saturated and the subject is a rounding error.
If the confident-error tail is substantially larger on fine-grained species discrimination, the
paper's subject stops being a tail.

CUB also separates two things POPE conflates. POPE's failures are *small* objects (median 0.16
merged tokens). CUB birds FILL much of the frame. So:
  * If CUB shows a large tail AND the objects are large -> the failure is about FINE DETAIL, not
    object size, and our token-budget thesis needs to be stated more carefully (or does not
    transfer). That is a genuinely useful negative.
  * If CUB shows little tail -> the phenomenon is specific to small objects, which sharpens the
    claim rather than weakening it.
Either way this is worth knowing BEFORE running the allocation experiment on CUB.

TASK CONSTRUCTION (benchmark repurposing, not dataset building)
---------------------------------------------------------------
Published CUB-200-2011 labels and the published train/test split; **TEST split only**. For each
image, the same yes/no prompt used everywhere in this project, asked three ways:

    positive       the TRUE species
    hard_negative  a DIFFERENT species sharing the same genus token (the last underscore-separated
                   word of the class name, e.g. "Sooty_Albatross" vs "Laysan_Albatross").
                   This is the confusable pair -- the fine-grained discrimination that matters.
    easy_negative  a random species from a DIFFERENT genus

Both negative arms are reported separately: CUB has no absent-object negatives, so the
false-positive control this project has held to all day has to be CONSTRUCTED here, not inherited.

Also logged per item: bbox area fraction and `tokens_on_object` (same formula as Phase 13/14), so
the CUB tail can be compared to POPE's on the token-budget axis rather than by eye.
"""
import json
import random
import sys
import time
from collections import defaultdict

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase6_attention_entanglement import PATCH, MERGE

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ROOT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/cub/CUB_200_2011"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase18_cub_tail.jsonl"
ANSWER_SUFFIX = " Please answer this question with yes or no."
N_ITEMS = 1200


def pretty(cls_name):
    """'001.Black_footed_Albatross' -> 'Black footed Albatross'"""
    return cls_name.split(".", 1)[1].replace("_", " ").strip()


def genus(cls_name):
    return cls_name.split(".", 1)[1].split("_")[-1].lower()


def main():
    rng = random.Random(97)

    classes = {}
    for line in open(f"{ROOT}/classes.txt"):
        i, name = line.split()
        classes[int(i)] = name
    images = {}
    for line in open(f"{ROOT}/images.txt"):
        i, p = line.split()
        images[int(i)] = p
    labels = {}
    for line in open(f"{ROOT}/image_class_labels.txt"):
        i, c = line.split()
        labels[int(i)] = int(c)
    boxes = {}
    for line in open(f"{ROOT}/bounding_boxes.txt"):
        parts = line.split()
        boxes[int(parts[0])] = [float(v) for v in parts[1:]]   # x, y, w, h
    is_test = {}
    for line in open(f"{ROOT}/train_test_split.txt"):
        i, t = line.split()
        is_test[int(i)] = (t == "0")   # 0 == test in CUB's convention

    by_genus = defaultdict(list)
    for cid, name in classes.items():
        by_genus[genus(name)].append(cid)

    test_ids = [i for i in images if is_test.get(i)]
    rng.shuffle(test_ids)
    test_ids = test_ids[:N_ITEMS]
    print(f"CUB test split: {sum(1 for i in images if is_test.get(i))} images, using {len(test_ids)}")
    print(f"genera with >=2 species: {sum(1 for g, v in by_genus.items() if len(v) >= 2)}/{len(by_genus)}")

    done = set()
    try:
        for line in open(OUT_PATH):
            done.add(json.loads(line)["image_id"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass
    test_ids = [i for i in test_ids if i not in done]

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

    def p_yes(img, species):
        q = f"Is there a {species} in the image?"
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q + ANSWER_SUFFIX}]}]
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=img, text=chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**inputs).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                       torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()
        return p, inputs["image_grid_thw"][0].tolist()

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for iid in test_ids:
            cid = labels[iid]
            cname = classes[cid]
            g = genus(cname)
            same = [c for c in by_genus[g] if c != cid]
            if not same:
                continue
            hard = classes[rng.choice(same)]
            other = [c for c in classes if genus(classes[c]) != g]
            easy = classes[rng.choice(other)]

            img = Image.open(f"{ROOT}/images/{images[iid]}").convert("RGB")
            W, H = img.size
            x, y, bw, bh = boxes[iid]

            p_pos, gthw = p_yes(img, pretty(cname))
            p_hard, _ = p_yes(img, pretty(hard))
            p_easy, _ = p_yes(img, pretty(easy))

            rz_w, rz_h = gthw[2] * PATCH, gthw[1] * PATCH
            toks = (bw * (rz_w / W) / (PATCH * MERGE)) * (bh * (rz_h / H) / (PATCH * MERGE))

            fout.write(json.dumps({
                "image_id": iid, "class": cname, "genus": g,
                "species": pretty(cname), "hard_species": pretty(hard),
                "easy_species": pretty(easy),
                "p_yes_true": p_pos, "p_yes_hard_neg": p_hard, "p_yes_easy_neg": p_easy,
                "bbox_area_frac": (bw * bh) / (W * H),
                "tokens_on_object": toks,
                "total_tokens": gthw[1] * gthw[2] // 4,
                "img_wh": [W, H]}) + "\n")
            fout.flush()
            n += 1
            if n % 100 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(test_ids)}] {n/el:.2f} it/s "
                      f"eta={(len(test_ids)-n)/(n/el)/60:.1f}min", flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
