"""
Phase 21: WHERE DOES QUERY-CONDITIONAL ALLOCATION STOP PAYING? (the crossing-point experiment)

WHY THIS, AND WHY IT IS NOT "does allocation hurt?"
---------------------------------------------------
Every allocation result so far was measured where allocation should win (sub-token targets). The
free V*Bench category/size split already showed the benefit DECAYS monotonically with target size:

    tokens on target   0.120 -> +39.7pp      0.629 -> +19.1pp      6.022 -> +13.9pp
    (direct_attributes +30.5pp vs relative_position +14.5pp)

...but never crosses zero, because V*Bench's *largest* targets are only ~6 tokens. So the open
question is not whether allocation hurts but **where the curve crosses zero, if it does** -- and
that needs a target-size range wide enough to contain the crossing.

POPE supplies it: RePOPE-clean positives span **0.08 to 290 merged tokens on object**
(p1..p99), computed offline from the `image_grid_thw` stored by Phase 12 plus the COCO bbox.

DESIGN
------
* **Stratified BY DESIGN on tokens_on_object**, not sampled broadly and binned after -- broad
  sampling overweights the middle and leaves the tails thin, and the tails are where the crossing
  lives. Five strata: [0,0.5) [0.5,2) [2,8) [8,32) [32,inf).
* **ALL confidence levels**, not the confident-denial cohort. A negative delta is unobservable on
  items uniform already gets wrong; we need items uniform gets RIGHT so allocation has something to
  lose.
* `alloc_random` travels with every stratum, so the decider is present along the whole curve.
* Negatives (clean POPE negatives) give the false-positive rate; they have no object size, so one
  pool serves all strata.

PRE-REGISTERED PREDICTION (written before the run)
--------------------------------------------------
For large, well-resolved objects `alloc_query` **halves the full-scene resolution to fund a crop it
does not need**, so the delta should go NEGATIVE. If a crossing appears, the follow-up question is
whether it is "allocation stops helping" or "the fixed 50/50 split is wrong" -- so a THIRD arm
varies the split (`alloc_query25`: crop gets 25% of budget, scene keeps 75%). If the crossing moves
with the split, the effect is about the budget division, not about allocation failing.

Metric: accuracy, i.e. P(yes)>0.5 on positives (correct) and P(yes)<=0.5 on negatives.
All arms matched on REALIZED tokens via the Phase 17 calibration loop.
"""
import json
import random
import sys
import time

import torch

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from lean_loader import LeanItems
from phase6_attention_entanglement import load_bbox_lookup, PATCH, MERGE
from phase7_vision_zoom import union_bbox, padded_crop_box, PAD_FRAC, CONNECTOR_TEXT
from phase17_budget_allocation import fit_to_budget

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase21_crossing_results.jsonl"
BUDGET = 300
STRATA = [(0, 0.5), (0.5, 2), (2, 8), (8, 32), (32, 1e9)]
PER_STRATUM = 120
N_NEG = 200
ANSWER_SUFFIX = " Please answer this question with yes or no."


def main():
    rng = random.Random(113)
    bb, isz = load_bbox_lookup()
    clean = [json.loads(l) for l in open(f"{DATA}/phase12_clean_pyes.jsonl")]

    pos = []
    for x in clean:
        if x["label"] != "yes":
            continue
        key = (x["split"], str(x["question_id"]))
        bxs, sz = bb.get(key), isz.get(key)
        if not bxs or not sz:
            continue
        W, H = sz
        g = x["image_grid_thw"]
        rz_w, rz_h = g[2] * PATCH, g[1] * PATCH
        x0, y0, x1, y1 = union_bbox(bxs)
        t = (max(0., (x1 - x0) * (rz_w / W) / (PATCH * MERGE))
             * max(0., (y1 - y0) * (rz_h / H) / (PATCH * MERGE)))
        x["tokens_on_object"] = t
        pos.append(x)

    sampled = []
    for lo, hi in STRATA:
        pool = [x for x in pos if lo <= x["tokens_on_object"] < hi]
        take = rng.sample(pool, min(PER_STRATUM, len(pool)))
        for x in take:
            x["stratum"] = f"[{lo},{hi})"
        sampled += take
        print(f"  stratum [{lo},{hi}): pool {len(pool)} -> sampled {len(take)}")
    negs = rng.sample([x for x in clean if x["label"] == "no"], N_NEG)
    for x in negs:
        x["stratum"] = "negative"
    targets = [(x, "positive") for x in sampled] + [(x, "negative") for x in negs]
    print(f"total {len(targets)} items")

    done = set()
    try:
        for l in open(OUT_PATH):
            done.add(json.loads(l)["uid"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass
    targets = [t for t in targets if t[0]["uid"] not in done]
    if not targets:
        print("Nothing to do.")
        return

    # LAZY image access: build_items() peaks at 17GB by decoding all 5553 POPE images at once,
    # which OOMs under current memory pressure. LeanItems decodes one image at a time.
    print("Indexing POPE (lazy image access)...")
    lean = LeanItems()
    targets = [(x, g) for x, g in targets if lean.has(x["uid"])]
    print(f"  resolvable targets: {len(targets)}")

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

    def _chat(imgs, q, conn):
        if len(imgs) == 1:
            content = [{"type": "image"}]
        else:
            content = ([{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                        {"type": "image"}] if conn else [{"type": "image"}, {"type": "image"}])
        content.append({"type": "text", "text": q + ANSWER_SUFFIX})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    def measure(imgs, q="x", conn=True):
        i = processor(images=imgs, text=_chat(imgs, q, conn), return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    def run(imgs, q, conn):
        i = processor(images=imgs, text=_chat(imgs, q, conn), return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                       torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()
        return p, int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for x, group in targets:
            it = lean.get(x["uid"])
            if it is None:
                continue
            img = it["image"].convert("RGB")
            key = (x["split"], str(x["question_id"]))
            W, H = isz.get(key, img.size)
            if group == "positive":
                bxs = bb.get(key)
                if not bxs:
                    continue
                a, b, c, d = union_bbox(bxs)
                cbox = padded_crop_box(a, b, c, d, W, H, PAD_FRAC)
            else:
                bw, bh = W * rng.uniform(.08, .35), H * rng.uniform(.08, .35)
                gx = rng.uniform(0, max(1, W - bw)); gy = rng.uniform(0, max(1, H - bh))
                cbox = (gx, gy, gx + bw, gy + bh)
            if cbox[2] - cbox[0] < 4 or cbox[3] - cbox[1] < 4:
                continue
            gw, gh = cbox[2] - cbox[0], cbox[3] - cbox[1]
            crop = img.crop(tuple(int(v) for v in cbox))
            rx = rng.uniform(0, max(1, W - gw)); ry = rng.uniform(0, max(1, H - gh))
            rcrop = img.crop((int(rx), int(ry), int(rx + gw), int(ry + gh)))

            q = it["question"]
            rec = {"uid": x["uid"], "group": group, "stratum": x["stratum"],
                   "split": x["split"], "question_id": x["question_id"],
                   "tokens_on_object": x.get("tokens_on_object"),
                   "baseline_p_yes": x["p_yes_fp16"], "arms": {}, "realized_tokens": {}}
            # NOTE the split arms: alloc_query funds the crop with HALF the budget; alloc_query25
            # gives the crop only a quarter, leaving the scene at 75%. If the crossing point moves
            # between them, the effect is the budget DIVISION, not allocation failing.
            cells = {
                "uniform":       ([img], False, None),
                "alloc_query":   ([img, crop], True, 0.5),
                "alloc_query25": ([img, crop], True, 0.25),
                "alloc_random":  ([img, rcrop], True, 0.5),
            }
            for name, (imgs, conn, frac) in cells.items():
                if frac is None:
                    fitted, _ = fit_to_budget(lambda ims: measure(ims, q, conn), imgs, BUDGET)
                else:
                    # calibrate each image separately so the crop gets `frac` of the budget
                    f1, _ = fit_to_budget(lambda ims: measure(ims, q, False),
                                          [imgs[0]], int(BUDGET * (1 - frac)))
                    f2, _ = fit_to_budget(lambda ims: measure(ims, q, False),
                                          [imgs[1]], int(BUDGET * frac))
                    fitted = [f1[0], f2[0]]
                p, realized = run(fitted, q, conn)
                rec["arms"][name] = p
                rec["realized_tokens"][name] = realized
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(targets)}] {n/el:.2f} it/s "
                      f"eta={(len(targets)-n)/(n/el)/60:.1f}min", flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
