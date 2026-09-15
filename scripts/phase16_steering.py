"""
Phase 16: activation steering at L16 -- an INTERNAL intervention (no image preprocessing).

WHY L16, AND WHY EXPECT A MODEST EFFECT
---------------------------------------
Phase 15 established that the object's own visual tokens carry CATEGORY-SPECIFIC information, but
weakly: AUROC 0.709 [0.637, 0.778] at L16 on a same-image, size-matched, other-category control
(vs 0.88 when objectness was left in). L16 is where that signal peaks, so that is where the
direction is derived and injected. The target is real but FAINT -- Phase 6e's null is the realistic
prior here, not an anomaly.

THE DIRECTION
-------------
v = mean(hidden @ L16 over the QUERIED category's visual tokens)
  - mean(hidden @ L16 over a size-matched OTHER category's visual tokens)
averaged over DERIVATION items, then unit-normalized. This is exactly the contrast Phase 15 showed
is decodable, so it is the direction with evidence behind it -- not a generic "truthfulness" vector.
Derivation items are disjoint from evaluation items (uid-level), or the result is circular.

Injection: hook `layers[15]` (whose output is `hidden_states[16]`) and add
    alpha * ||h_mean_at_those_positions|| * v_hat
so alpha is scale-free relative to the local activation norm.

ARMS (all sweep alpha)
----------------------
  steer_obj        v at the object's own visual tokens        -- ORACLE location (like Phase 7's
                                                                 oracle zoom; label it as such)
  rand_obj         NORM-MATCHED RANDOM direction, same positions   <-- THE control that decides
  steer_all_img    v at ALL image tokens                      -- needs no localizer
  steer_last       v at the final position                    -- the standard CAA-style injection
  rand_last        norm-matched random direction, final position    <-- control

PRE-REGISTERED VERDICT RULES (fixed before any number is seen; Phase 6e is the precedent where
object-target and random-target interventions were indistinguishable and the honest conclusion was
a statement about the tool, not the mechanism):
  * SUCCESS requires steer_obj to beat rand_obj on DISCRIMINATION (recovery on confident denials
    MINUS false-positive rate on genuine absences), with a CI excluding zero.
  * If steer_obj ~= rand_obj at every alpha -> NULL. Report as such; do not tune alpha until they
    separate.
  * If both raise P(yes) on denials AND on true negatives -> it is a BIAS SHIFT, not perception.
    The true-negative arm is what distinguishes these, and it is mandatory (Sec 5.1's mathematical
    counterfactual and bugs #7/#10 are all instances of this trap).

Cohort: RePOPE-clean (Phase 11 -- the raw confident-denial cohort is 58.1% mislabeled/ambiguous),
with P(yes) re-identified in fp16 by Phase 12 (bug #6: cached values are 4-bit).
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
from phase15_category_specificity import build_full_lookup

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
OUT_PATH = f"{DATA}/phase16_steering_results.jsonl"
LAYER = 16                      # hidden_states[16] == output of layers[15]
ALPHAS = [0.25, 0.5, 1.0, 2.0]
ANSWER_SUFFIX = " Please answer this question with yes or no."
N_NEG_EVAL = 220


def main():
    rng = random.Random(71)

    clean = [json.loads(l) for l in open(f"{DATA}/phase12_clean_pyes.jsonl")]
    denials = [r for r in clean if r["label"] == "yes" and r["p_yes_fp16"] < 0.01]
    negs = [r for r in clean if r["label"] == "no"]
    negs = rng.sample(negs, min(N_NEG_EVAL, len(negs)))
    print(f"RePOPE-clean fp16 confident denials: {len(denials)}")
    print(f"clean negatives for the FP arm:      {len(negs)}")
    eval_uids = {r["uid"] for r in denials} | {r["uid"] for r in negs}

    # ---- derive v from Phase 15 features, on items DISJOINT from the eval set ----
    d = np.load(f"{DATA}/phase15_catspec_feats.npz")
    F = d["feats"].astype(np.float32)
    layers = d["layers"].tolist()
    meta15 = [json.loads(l) for l in open(f"{DATA}/phase15_catspec_meta.jsonl")]
    D = F.shape[1] // (len(layers) * 2)
    li = layers.index(LAYER)
    keep = [i for i, m in enumerate(meta15) if m["uid"] not in eval_uids]
    q = F[keep, (li * 2 + 0) * D:(li * 2 + 0) * D + D]
    o = F[keep, (li * 2 + 1) * D:(li * 2 + 1) * D + D]
    v = (q.mean(0) - o.mean(0))
    v = v / (np.linalg.norm(v) + 1e-8)
    print(f"direction derived at L{LAYER} from {len(keep)} disjoint items "
          f"(dropped {len(meta15)-len(keep)} overlapping)")

    lut = build_full_lookup()
    want = eval_uids
    print("Loading items...")
    items = {it["uid"]: it for it in build_items() if it["uid"] in want}
    print(f"  resolved {len(items)}/{len(want)}")

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
    dev = model.device
    v_t = torch.tensor(v, dtype=torch.float16, device=dev)
    rand_dir = torch.randn(len(v), generator=torch.Generator().manual_seed(5)).numpy()
    rand_dir = rand_dir / np.linalg.norm(rand_dir)
    r_t = torch.tensor(rand_dir, dtype=torch.float16, device=dev)

    layer_mod = model.model.language_model.layers[LAYER - 1]
    state = {"pos": None, "vec": None, "alpha": 0.0}

    def hook(module, args, output):
        if state["vec"] is None or not state["pos"]:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        idx = torch.tensor(state["pos"], device=hs.device)
        norm = hs[0, idx].norm(dim=-1).mean()
        hs[0, idx] = hs[0, idx] + state["alpha"] * norm * state["vec"]
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    layer_mod.register_forward_hook(hook)

    def p_yes(inputs):
        with torch.no_grad():
            lg = model(**inputs).logits[0, -1].float()
        return torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                          torch.logsumexp(lg[no_ids], 0)]), 0)[0].item()

    targets = [(r, "denial") for r in denials] + [(r, "negative") for r in negs]
    done = set()
    try:
        for line in open(OUT_PATH):
            done.add(json.loads(line)["uid"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass
    targets = [t for t in targets if t[0]["uid"] not in done]

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for r, group in targets:
            it = items.get(r["uid"])
            if it is None:
                continue
            key = (r["split"], str(r["question_id"]))
            ent = lut.get(key)
            img = it["image"].convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": it["question"] + ANSWER_SUFFIX}]}]
            chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=img, text=chat, return_tensors="pt").to(dev)
            gthw = inputs["image_grid_thw"][0].tolist()
            gh, gw = gthw[1] // MERGE, gthw[2] // MERGE
            rz_h, rz_w = gthw[1] * PATCH, gthw[2] * PATCH
            img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
            if len(img_pos) < gh * gw:
                continue
            base = img_pos[0].item()
            all_img = [base + i for i in range(gh * gw)]
            last = [inputs["input_ids"].shape[1] - 1]

            if group == "denial" and ent is not None:
                (W, H), by_cat, qcat = ent
                bxs = by_cat.get(qcat) if qcat is not None else None
                if not bxs:
                    continue
                s = object_token_indices(bxs, W, H, rz_w, rz_h, gh, gw)
                if not s:
                    x0 = min(b[0] for b in bxs); y0 = min(b[1] for b in bxs)
                    x1 = max(b[0]+b[2] for b in bxs); y1 = max(b[1]+b[3] for b in bxs)
                    cx = (x0+x1)/2/W*gw; cy = (y0+y1)/2/H*gh
                    s = {min(max(int(cy),0),gh-1)*gw + min(max(int(cx),0),gw-1)}
                obj = [base + i for i in sorted(s)]
            else:
                # negatives have no object: use a random contiguous token block of similar size
                k = max(1, int(0.01 * gh * gw))
                st = rng.randrange(max(1, gh * gw - k))
                obj = [base + i for i in range(st, st + k)]

            state.update(vec=None, pos=None, alpha=0.0)
            rec = {"uid": r["uid"], "group": group, "split": r["split"],
                   "question_id": r["question_id"], "category": r.get("category"),
                   "pixel_area_frac": r.get("pixel_area_frac"),
                   "n_obj_tokens": len(obj), "baseline": p_yes(inputs), "arms": {}}
            for a in ALPHAS:
                for name, vec, pos in [("steer_obj", v_t, obj), ("rand_obj", r_t, obj),
                                       ("steer_all_img", v_t, all_img),
                                       ("steer_last", v_t, last), ("rand_last", r_t, last)]:
                    state.update(vec=vec, pos=pos, alpha=a)
                    rec["arms"][f"{name}@{a}"] = p_yes(inputs)
            state.update(vec=None, pos=None, alpha=0.0)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"[{n}/{len(targets)}] {n/el:.2f} it/s "
                      f"eta={(len(targets)-n)/(n/el)/60:.1f}min", flush=True)

    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
