"""
Phase 27: IS THERE ANY BUDGET AT WHICH QUERY-INDEPENDENT ALLOCATION CATCHES UP?

THE QUESTION, AND WHY IT IS THE AMBITIOUS ONE
---------------------------------------------
Phase 25 (LLaVA-NeXT, matched realized tokens) contains an unnamed exchange rate:

    uniform      @1464 tok -> 47.6%
    anyres       @2144 tok -> 55.0%      (+47% budget, spent the way the architecture ships: +7.4pp)
    alloc_query  @2144 tok -> 88.5%      (the SAME 2144 tokens, placed by the question: +33.5pp)

So on that one comparison a spatial prior is worth ~4.5x its weight in compute. The ratio is
interesting; **the asymptote is the result**. If query-independent spending catches up at some
feasible budget, this is an efficiency argument -- useful, incremental. If it provably does NOT
inside the architecture's realizable range, the claim becomes a statement about a DESIGN CHOICE:

    **No amount of query-independent budget substitutes for placement in this regime.**

Q-CueGraph matched image AREA and RUTA explicitly declined to rate-match, so neither can ask this.
The matched-budget apparatus built across Phases 17-25 is what makes it askable.

DESIGN -- both arms SINGLE IMAGE, so the sweep is format-clean at every point
-----------------------------------------------------------------------------
    crop_only@B_small   the query region alone, ONE image, held at a FIXED small budget
    uniform@B           the whole image, ONE image, B swept across the widest feasible ladder

Phase 23 was confounded because arms differed in image count; that cannot happen here -- every
point on the sweep is one image through the same processor. `alloc_query` (thumbnail + crop, TWO
images) is recorded alongside for continuity with Phases 20/25 but is flagged and never used for
the crossing point.

**`crop_only` is held at ONE budget for the whole sweep.** If both quantities moved, the crossing
point would be meaningless.

Model: Qwen3-VL-2B, because its budget knob is CONTINUOUS. LLaVA-NeXT realizes only two counts at
V*Bench's aspect ratio (measured, Phase 25) and cannot sweep.

PRE-REGISTERED READINGS
-----------------------
* uniform@B reaches crop_only@B_small at some B* in range
      => report the exchange rate B*/B_small as the headline number, with its CI.
* no crossing up to the largest realizable B
      => "query-independent budget does not substitute for placement in this regime", stated as a
         BOUND: no crossing up to B_max, and B_max is reported with its realized token count. It is
         not a claim about all budgets and must not be written as one.
* crop_only@B_small is beaten early (small B*)
      => placement is a cheap efficiency trick, not a structural limit; the paper's framing softens
         and 4M's emphasis moves to efficiency.

The floor (bug #21: every image costs >=64 merged tokens) bounds the ladder from below; the
realizable ladder is MEASURED before the sweep, not assumed.
"""
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase17_budget_allocation import fit_to_budget
from phase7_vision_zoom import CONNECTOR_TEXT

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase27_exchange_results.jsonl"
B_SMALL = 300                                   # the FIXED query-placed budget
SWEEP = [150, 300, 600, 1200, 2400, 4800, 8000]  # query-independent ladder, filtered to realizable
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None


def main():
    rng = random.Random(107)
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    print(f"V*Bench: {len(ds)} items")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    tok = processor.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]

    def _chat(imgs, text, conn):
        if len(imgs) == 1:
            content = [{"type": "image"}]
        else:
            content = [{"type": "image"}, {"type": "text", "text": CONNECTOR_TEXT},
                       {"type": "image"}] if conn else [{"type": "image"}, {"type": "image"}]
        content.append({"type": "text", "text": text})
        return processor.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    def measure(imgs, text="x", conn=True):
        i = processor(images=imgs, text=_chat(imgs, text, conn), return_tensors="pt")
        return int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    def run(imgs, text, conn):
        i = processor(images=imgs, text=_chat(imgs, text, conn), return_tensors="pt").to(model.device)
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return p.tolist(), int(p.argmax().item()), \
            int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))

    # MEASURE the realizable ladder rather than assuming it (bug #21 was assuming)
    probe = Image.open(os.path.join(root, ds[0]["image"])).convert("RGB")
    ladder = []
    for B in SWEEP:
        f, r = fit_to_budget(lambda ims: measure(ims, "x", False), [probe], B)
        ladder.append((B, r))
    print("  realizable ladder (requested -> realized):", ladder)
    SWEEP_OK = [B for B, r in ladder if abs(r - B) / B < 0.15]
    print(f"  sweeping uniform over {SWEEP_OK};  crop_only FIXED at {B_SMALL}")

    done = set()
    try:
        for l in open(OUT_PATH):
            done.add(json.loads(l)["question_id_full"])
        print(f"Resuming: {len(done)}")
    except FileNotFoundError:
        pass

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for ex in ds:
            rel = ex["image"]
            qid = f"{rel}::{ex['question_id']}"
            ip = os.path.join(root, rel); ap = os.path.splitext(ip)[0] + ".json"
            if not (os.path.exists(ip) and os.path.exists(ap)):
                continue
            ann = json.load(open(ap))
            if not ann.get("bbox") or qid in done:
                continue
            img = Image.open(ip).convert("RGB"); W, H = img.size
            x0 = min(b[0] for b in ann["bbox"]); y0 = min(b[1] for b in ann["bbox"])
            x1 = max(b[0]+b[2] for b in ann["bbox"]); y1 = max(b[1]+b[3] for b in ann["bbox"])
            pw, ph = (x1-x0)*PAD, (y1-y0)*PAD
            cb = (max(0, x0-pw), max(0, y0-ph), min(W, x1+pw), min(H, y1+ph))
            if cb[2]-cb[0] < 8 or cb[3]-cb[1] < 8:
                continue
            crop = img.crop(tuple(int(v) for v in cb))
            gw, gh = cb[2]-cb[0], cb[3]-cb[1]
            rx = rng.uniform(0, max(1, W-gw)); ry = rng.uniform(0, max(1, H-gh))
            rcrop = img.crop((int(rx), int(ry), int(rx+gw), int(ry+gh)))

            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            text = ex["text"]
            rec = {"question_id_full": qid, "image": rel, "category": ex["category"],
                   "label": label, "img_wh": [W, H],
                   "bbox_area_frac": ((x1-x0)*(y1-y0))/(W*H),
                   "pred": {}, "realized_tokens": {}}
            # FIXED query-placed reference arms
            for nm, src, conn in [("crop_only", [crop], False),
                                  ("crop_random", [rcrop], False),
                                  ("alloc_query_2img", [img, crop], True)]:
                f, _ = fit_to_budget(lambda ims: measure(ims, text, conn), src, B_SMALL)
                _, pr, rz = run(f, text, conn)
                rec["pred"][f"{nm}@{B_SMALL}"] = pr
                rec["realized_tokens"][f"{nm}@{B_SMALL}"] = rz
            # the sweep: query-INDEPENDENT spending, single image, increasing budget
            for B in SWEEP_OK:
                f, _ = fit_to_budget(lambda ims: measure(ims, text, False), [img], B)
                _, pr, rz = run(f, text, False)
                rec["pred"][f"uniform@{B}"] = pr
                rec["realized_tokens"][f"uniform@{B}"] = rz
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 10 == 0:
                el = time.time()-t0
                print(f"[{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
