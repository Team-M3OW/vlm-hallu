"""
Phase 28 (RQ-C): IS THE EXCHANGE RATE A PROPERTY OF THE TASK, OR OF ONE MODEL?

THE QUESTION
------------
Phase 27 measured, on Qwen3-VL-2B: a 300-token query-PLACED crop reaches 93.7% on V*Bench, and
query-INDEPENDENT spending is still 8.9pp behind at 7957 tokens -- **no crossing at 26.5x**.
Phase 25 measured a compatible number on LLaVA-NeXT at matched budget (+33.5pp over native AnyRes).

Two models is an observation. The claim worth making needs the rate to hold across tokenizers that
**disagree about everything**:

| model | patch | how budget is spent | token floor |
|---|---|---|---|
| Qwen3-VL-2B | 16px, merge 2 | continuous dynamic resolution | 64 / image (measured) |
| Qwen2-VL-7B | 14px, merge 2 | continuous dynamic resolution | measured at runtime |
| LLaVA-NeXT-7B | 336px tiles | AnyRes, 5 fixed pinpoints | 1416 total (measured) |
| llava-onevision-7B | 384px tiles | AnyRes, many pinpoints | measured at runtime |

* Rate roughly CONSTANT across them => it is a property of the **task** (find a ~0.001-area target
  in a crowded scene), not of any architecture. That is the strongest form this paper can take.
* Rate VARIES WILDLY => the honest result is that it is architecture-specific, and the paper's
  claim narrows from "a design flaw in VLMs" to "a survey of four systems". Report that.

DESIGN -- Phase 27 replicated per model, nothing new invented
------------------------------------------------------------
    crop_only@B0    the query region alone, ONE image, FIXED at that model's smallest usable budget
    uniform@B       the whole image, ONE image, B swept over a ladder MEASURED per model

Both arms are single-image on every model, so the image-count confound that voided Phase 23 cannot
recur. Budgets are never assumed: each model's realizable ladder is probed at startup, because the
per-image floor differs per architecture (bug #21 was assuming Qwen's arithmetic).

**Comparability across models** is by RATIO, not absolute tokens: each model's ladder is expressed
as multiples of its own B0, and the reported quantity is "the multiple at which uniform catches
crop_only@B0, or a bound if it never does". Absolute token counts are not comparable across
tokenizers and are reported only for reference.

PRE-REGISTERED
--------------
* All models show no crossing within their measured range => report the minimum bound across models.
* Some cross, some do not => report the spread and DO NOT average it; name which architectures
  differ and look for what distinguishes them (patch size? tiling vs dynamic? model scale?).
* Note the scale confound up front: Qwen3-VL here is **2B** while the others are **7B**. If the 2B
  model shows a larger gap, model capacity is an alternative explanation and must be flagged, not
  explained away.
"""
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoProcessor, AutoConfig

OUT_PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase28_arch_results.jsonl"
PAD = 0.25
Image.MAX_IMAGE_PIXELS = None

# requested ladder as MULTIPLES of each model's B0; realizability is measured per model
LADDER_MULT = [0.5, 1, 2, 4, 8, 16, 26]


def load_model(key):
    """Return (model, processor, prompt_fn, img_token_id, kind). Kept explicit per family rather
    than guessed, because the prompt templates and token bookkeeping genuinely differ."""
    if key == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration
        mid = "Qwen/Qwen2-VL-7B-Instruct"
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float16, device_map={"": 0})
        pr = AutoProcessor.from_pretrained(mid)
        return m, pr, "qwen", mid
    if key == "onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        mid = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        m = LlavaOnevisionForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float16, device_map={"": 0})
        pr = AutoProcessor.from_pretrained(mid)
        return m, pr, "llava", mid
    if key == "llavanext":
        from transformers import LlavaNextForConditionalGeneration
        mid = "llava-hf/llava-v1.6-vicuna-7b-hf"
        m = LlavaNextForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float16, device_map={"": 0})
        pr = AutoProcessor.from_pretrained(mid)
        return m, pr, "llava", mid
    raise ValueError(key)


def main():
    key = sys.argv[1]
    rng = random.Random(107)
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    print(f"=== Phase 28: {key} ===", flush=True)
    model, pr, fam, mid = load_model(key)
    model.eval()
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [L, f" {L}"]}) for L in "ABCD"]
    cfg = AutoConfig.from_pretrained(mid)
    img_tok_id = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)

    def build(img, text):
        if fam == "qwen":
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            chat = (f"USER: <image>\n{text}\n"
                    "Answer with the option's letter from the given choices directly. ASSISTANT:")
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img, text="x"):
        o = build(img, text)
        if "image_grid_thw" in o:
            return int(sum(g[1] * g[2] // 4 for g in o["image_grid_thw"].tolist()))
        return int((o["input_ids"][0] == img_tok_id).sum())

    def run(img, text):
        i = build(img, text).to(model.device)
        rz = (int(sum(g[1] * g[2] // 4 for g in i["image_grid_thw"].tolist()))
              if "image_grid_thw" in i else int((i["input_ids"][0] == img_tok_id).sum()))
        with torch.no_grad():
            lg = model(**i).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[ids], 0) for ids in opt_ids]), 0)
        return int(p.argmax().item()), rz

    SCALES = (0.04, 0.07, 0.1, 0.15, 0.22, 0.32, 0.45, 0.65, 0.9, 1.3, 1.9, 2.8, 4.0)
    _rung_cache = {}
    MAX_MP = 24_000_000   # host has 30 GB RAM; the fast image processor peaked at 26 GB on 54 MP

    def _rz(img, sc):
        """Resize, or None if the result would exceed the megapixel guard."""
        W, H = img.size
        w, h = max(32, int(W * sc)), max(32, int(H * sc))
        if w * h > MAX_MP:
            return None
        return img.resize((w, h), Image.BICUBIC)


    def rungs(img, max_target):
        """Realized token count at each ladder scale, computed ONCE per image, and STOPPED EARLY.

        Two costs were hiding here. (1) The ladder is identical for every budget target, so the
        naive version re-measured all 13 scales for each of ~9 arms -- ~117 processor calls per
        item. Caching by image identity fixes that. (2) The top scales are ruinous on a full-size
        source: 2246x1582 at scale 4.0 is **57 megapixels**, and no budget in the sweep ever needs
        it. Once a scale realizes comfortably more than the largest target we will ask for, every
        larger scale is dead weight, so we stop. Small crops still climb the whole ladder, which is
        exactly where the large scales ARE needed.
        """
        k = id(img)
        if k not in _rung_cache:
            W, H = img.size
            out = []
            prev = None
            for sc in SCALES:
                cur = _rz(img, sc)
                if cur is None:          # megapixel guard tripped: no larger scale is affordable
                    break
                r = measure(cur)
                out.append((sc, r))
                if r > 1.3 * max_target:
                    break
                # CEILING STOP: a tiled model saturates -- LLaVA-NeXT returns 2144 at 562x375 AND
                # at 4500x3000. Without this, an unreachable target drives the ladder to scale 4.0
                # on a 2250x1500 source (54 MP) and the host OOM-kills the run (observed: 26 GB
                # RSS, pid 381958). Two consecutive identical counts means the ladder has topped
                # out and every larger scale is wasted memory, not more tokens.
                if prev is not None and r == prev:
                    break
                prev = r
            _rung_cache[k] = out
        return _rung_cache[k]

    def fit(img, target, max_target=None, refine=4, tol=0.04):
        """HYBRID: coarse ladder, then proportional refinement.

        The ladder alone is too coarse for a CONTINUOUS-budget model: on Qwen2-VL a requested 300
        landed on a rung realizing **391** (30% off), which the budget gate would void. But a purely
        proportional solver cannot cross the token-count plateaus of a TILED model (measured on
        LLaVA-NeXT, Phase 25). So: use the ladder to land on the right plateau, then refine
        proportionally from there. On a step-function model the refinement finds nothing better and
        the ladder's pick stands; on a continuous model it converges. Best-seen is always kept.
        """
        W, H = img.size
        sc, r = min(rungs(img, max_target or target), key=lambda t: abs(t[1] - target))
        best = (sc, r)
        for _ in range(refine):
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc = sc * (target / r) ** 0.5
            cur = _rz(img, sc)
            if cur is None:
                break
            r = measure(cur)
            if abs(r - target) < abs(best[1] - target):
                best = (sc, r)
        sc, r = best
        return _rz(img, sc) or img, r

    # measure this model's realizable ladder before choosing anything
    probe = Image.open(os.path.join(root, ds[0]["image"])).convert("RGB")
    tiny = probe.resize((64, 48))
    floor = measure(tiny)

    # CEILING, measured the same way the floor is -- and for the same reason (bug #21: I once
    # restarted a run instead of measuring the constraint). A tiled model cannot be pushed past
    # what its `image_grid_pinpoints` set can express, no matter how large the input:
    #
    #   model            mechanism      floor  ceiling   dynamic range
    #   Qwen2-VL-7B      continuous         4     7776         1944x
    #   Qwen3-VL-2B      continuous        64     7957          124x
    #   llava-onevision  tiles 384px     1261     7329          5.8x
    #   LLaVA-NeXT       tiles 336px     1416     2144          1.5x
    #
    # Verified structural, not a ladder artifact (the trap in the phantom-cap correction): LLaVA-
    # NeXT returns exactly 2144 at 562x375, 1125x750, 2250x1500 AND 4500x3000 -- four inputs
    # spanning 64x in area. Asking for 26*B0 = 35568 tokens there is not a hard sweep, it is an
    # impossible one, and it is what OOM-killed the first attempt.
    ceil_img = _rz(probe, 2.0) or probe
    ceiling = measure(ceil_img)
    print(f"  per-image token floor (64x48 px input): {floor}", flush=True)
    print(f"  realizable ceiling (2x source): {ceiling}  -> dynamic range {ceiling/max(floor,1):.1f}x",
          flush=True)
    B0 = max(floor, 300)
    ladder = []
    for mlt in LADDER_MULT:
        t = int(B0 * mlt)
        if t < floor:
            continue
        if t > 1.1 * ceiling:
            print(f"  requested {t} exceeds realizable ceiling {ceiling} -- UNREACHABLE, skipped",
                  flush=True)
            continue
        _, r = fit(probe, t, max_target=min(int(B0*max(LADDER_MULT)), ceiling))
        ladder.append((t, r))
    print(f"  B0={B0}   realizable ladder (requested -> realized): {ladder}", flush=True)
    # Sweep the REQUESTED budgets -- refinement hits them precisely on a continuous model. Drop any
    # target the architecture cannot realize: Qwen2-VL caps at 6510 tokens (measured: the 4800 and
    # 7800 requests both land on 6510), and asking past the cap would silently compare two arms at
    # the same ceiling while labelling them different budgets.
    cap = max(r for _, r in ladder)
    SW = sorted({t for t, _ in ladder if t <= 1.05 * cap})
    print(f"  model token cap ~{cap}; sweeping requested budgets {SW}", flush=True)

    done = set()
    try:
        for l in open(OUT_PATH):
            j = json.loads(l)
            if j["model"] == key:
                done.add(j["question_id_full"])
        print(f"  resuming: {len(done)}", flush=True)
    except FileNotFoundError:
        pass

    t0, n = time.time(), 0
    with open(OUT_PATH, "a") as fout:
        for ex in ds:
            rel = ex["image"]; qid = f"{rel}::{ex['question_id']}"
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

            _rung_cache.clear()
            rec = {"model": key, "question_id_full": qid, "category": ex["category"],
                   "label": label, "B0": B0, "img_wh": [W, H],
                   "bbox_area_frac": ((x1-x0)*(y1-y0))/(W*H),
                   "pred": {}, "realized_tokens": {}}
            for nm, src in [("crop_only", crop), ("crop_random", rcrop)]:
                f, _ = fit(src, B0, max_target=B0)
                p, rz = run(f, text)
                rec["pred"][f"{nm}@B0"] = p; rec["realized_tokens"][f"{nm}@B0"] = rz
            for B in SW:
                f, _ = fit(img, B, max_target=max(SW))
                p, rz = run(f, text)
                rec["pred"][f"uniform@{B}"] = p; rec["realized_tokens"][f"uniform@{B}"] = rz
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time()-t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/(n/el)/60:.1f}min", flush=True)
    print(f"Done {key}.", flush=True)


if __name__ == "__main__":
    main()
