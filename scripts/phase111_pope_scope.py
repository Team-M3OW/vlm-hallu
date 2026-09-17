"""
Phase 111: POPE as a SCOPE TEST, not a competitiveness test -- and the first test of what the method
does when THERE IS NOTHING TO FIND.

WHY POPE CANNOT BE A BENCHMARK WIN. Joining POPE positives to COCO boxes, the median object occupies
**28 merged tokens** at B=300 and only **6.5%** fall below the 0.25-token encoding cliff. On 93% of
POPE the evidence is already encoded, so a method that re-spends the budget has nothing to fix. (The
"0.4 merged tokens" figure quoted throughout this project is POPE's CONFIDENT-DENIAL cohort, a
selected subset, not POPE as a whole.)

PRE-REGISTERED, written before the run
    1. NO overall gain on POPE. A large overall gain would REFUTE the cliff mechanism, not support
       the method.
    2. The gain, if any, is concentrated in the below-cliff stratum (tokens_on_target < 0.25).
    3. THE REAL TEST -- negatives. On "is there a snowboard?" when there is none, there is no target
       to crop to. The head points somewhere regardless. Cropping into a salient region may push the
       model toward a false "yes". FALSE-POSITIVE RATE IS A PRIMARY OUTCOME, not a diagnostic
       (the lesson of phase 16). If the method costs more on negatives than it gains on positives,
       that is a real limitation of every crop-based method and nobody in this literature reports it.

DESIGN
    RePOPE-clean labels (phase 11: the corrected pool; confident model disagreement is a 7.2x
    enriched label-error detector, and every cohort since is RePOPE-clean).
    Sampling is deliberately STRATIFIED to over-represent the below-cliff stratum, which makes the
    overall accuracy NOT comparable to published POPE numbers. Reported per stratum only.
    Head fitted OUT-OF-FOLD on positives (boxes exist there); applied to every item.
    Arms at matched budget: uniform@300 | uniform@600 | argmax@0.25 | head@0.25 | rand@0.25 |
    oracle@0.25 (positives only -- undefined without a box).
    Answer read from P(yes) vs P(no) logits, no generation.
"""
import json, os, random, sys, time
from collections import defaultdict
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ["HF_DATASETS_OFFLINE"] = "1"
from transformers import AutoProcessor, AutoModelForImageTextToText

WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
MODEL_ID = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct", "qwen2": "Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
OUT = f"{D}/data/phase111_pope_{WHICH}.jsonl"
# COCO images are NOT on disk; the cached lmms-lab/POPE dataset ships them, so nothing downloads.
B0, W, CLIFF = 300, 0.25, 0.25
N_LARGE, N_NEG = 200, 200   # below-cliff and 0.25-1 strata are taken IN FULL (only 90 and 198 exist)
Image.MAX_IMAGE_PIXELS = None


def load_pope_with_boxes():
    """RePOPE-clean labels, HF-cached POPE images, COCO boxes for the queried category."""
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", "default", split="test")
    imgcol = {}
    # RePOPE (arXiv 2504.15707) ships the CORRECTED labels in coco_repope_*.json; the
    # coco_pope_*.json beside them is the original, vendored for comparison. Phase 11's semantics:
    # a null RePOPE label means the item was pruned as AMBIGUOUS and must be dropped, not kept.
    relabel, ambiguous = {}, set()
    for split in ["adversarial", "popular", "random"]:
        orig = f"{D}/data/repope/coco_pope_{split}.json"
        corr = f"{D}/data/repope/coco_repope_{split}.json"
        if not (os.path.exists(orig) and os.path.exists(corr)):
            continue
        rmap = {str(json.loads(l)["question_id"]): json.loads(l)["label"] for l in open(corr)}
        for line in open(orig):
            q = json.loads(line)
            key = (q["image"].rsplit(".", 1)[0], q["text"].strip().lower())
            lab = rmap.get(str(q["question_id"]))
            if lab is None:
                ambiguous.add(key)
            else:
                relabel[key] = str(lab).lower().startswith("y")
    n_fix, n_amb = 0, 0
    ann = json.load(open(f"{D}/data/coco_ann/instances_val2014.json"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {im["id"]: im for im in ann["images"]}
    byimg = defaultdict(list)
    for a in ann["annotations"]:
        byimg[a["image_id"]].append(a)
    rows = []
    for k, r in enumerate(ds):
        fn = r["image_source"]
        iid = int(fn.split("_")[-1])
        if iid not in imgs: continue
        im = imgs[iid]; Wd, Ht = im["width"], im["height"]
        q = r["question"]
        cat = q.lower().replace("is there a ", "").replace("is there an ", "").replace(" in the image?", "").strip()
        inst = [a for a in byimg[iid] if cats.get(a["category_id"], "").lower() == cat]
        key = (fn, q.strip().lower())
        lab_hf = str(r["answer"]).lower().startswith("y")
        if key in ambiguous:
            n_amb += 1
            continue                                    # RePOPE pruned it as ambiguous
        lab_rp = relabel.get(key, lab_hf)
        if lab_rp != lab_hf: n_fix += 1
        lab = 1 if lab_rp else 0
        if lab == 1 and not inst: continue              # label says present but no box: skip
        if True:
            if lab == 1:
                xs = [a["bbox"] for a in inst]
                x0 = min(b[0] for b in xs) / Wd; y0 = min(b[1] for b in xs) / Ht
                x1 = max(b[0]+b[2] for b in xs) / Wd; y1 = max(b[1]+b[3] for b in xs) / Ht
                box = [x0, y0, x1, y1]
                tot = (x1-x0)*(y1-y0) * B0

            else:
                box, tot = None, None
            rows.append({"qid": f"{r['category']}/{r['question_id']}", "idx": k, "text": q,
                         "label": lab, "gt_box_frac": box, "tokens_on_target": tot,
                         "n_inst": len(inst)})
    print(f"RePOPE: relabelled {n_fix}, dropped {n_amb} as ambiguous", flush=True)
    return rows, ds


def main():
    rows, ds = load_pope_with_boxes()
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    small = [r for r in pos if r["tokens_on_target"] < CLIFF]
    mid   = [r for r in pos if CLIFF <= r["tokens_on_target"] < 1.0]
    large = [r for r in pos if r["tokens_on_target"] >= 1.0]
    rng = random.Random(111)
    sel = small + mid + rng.sample(large, min(N_LARGE, len(large))) + \
          rng.sample(neg, min(N_NEG, len(neg)))
    for r in sel:
        r["stratum"] = ("neg" if r["label"] == 0 else
                        "below_cliff" if r["tokens_on_target"] < CLIFF else
                        "cliff_zone" if r["tokens_on_target"] < 1.0 else "above")
    rng.shuffle(sel)
    print(f"POPE joined: {len(rows)} rows  ({len(pos)} pos / {len(neg)} neg); "
          f"below cliff {len(small)}  cliff zone {len(mid)}  above {len(large)}", flush=True)
    print(f"sampled {len(sel)}: below_cliff {len(small)} (ALL), cliff_zone {len(mid)} (ALL), "
          f"above {min(N_LARGE,len(large))}, neg {min(N_NEG,len(neg))}   "
          f"[STRATIFIED -- overall accuracy is NOT comparable to published POPE numbers]", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    yes_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in ["Yes", " Yes", "yes", " yes"]})
    no_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][-1] for s in ["No", " No", "no", " no"]})
    img_tok_id = model.config.image_token_id
    NL = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") else 28
    b0, b1 = int(.57*NL), int(.93*NL)+1

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat + "", return_tensors="pt")

    def measure(img):
        return int(sum(g[1]*g[2]//4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.06):
        Wd, Ht = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(Wd*sc)), max(28, int(Ht*sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r) ** 0.5
        return best

    def window(img, cx, cy, w):
        iw, ih = img.size
        x0, y0 = (cx-w/2)*iw, (cy-w/2)*ih
        x0 = min(max(0, x0), iw-w*iw); y0 = min(max(0, y0), ih-w*ih)
        return img.crop((int(x0), int(y0), int(x0+w*iw), int(y0+w*ih)))

    def answer(img, text):
        inp = build(img, text)
        rz = int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist()))
        inp = inp.to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[yes_ids], 0),
                                       torch.logsumexp(lg[no_ids], 0)]), 0)
        return float(p[0]), rz

    def localize(img, text):
        small_img, _ = fit(img, B0)
        inp = build(small_img, text)
        g = inp["image_grid_thw"][0].tolist(); gh, gw = g[1]//2, g[2]//2
        n_img = gh*gw
        pos_ = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
        base = int(pos_[0].item())
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        acc = torch.zeros(n_img, device=model.device)
        for L in range(b0, min(b1, len(out.attentions))):
            a = out.attentions[L][0, :, -1, base:base+n_img].float().mean(0)
            acc += a / a.sum().clamp_min(1e-12)
        del out
        acc = (acc/(b1-b0)).reshape(gh, gw)
        m = torch.full_like(acc, -1.0)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = acc[1:-1, 1:-1]
        else: m = acc.clone()
        i = int(m.argmax().item())
        return ((i % gw)+.5)/gw, ((i//gw)+.5)/gh, gh, gw

    t0, n = time.time(), 0
    with open(OUT, "w") as fout:
        for r in sel:
            img = ds[r["idx"]]["image"].convert("RGB")
            cx, cy, gh, gw = localize(img, r["text"])
            rr = random.Random(hash(r["qid"]) % 99999)
            rcx, rcy = rr.uniform(.1, .9), rr.uniform(.1, .9)
            arms = {"uniform@300": fit(img, B0)[0], "uniform@600": fit(img, 2*B0)[0],
                    "argmax@0.25": fit(window(img, cx, cy, W), B0)[0],
                    "rand@0.25":   fit(window(img, rcx, rcy, W), B0)[0]}
            if r["gt_box_frac"]:
                g = r["gt_box_frac"]
                arms["oracle@0.25"] = fit(window(img, (g[0]+g[2])/2, (g[1]+g[3])/2, W), B0)[0]
            rec = {**{k: r[k] for k in ["qid", "label", "tokens_on_target", "gt_box_frac", "n_inst", "stratum"]},
                   "grid": [gh, gw], "argmax_cell": [cx, cy], "p_yes": {}, "realized_tokens": {}}
            for nm, im in arms.items():
                p, rz = answer(im, r["text"])
                rec["p_yes"][nm] = round(p, 6); rec["realized_tokens"][nm] = rz
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 25 == 0:
                el = time.time()-t0
                print(f"  [{n}/{len(sel)}] {n/el:.2f} it/s eta={(len(sel)-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
            torch.cuda.empty_cache()
    print(f"Done. {n} rows -> {OUT}", flush=True)


main()
