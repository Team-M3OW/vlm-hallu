"""
Phase 113 (variant of 106): does the training-free MAX rule CONVERT, or only propose better?

Phase 105 found that CLAA's aggregation rule -- max over a 4-layer window instead of the deployed
mean over a block -- is worth +8.4 / +12.0pp of evidence coverage on two models, training-free, and
that it takes most of the learned head's margin (head - max_win4 is 1 of 2). Every end-task number
in this project uses the HEAD's proposals. This runs the max rule through the answer.

PRE-REGISTERED, before the run:
    PRIMARY   maxwin4@0.25 - uniform@600    (the equal-compute bar, tokens MEASURED)
    SECONDARY maxwin4@0.25 - head@0.25      (does the simple rule match the learned head end-task?)
    CONTROL   head@0.25 is RE-RUN here, not copied, so a mismatch against the stored value exposes
              any pipeline drift before it is mistaken for a result.
    Uniform arms are joined from the stored sweeps (phase78 / phase97m), which used the identical
    fit() and prompt; the head re-run is what validates that join.

    DECISION  maxwin4 ~= head end-task  -> the method becomes ONE LINE and no training at all;
                                           the learned head moves to an ablation.
              maxwin4 << head            -> coverage gains do not convert, and the head is earning
                                           its keep somewhere coverage does not measure.

Window index is chosen OUT-OF-FOLD by item, exactly as in phase 105.
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/phase30c_attn_maps_all.jsonl",
                 f"{D}/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl",
                 f"{D}/phase80a_qwen2vl_proposals.json")}
MODEL_ID, ATTN, PROP = CFG[WHICH]
OUT = f"{D}/phase113_normwmax_{WHICH}.jsonl"
B0, W, NW, K, COV_HIT = 300, 0.25, 4, 5, 0.5
Image.MAX_IMAGE_PIXELS = None


def coverage(cx, cy, gt, w=W):
    x0, x1, y0, y1 = cx - w/2, cx + w/2, cy - w/2, cy + w/2
    if x0 < 0: x0, x1 = 0.0, w
    if y0 < 0: y0, y1 = 0.0, w
    if x1 > 1: x0, x1 = 1-w, 1.0
    if y1 > 1: y0, y1 = 1-w, 1.0
    gx0, gy0, gx1, gy1 = gt
    i = max(0., min(gx1,x1)-max(gx0,x0)) * max(0., min(gy1,y1)-max(gy0,y0))
    return i / max((gx1-gx0)*(gy1-gy0), 1e-12)


def maxwin4_proposals():
    """NORM-WEIGHTED x MAX: the best training-free locator (phase 104b, 56.5% coverage on Qwen3-VL,
    +10.5pp over the deployed block-mean argmax with no training and NO out-of-fold selection).
    Fixed block, fixed rule -- nothing is tuned here, so there is no fold machinery to get wrong."""
    props = json.load(open(PROP))
    src = f"{D}/phase104b_locators_{WHICH}.jsonl"
    rows = [json.loads(l) for l in open(src)]
    rows = [r for r in rows if r["question_id_full"] in props]
    N = len(rows); items = []
    for r in rows:
        gh, gw = r["grid"]; n = r["n_img"]
        A = np.asarray(r["nw_layers"], float)          # already per-layer sum-normalised
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
        c = np.array([coverage(float(fx[i]), float(fy[i]), r["gt_box_frac"]) for i in range(n)])
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        items.append((A, c, rm.ravel(), fx, fy))
    b0, b1 = (16, 27) if WHICH == "qwen3" else (15, 27)
    out = {}
    for i, (A, c, rm, fx, fy) in enumerate(items):
        j = int(np.argmax(np.where(rm, A[b0:b1].max(0), -1e9)))
        out[rows[i]["question_id_full"]] = (float(fx[j]), float(fy[j]), float(c[j]), b0)
    return out, props


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    mw, props = maxwin4_proposals()
    print(f"normw-max proposals: {len(mw)} items, coverage "
          f"{100*np.mean([v[2] >= COV_HIT for v in mw.values()]):.1f}%", flush=True)
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1]*g[2]//4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.06):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r) ** 0.5
        return best

    def window(img, cx, cy, w):
        iw, ih = img.size
        x0, y0 = (cx - w/2)*iw, (cy - w/2)*ih
        x0 = min(max(0, x0), iw - w*iw); y0 = min(max(0, y0), ih - w*ih)
        return img.crop((int(x0), int(y0), int(x0 + w*iw), int(y0 + w*ih)))

    def answer(img, text):
        inp = build(img, text)
        rz = int(sum(g[1]*g[2]//4 for g in inp["image_grid_thw"].tolist()))
        inp = inp.to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt_ids]), 0)
        return [round(float(v), 6) for v in p.tolist()], rz

    t0, n = time.time(), 0
    with open(OUT, "w") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}" if "question_id" in ex else ex.get("question_id_full")
            if qid not in mw or qid not in props: continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            label = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            arms = {"normwmax@0.25": window(img, mw[qid][0], mw[qid][1], W),
                    "head@0.25":    window(img, *props[qid]["head"], W)}
            rec = {"question_id_full": qid, "category": ex["category"], "label": label,
                   "mw_cov": mw[qid][2], "mw_win": mw[qid][3], "probs": {}, "realized_tokens": {}}
            for nm, im in arms.items():
                fitted, _ = fit(im, B0)
                p, rz = answer(fitted, ex["text"])
                rec["probs"][nm] = p; rec["realized_tokens"][nm] = rz
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. {n} rows -> {OUT}", flush=True)


main()
