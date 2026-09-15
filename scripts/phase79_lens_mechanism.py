"""
Phase 79: WHY DCR WORKS, read through the logit lens. The causal mechanism, not a correlation.

WHAT THE LENS ALREADY ESTABLISHED
---------------------------------
SS13/Phase 66: under uniform encoding the correct answer is decodable at NO LAYER -- no layer
exceeds 52.9% against the oracle crop's 92.6%, and the apparent "L20 peak" is exactly what a
near-chance layer produces. Under an ORACLE CROP the answer appears ABRUPTLY at ~L22, taking 90% of
the gain in a single transition. So cropping does not make the model reason better; it supplies
information that a specific depth then converts into an answer.

THE CLAIM THIS RUN TESTS
------------------------
    DCR works by moving items OUT of the "answer absent at every layer" regime and INTO the
    "answer forms at L22" regime.

and it carries its own control, fixed before the run:

    head arm, items where the proposal COVERS   -> trajectory should look like ORACLE
    head arm, items where the proposal MISSES   -> trajectory should look like UNIFORM

If the covered/missed split does NOT separate, the method's gain is not the restoration of answer
formation and this mechanism is wrong -- report that instead.

ARMS (identical crops to Phase 71, so the lens reads the arms that produced SS14D's numbers)
    uniform@300   the vanilla baseline: answer absent at every layer
    argmax@0.15   the incumbent proposer
    head@0.15     DCR                                    <- the method
    oracle@0.15   ceiling

THE LENS, IMPLEMENTED CORRECTLY -- this cost us a retraction once (SS12D)
------------------------------------------------------------------------
    intermediate layer i :  lm_head(final_norm(hidden_states[i]))
    FINAL layer          :  model.logits   -- NOT norm(hidden_states[-1])
HF has ALREADY normalised the last hidden state. Applying final_norm again gives reconstruction
error 23.47 against 0.06, corrupting ONLY the final layer, and it manufactured a "+5.1pp free
read-out gain", an "evidence-dependent late-layer bias", and a favourable comparison against DoLa.
The bug was caught because a second code path computed the same quantity and disagreed (100%
agreement at L24, 82.2% at the final layer). Both paths are computed here and their agreement is
ASSERTED, so the bug cannot come back silently.
"""
import json
import os
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROP = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase71a_head_proposals.json"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase79_lens_mechanism.jsonl"
B0, W = 300, 0.15
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    lm = model.model.language_model
    norm, head_w = lm.norm, model.lm_head
    print(f"{len(lm.layers)} layers; lens = lm_head(final_norm(h_i)), final from model.logits",
          flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.06):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best

    def opts(logits):
        return torch.softmax(torch.stack([torch.logsumexp(logits[ids], 0)
                                          for ids in opt_ids]), 0)

    def lens(img, text):
        inp = build(img, text).to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        H = out.hidden_states
        traj = []
        for i in range(1, len(H)):
            h = H[i][0, -1]
            if i == len(H) - 1:
                lg = out.logits[0, -1].float()          # HF already normalised this one
            else:
                lg = head_w(norm(h)).float()
            traj.append([round(float(v), 6) for v in opts(lg).tolist()])
        # cross-check: the naive path on the FINAL layer must DISAGREE with model.logits,
        # which is the signature of the SS12D bug. If it agrees, the model changed and the
        # lens needs re-deriving rather than trusting.
        naive_final = opts(head_w(norm(H[-1][0, -1])).float())
        true_final = opts(out.logits[0, -1].float())
        drift = float((naive_final - true_final).abs().sum())
        rz = int(sum(g[1] * g[2] // 4 for g in inp["image_grid_thw"].tolist()))
        del out
        torch.cuda.empty_cache()
        return traj, rz, drift

    def window(img, cx, cy, Wn):
        iw, ih = img.size
        x0, y0 = (cx - Wn / 2) * iw, (cy - Wn / 2) * ih
        x1, y1 = (cx + Wn / 2) * iw, (cy + Wn / 2) * ih
        if x0 < 0: x0, x1 = 0, Wn * iw
        if y0 < 0: y0, y1 = 0, Wn * ih
        if x1 > iw: x0, x1 = iw - Wn * iw, iw
        if y1 > ih: y0, y1 = ih - Wn * ih, ih
        return img.crop((int(x0), int(y0), int(max(x1, x0 + 8)), int(max(y1, y0 + 8))))

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    print(f"resuming: {len(done)}", flush=True)

    n, t0 = 0, time.time()
    for ex in ds:
        qid = f"{ex['category']}/{ex['question_id']}"
        ip = os.path.join(root, ex["image"])
        if qid in done or qid not in props or not os.path.exists(ip):
            continue
        img = Image.open(ip).convert("RGB")
        pp = props[qid]
        gt = pp["gt_box_frac"]
        gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        arms = {
            "uniform@300": fit(img, B0)[0],
            "argmax@0.15": fit(window(img, *pp["argmax"], W), B0)[0],
            "head@0.15":   fit(window(img, *pp["head"], W), B0)[0],
            "oracle@0.15": fit(window(img, gcx, gcy, W), B0)[0],
        }
        rec = {"question_id_full": qid, "category": ex["category"],
               "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
               "head_cov": pp["head_cov"], "argmax_cov": pp["argmax_cov"],
               "traj": {}, "realized_tokens": {}, "lens_drift": {}}
        for nm, im in arms.items():
            t, rz, d = lens(im, ex["text"])
            rec["traj"][nm] = t
            rec["realized_tokens"][nm] = rz
            rec["lens_drift"][nm] = round(d, 4)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 20 == 0:
            el = time.time() - t0
            print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
