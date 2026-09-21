"""
Phase 204e (GPU): does the mechanism correction survive at the END TASK?

The CPU phase 204 showed non-negative (sparse) depth weighting retains TWR's coverage advantage, so the
paper's "the correction requires subtraction / a mean can only add" account is not causal. This run puts
the decisive arms through the deployed pipeline (localise@300 -> crop W=0.25 -> answer@300, bar
uniform@600), same items, paired.

Arms: uniform@600 (bar), ridge (deployed TWR), nnls (non-negative log-a + rank weights), map_free
(28 free map-space weights: the "a mean can only add" test in its cleanest form).

PRE-REGISTERED
  P1  nnls - ridge pooled: CI must NOT show a loss for the non-negative arm on either model
      (if signedness were necessary, the non-negative arm would lose).
  P2  map_free - bar clears zero on both models (a signed map-space filter is a usable read-out).
  P3  ridge - bar replicates the published +8.9 / +12.0 on the same items.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
from scipy.optimize import lsq_linear
from sklearn.model_selection import GroupKFold
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
sys.path.insert(0, f"{D}/scripts")
import phase70_rerank_head as P70, phase80a_qwen2vl_head as P80
WHICH = sys.argv[1]
MODEL_ID = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct", "qwen2": "Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT = f"{D}/data/phase204_endtask_{WHICH}.jsonl"; B0, W = 300, 0.25; Image.MAX_IMAGE_PIXELS = None
NL, BLK = (28, (16, 27)) if WHICH == "qwen3" else (28, (15, 27))
SEEDS = (700, 701, 702)


def covfn(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def placements():
    build = P70.build if WHICH == "qwen3" else P80.build
    P70.W = W
    r = build(); rows = r[4]
    Xr, Yr, Gr, Xmap = [], [], [], []
    for gi, q in enumerate(rows):
        gh, gw = q["grid"]; n = q["n_img_tokens"]
        A = np.stack([np.asarray(q["attn"][f"L{i}"], float) for i in range(NL)])
        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        LA = np.log(A + 1e-12).T
        R = (np.argsort(np.argsort(-A, axis=1), axis=1) / max(n - 1, 1)).T
        dep = A[BLK[0]:BLK[1]].mean(0).reshape(gh, gw); pad = np.pad(dep, 1, mode="edge")
        nb = (sum(pad[i:i + gh, j:j + gw] for i in range(3) for j in range(3)) / 9.0).ravel()
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
        geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                    np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                    (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
        Xr.append(np.c_[LA, R, geo]); Xmap.append(A.T)
        Yr.append(np.array([covfn(float(fx[i]), float(fy[i]), q["gt_box_frac"]) for i in range(n)]))
        Gr.append(np.full(n, gi))
    Xr = np.vstack(Xr); Xmap = np.vstack(Xmap); Yr = np.concatenate(Yr); Gr = np.concatenate(Gr)
    gs = np.unique(Gr)

    def oof(X, kind):
        P = np.zeros(len(Yr))
        for s in SEEDS:
            rng = np.random.default_rng(s); perm = {g: i for i, g in enumerate(rng.permutation(gs))}
            Gp = np.vectorize(perm.get)(Gr)
            for tr, te in GroupKFold(5).split(X, Yr, Gp):
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Xt = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
                if kind == "ridge":
                    A_ = Xt.T @ Xt + np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
                    w = np.linalg.solve(A_, Xt.T @ Yr[tr])
                else:
                    Aa = np.vstack([Xt, np.sqrt(1.0) * np.c_[np.eye(X.shape[1]), np.zeros(X.shape[1])[:, None]]])
                    ba = np.concatenate([Yr[tr], np.zeros(X.shape[1])])
                    lb = np.r_[np.zeros(56), -np.inf * np.ones(X.shape[1] - 56), -np.inf]
                    w = lsq_linear(Aa, ba, bounds=(lb, np.inf * np.ones(X.shape[1] + 1)),
                                   method="bvls", max_iter=200).x
                P[te] += np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
        return P / len(SEEDS)

    Pr, Pn, Pm = oof(Xr, "ridge"), oof(Xr, "nnls"), oof(Xmap, "ridge")
    out = {}
    for gi, q in enumerate(rows):
        gh, gw = q["grid"]; m = Gr == gi; rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        rm = rm.ravel()
        c = lambda j: [float((j % gw + .5) / gw), float((j // gw + .5) / gh)]
        out[q["question_id_full"]] = {"ridge": c(int(np.argmax(np.where(rm, Pr[m], -1e9)))),
                                      "nnls": c(int(np.argmax(np.where(rm, Pn[m], -1e9)))),
                                      "map_free": c(int(np.argmax(np.where(rm, Pm[m], -1e9))))}
    return out


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText
    PL = placements()
    print(f"placements for {len(PL)} items", flush=True)
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}).eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer
    opt = [sorted({tok(x, add_special_tokens=False)["input_ids"][-1] for x in [c, f" {c}"]}) for c in "ABCD"]
    def chat(n, t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * n + [{"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
    def build(i, t): return pr(images=i, text=chat(len(i), t), return_tensors="pt")
    def measure(i): return int(sum(g[1] * g[2] // 4 for g in build([i], "x")["image_grid_thw"].tolist()))
    def fit(img, target, refine=6, tol=0.08):
        W_, H_ = img.size; sc = (target / max(measure(img), 1)) ** 0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC); rr = measure(cur)
            if best is None or abs(rr - target) < abs(best[1] - target): best = (cur, rr)
            if rr == 0 or abs(rr - target) / target <= tol: break
            sc *= (target / rr) ** 0.5
        return best[0]
    def win(img, cx, cy, w):
        iw, ih = img.size; x0 = min(max(0, (cx - w / 2) * iw), iw - w * iw); y0 = min(max(0, (cy - w / 2) * ih), ih - w * ih)
        return img.crop((int(x0), int(y0), int(x0 + w * iw), int(y0 + w * ih)))
    def answer(imgs, t):
        inp = build(imgs, t); rz = int(sum(g[1] * g[2] // 4 for g in inp["image_grid_thw"].tolist())); inp = inp.to(model.device)
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)
        del inp; torch.cuda.empty_cache(); return [round(float(v), 6) for v in p.tolist()], rz
    t0, n = time.time(), 0
    with open(OUT, "w") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"; ip = os.path.join(root, ex["image"])
            if qid not in PL or not os.path.exists(ip): continue
            p = PL[qid]; img = Image.open(ip).convert("RGB")
            arms = {"uniform@600": [fit(img, 2 * B0)]}
            for k in ("ridge", "nnls", "map_free"):
                arms[k] = [fit(win(img, *p[k], W), B0)]
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
                   "probs": {}, "realized_tokens": {}}
            for nm, ims in arms.items():
                pr_, rz = answer(ims, ex["text"]); rec["probs"][nm] = pr_; rec["realized_tokens"][nm] = rz
            fout.write(json.dumps(rec) + "\n"); fout.flush(); n += 1
            if n % 25 == 0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    print(f"Done -> {OUT}", flush=True)


main()
