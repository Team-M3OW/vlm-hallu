"""
M2 (GPU): THE BUDGET-OPTIMAL ALLOCATION POLICY AT A NEW BUDGET POINT.

The two policies are corners of one allocation problem: with a token-layer budget B, spend it on (i) placement
(crop: localise@300 + answer@E_c) or (ii) resolution (AVR: encode@E, prune 10% at the boundary). The policy
predicts each arm's accuracy from two baseline runs and the measured placement gain, then picks per stratum.

Arms at B = 1200 token-layers (bar = 1200*28 = 33,600):
  uniform@1200        the equal-compute bar, one pass at 1200
  DWA@1200            cached localise@300 + crop@900 at the ridge cell       TL = 300*28 + nt*28
  AVR@1200            encode@1800, keep 10% at L16                          TL = nt*17 + keep*11 <= 33,600
  uniform@1800        diagnostic only (over budget), the AVR prediction source

PRE-REGISTERED
  P1  AVR@1200 ~= uniform@1800 (the identity at a new budget): |diff| <= 1.5 points pooled, both models.
  P2  DWA@1200 - uniform@1200 is positive and CI-clear on single-instance, both models (the placement gain
      carries from a 300- to a 900-token crop).
  P3  the policy's per-stratum pick (DWA on single, AVR on cross) is within 3 points of the better realized
      arm in all four cells (2 models x 2 strata).
"""
import json, os, sys, time, random, numpy as np, torch
from PIL import Image
os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import transformers.models.qwen3_vl.modeling_qwen3_vl as QM3
import transformers.models.qwen2_vl.modeling_qwen2_vl as QM2
from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.model_selection import GroupKFold
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
WHICH = sys.argv[1]
MODEL_ID = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct", "qwen2": "Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
MAPS = {"qwen3": ("phase30c_attn_maps_all.jsonl", (16, 27)), "qwen2": ("phase74_Qwen2_VL_7B_Instruct.jsonl", (15, 27))}
OUT = f"{D}/data/m2_budget_policy_{WHICH}.jsonl"; NL = 28; P = 16; W = 0.25; K = 0.10
B_TOK = 1200; E_AVR = 1800; Image.MAX_IMAGE_PIXELS = None
SEEDS = (700, 701, 702)


def cov(cx, cy, gt):
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0., W
    if y0 < 0: y0, y1 = 0., W
    if x1 > 1: x0, x1 = 1 - W, 1.
    if y1 > 1: y0, y1 = 1 - W, 1.
    gx0, gy0, gx1, gy1 = gt
    return max(0., min(gx1, x1) - max(gx0, x0)) * max(0., min(gy1, y1) - max(gy0, y0)) / max((gx1 - gx0) * (gy1 - gy0), 1e-12)


def feats_from(a, gh, gw, BLK, xx, yy, fx, fy):
    LA = np.log(a + 1e-12).T
    R = (np.argsort(np.argsort(-a, axis=1), axis=1) / max(gh * gw - 1, 1)).T
    dep = a[BLK[0]:BLK[1]].mean(0).reshape(gh, gw)
    pad = np.pad(dep, 1, mode="edge")
    nb = (sum(pad[p:p + gh, q:q + gw] for p in range(3) for q in range(3)) / 9.0).ravel()
    geo = np.c_[np.log(nb + 1e-12), fx, fy, np.sqrt((fx - .5) ** 2 + (fy - .5) ** 2),
                np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)),
                (xx.ravel() == gw - 1).astype(float), (yy.ravel() == gh - 1).astype(float)]
    return np.c_[LA, R, geo]


def placements():
    fn, BLK = MAPS[WHICH]
    rows = [json.loads(l) for l in open(f"{D}/data/{fn}")]
    rows = [r for r in rows if "attn" in r and "grid" in r and "gt_box_frac" in r]
    X, Y, G, cells = [], [], [], {}
    for gi, r in enumerate(rows):
        gh, gw = r["grid"]; n = r["n_img_tokens"]; nc = gh * gw
        yy, xx = np.mgrid[0:gh, 0:gw]; fy, fx = ((yy + .5) / gh).ravel(), ((xx + .5) / gw).ravel()
        A = np.stack([np.asarray(r["attn"][f"L{l}"], float) for l in range(NL)]); A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        X.append(feats_from(A, gh, gw, BLK, xx, yy, fx, fy))
        Y.append(np.array([cov(float(fx[c]), float(fy[c]), r["gt_box_frac"]) for c in range(nc)]))
        G.append(np.full(nc, gi)); cells[r["question_id_full"]] = (gh, gw, nc)
    X = np.vstack(X); Y = np.concatenate(Y); Gr = np.concatenate(G)
    P_ = np.zeros(len(Y)); gs = np.unique(Gr)
    for s in SEEDS:
        rng = np.random.default_rng(s); perm = {g: i for i, g in enumerate(rng.permutation(gs))}
        Gp = np.vectorize(perm.get)(Gr)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xt = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
            A_ = Xt.T @ Xt + np.eye(Xt.shape[1]); A_[-1, -1] -= 1.0
            w = np.linalg.solve(A_, Xt.T @ Y[tr])
            P_[te] += np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ w
    P_ /= len(SEEDS)
    out = {}
    for gi, r in enumerate(rows):
        gh, gw, nc = cells[r["question_id_full"]]
        rm = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: rm[1:-1, 1:-1] = True
        else: rm[:] = True
        m = Gr == gi
        j = int(np.argmax(np.where(rm.ravel(), P_[m], -1e9)))
        out[r["question_id_full"]] = [float((j % gw + .5) / gw), float((j // gw + .5) / gh)]
    return out


def make_patched(QM):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        ks = QM.repeat_kv(key, module.num_key_value_groups); vs = QM.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attention_mask is not None: w = w + attention_mask[:, :, :, :ks.shape[-2]]
        b = getattr(module, "_prune_bias", None)
        if b is not None and b.shape[-1] == w.shape[-1]: w = w + b.to(w.dtype).view(1, 1, 1, -1)
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, vs).transpose(1, 2).contiguous(), w
    return patched


QM3.eager_attention_forward = make_patched(QM3)
QM2.eager_attention_forward = make_patched(QM2)


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID); tok = pr.tokenizer; itid = model.config.image_token_id
    layers = model.model.language_model.layers; assert len(layers) == NL
    opt = [sorted({tok(x, add_special_tokens=False)["input_ids"][-1] for x in [c, f" {c}"]}) for c in "ABCD"]
    PL = placements()
    print(f"placements for {len(PL)} items", flush=True)

    def chat(t): return pr.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], tokenize=False, add_generation_prompt=True)
    def build(i, t): return pr(images=i, text=chat(t), return_tensors="pt")
    def measure(i): return int(sum(g[1] * g[2] // 4 for g in build(i, "x")["image_grid_thw"].tolist()))
    def fit(img, target, refine=6, tol=0.06):
        W_, H_ = img.size; sc = (target / max(measure(img), 1)) ** 0.5; best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_ * sc)), max(28, int(H_ * sc))), Image.BICUBIC); r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target): best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol: break
            sc *= (target / r) ** 0.5
        return best
    def win(img, cx, cy, w):
        iw, ih = img.size; x0 = min(max(0, (cx - w / 2) * iw), iw - w * iw); y0 = min(max(0, (cy - w / 2) * ih), ih - w * ih)
        return img.crop((int(x0), int(y0), int(x0 + w * iw), int(y0 + w * ih)))
    def clear():
        for l in layers:
            if hasattr(l.self_attn, "_prune_bias"): del l.self_attn._prune_bias

    def run(inp, drop=None, from_layer=None, want_attn=False):
        clear()
        if drop is not None and len(drop):
            b = torch.zeros(inp["input_ids"].shape[1], device=model.device)
            b[torch.as_tensor(drop, device=model.device)] = -1e4
            for li in range(from_layer, NL): layers[li].self_attn._prune_bias = b
        with torch.no_grad(): out = model(**inp, output_attentions=want_attn)
        lg = out.logits[0, -1].float()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)
        A = None
        if want_attn: A = np.stack([out.attentions[L][0, :, -1, :].float().mean(0).cpu().numpy() for L in range(NL)])
        del out; torch.cuda.empty_cache(); clear()
        return [round(float(v), 6) for v in p.tolist()], A

    rng = random.Random(212); n = 0; t0 = time.time(); done = set()
    if os.path.exists(OUT): done = {json.loads(l)["question_id_full"] for l in open(OUT)}
    with open(OUT, "a") as fout:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}"; ip = os.path.join(root, ex["image"])
            if qid in done or qid not in PL or not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            lab = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            u12, _ = fit(img, B_TOK); u18, _ = fit(img, E_AVR); c9, _ = fit(win(img, *PL[qid], W), 900)
            rec = {"question_id_full": qid, "category": ex["category"], "label": lab, "probs": {}, "realized_tokens": {}, "token_layers": {}}
            for nm, im in (("uniform@1200", u12), ("uniform@1800", u18), ("dwa@1200", c9)):
                inp = build(im, ex["text"]).to(model.device)
                nt = int((inp["input_ids"][0] == itid).sum())
                p, _ = run(inp)
                rec["probs"][nm] = p; rec["realized_tokens"][nm] = nt
                rec["token_layers"][nm] = nt * NL
            inp = build(u18, ex["text"]).to(model.device)
            pos = (inp["input_ids"][0] == itid).nonzero().flatten(); base, nt = int(pos[0]), int(len(pos))
            _, A = run(inp, want_attn=True)
            Ai = A[:, base:base + nt]; Ai = Ai / np.maximum(Ai.sum(1, keepdims=True), 1e-12)
            s = Ai[max(0, P - 4):P + 1].mean(0)
            keep = max(1, int(round(K * nt))); drop = (base + np.argsort(-s)[keep:]).tolist()
            p, _ = run(inp, drop=drop, from_layer=P + 1)
            rec["probs"]["avr@1200"] = p; rec["realized_tokens"]["avr@1200"] = nt
            rec["token_layers"]["avr@1200"] = nt * (P + 1) + keep * (NL - 1 - P)
            fout.write(json.dumps(rec) + "\n"); fout.flush(); n += 1
            if n % 20 == 0: print(f"  [{n}] {n/(time.time()-t0):.2f} it/s", flush=True)
    print(f"Done -> {OUT}", flush=True)


main()
