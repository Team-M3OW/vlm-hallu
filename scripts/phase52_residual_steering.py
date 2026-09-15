"""
Phase 52: RESIDUAL-STREAM STEERING. Is the oracle-crop state reachable by a linear shift?

WHY RUN THIS WHEN §10B PREDICTS FAILURE
---------------------------------------
§10B showed attention-space allocation recovers only 15.9% of the pixel-space oracle's +36.2pp, and
argued the reason is that at B0=300 the target occupies less than one merged token -- the oracle
amplification set is literally 1 cell of 294 -- so there is no token carrying the information to
reweight. That argument predicts residual steering fails too, for the same reason. A prediction that
is only asserted is worth little; this measures it.

THE DESIGN, AND WHY IT IS DECISIVE EITHER WAY
---------------------------------------------
For each item we run the SAME prompt twice: once on the uniform image, once on the oracle crop (the
92.7% arm). Both are 300 tokens. The difference in the last token's residual at layer L,

        delta_i(L) = h_oracle_i(L) - h_uniform_i(L),

is the activation-space displacement produced by actually allocating the pixels. Steering asks
whether that displacement can be INJECTED instead of earned.

    v_item     inject the item's OWN delta. This is not a method -- it is a REACHABILITY TEST.
               If injecting the exact displacement that the oracle crop produces does not restore
               oracle accuracy, then the crop's benefit is not a single-layer linear shift of the
               residual, and no steering vector of this form can work. This is the arm that makes
               the experiment worth running.
    v_mean     the mean delta over OTHER items, 5-fold, applied out of fold. This is the actual
               method: a transferable "allocate to the evidence" direction.
    v_rand     a random direction matched in norm. Controls for "any perturbation of this size".

Sweeping alpha and layer, because a single flattering combination could otherwise be selected after
the fact. All steered arms are ONE pass at 294 tokens -- same cost as baseline, so paired and with
no compute bar.

STAGE 1 collects the deltas (2 passes/item). STAGE 2 runs the steered passes using fold-wise
vectors, so v_mean never sees the item it is applied to.
"""
import json
import os
import random
import time

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DELTAS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase52_deltas.pt"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase52_residual_steering.jsonl"
B0, PAD = 300, 0.25
LAYERS = [8, 14, 20, 26]
ALPHAS = [0.5, 1.0, 2.0]
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    rng = random.Random(52)
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    opt_ids = [sorted({tok(s, add_special_tokens=False)["input_ids"][-1]
                       for s in [C, f" {C}"]}) for C in "ABCD"]
    layers = model.model.language_model.layers
    nL = len(layers)
    print(f"{nL} decoder layers; steering at {LAYERS}", flush=True)

    STEER = {"L": None, "v": None}

    def mk_hook(idx):
        def hook(mod, args, out):
            if STEER["L"] != idx or STEER["v"] is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + STEER["v"].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    for i, l in enumerate(layers):
        l.register_forward_hook(mk_hook(i))

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // 4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=5, tol=0.04):
        W_, H_ = img.size
        sc = (target / max(measure(img), 1)) ** 0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(32, int(W_ * sc)), max(32, int(H_ * sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (cur, r)
            if r == 0 or abs(r - target) / target <= tol:
                break
            sc *= (target / r) ** 0.5
        return best[0]

    def run(inp, want_hidden=False):
        with torch.no_grad():
            o = model(**inp, output_hidden_states=want_hidden)
        lg = o.logits[0, -1].float()
        assert torch.isfinite(lg).any(), "non-finite logits"
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt_ids]), 0)
        p = [round(float(x), 6) for x in p.tolist()]
        if not want_hidden:
            return p, None
        # hidden_states[i+1] is the OUTPUT of layer i
        h = {i: o.hidden_states[i + 1][0, -1, :].detach().float().cpu() for i in LAYERS}
        del o
        return p, h

    items = []
    for ex in ds:
        ip = os.path.join(root, ex["image"])
        ap = os.path.splitext(ip)[0] + ".json"
        if not os.path.exists(ap):
            continue
        ann = json.load(open(ap))
        if not ann.get("bbox"):
            continue
        items.append((f"{ex['category']}/{ex['question_id']}", ip, ann, ex))

    # ---------------- STAGE 1: collect deltas
    if os.path.exists(DELTAS):
        D = torch.load(DELTAS)
        print(f"loaded deltas for {len(D['qid'])} items", flush=True)
    else:
        D = {"qid": [], "delta": {L: [] for L in LAYERS}, "base": [], "oracle": [], "label": [],
             "category": []}
        t0 = time.time()
        for k, (qid, ip, ann, ex) in enumerate(items):
            img = Image.open(ip).convert("RGB")
            IW, IH = img.size
            gx0 = min(b[0] for b in ann["bbox"]); gy0 = min(b[1] for b in ann["bbox"])
            gx1 = max(b[0] + b[2] for b in ann["bbox"]); gy1 = max(b[1] + b[3] for b in ann["bbox"])
            dx, dy = (gx1 - gx0) * PAD, (gy1 - gy0) * PAD
            ob = (max(0, gx0 - dx), max(0, gy0 - dy), min(IW, gx1 + dx), min(IH, gy1 + dy))
            text = ex["text"]
            STEER["L"] = None
            pu, hu = run(build(fit(img, B0), text).to(model.device), True)
            po, ho = run(build(fit(img.crop(tuple(int(v) for v in ob)), B0), text).to(model.device), True)
            D["qid"].append(qid)
            D["base"].append(pu)
            D["oracle"].append(po)
            D["label"].append("ABCD".index(ex["label"]) if isinstance(ex["label"], str)
                              else int(ex["label"]))
            D["category"].append(ex["category"])
            for L in LAYERS:
                D["delta"][L].append(ho[L] - hu[L])
            if (k + 1) % 30 == 0:
                el = time.time() - t0
                print(f"  stage1 [{k+1}/{len(items)}] eta="
                      f"{(len(items)-k-1)/max((k+1)/el,1e-9)/60:.1f}min", flush=True)
        for L in LAYERS:
            D["delta"][L] = torch.stack(D["delta"][L])
        torch.save(D, DELTAS)
        print(f"stage1 done, saved {DELTAS}", flush=True)

    n = len(D["qid"])
    hit = lambda p, y: 1.0 * (max(range(4), key=lambda i: p[i]) == y)
    print(f"\nbaseline {100*sum(hit(p,y) for p,y in zip(D['base'],D['label']))/n:.1f}%   "
          f"oracle {100*sum(hit(p,y) for p,y in zip(D['oracle'],D['label']))/n:.1f}%", flush=True)
    for L in LAYERS:
        dv = D["delta"][L]
        print(f"  L{L}: mean |delta| {dv.norm(dim=1).mean():.2f}   "
              f"|mean delta| {dv.mean(0).norm():.2f}   "
              f"cos(delta_i, mean) {torch.nn.functional.cosine_similarity(dv, dv.mean(0)[None], dim=1).mean():.3f}")

    # ---------------- STAGE 2: steered passes
    idx = list(range(n))
    random.Random(7).shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    foldof = {}
    for f, fo in enumerate(folds):
        for i in fo:
            foldof[i] = f
    vmean = {}
    for L in LAYERS:
        for f in range(5):
            tr = [i for i in idx if foldof[i] != f]
            vmean[(L, f)] = D["delta"][L][tr].mean(0)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["question_id_full"])
    t0 = time.time()
    with open(OUT, "a") as fout:
        for i, (qid, ip, ann, ex) in enumerate(items):
            if qid in done:
                continue
            img = Image.open(ip).convert("RGB")
            text = ex["text"]
            inp = build(fit(img, B0), text).to(model.device)
            rec = {"question_id_full": qid, "category": D["category"][i], "label": D["label"][i],
                   "probs": {"baseline": D["base"][i], "oracle": D["oracle"][i]}}
            for L in LAYERS:
                di = D["delta"][L][i].to(model.device)
                vm = vmean[(L, foldof[i])].to(model.device)
                g = torch.randn(di.shape[0], generator=torch.Generator(device="cpu").manual_seed(
                    52 * 1000 + i), device="cpu").to(model.device)
                vr = g / g.norm() * vm.norm()
                for nm, v in [("v_item", di), ("v_mean", vm), ("v_rand", vr)]:
                    for a in ALPHAS:
                        STEER["L"], STEER["v"] = L, a * v
                        p, _ = run(inp)
                        rec["probs"][f"{nm}@L{L}@a{a}"] = p
            STEER["L"], STEER["v"] = None, None
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  stage2 [{i+1}/{len(items)}] eta="
                      f"{(len(items)-i-1)/max((i+1)/el,1e-9)/60:.1f}min", flush=True)
    print(f"Done. Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
