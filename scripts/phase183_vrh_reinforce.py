"""
Phase 183: DPR -> VRH REINFORCEMENT. The eighth internal intervention, and the first with BOTH a
working localiser AND the retrieval channel.

    a'_{h,p} = a_{h,p} + lambda * M_R(p)     for VRH-selected heads h only

DPR is the spatial router (region R = its W=0.25 window); VRH heads are the retrieval channel. The
bias is added to the attention LOGITS pre-softmax, for keys inside R, on selected (layer, head) pairs.
Model sees the full image; nothing is cropped, added, or trained.

WHY THIS IS WORTH RUNNING DESPITE THE BOUND
    §10 already ran attention amplification: amp_oracle (GT region) +5.8pp [+1,+10], while top-1,
    top-5, top-15 and the deployed window were ALL indistinguishable from the random control -- but
    that used the OLD argmax proposer (~39% coverage) and amplified ALL heads. §14H re-ran the
    logit-space version with the learned head: cd_head +2.1pp [-1.1,+5.3]. Never run: the learned
    head as the target AND the intervention restricted to the heads that carry retrieval (§20H).
    Expected +2-4pp with a CI that may not clear at n=191. The value is closing the family: seven
    nulls, the eighth aimed at the right place through the right channel.

BAR. The intervention is FREE -- one modified forward pass, no second pass -- so its bar is
uniform@300, not uniform@600. It is not required to beat uniform@600.

ARMS (lambda=2 pre-registered as PRIMARY; the sweep is secondary and a sweep-selected lambda is NOT
a claim)
    base                 unpatched uniform@300
    vrh_dpr_l{1,2,4}     bias toward the DPR window, VRH heads only          <- THE METHOD
    all_dpr_l2           same bias, ALL heads        <- does head restriction matter (§10 amplified all)
    vrh_rand_l2          same bias, a random window  <- §10's control; must be null
    vrh_oracle_l2        same bias, the GT window    <- CEILING with a perfect router
    uniform@600          reference only, not the bar

CORRECTNESS GATE: the patch with lambda=0 must reproduce the unpatched logits exactly, or nothing
below is interpretable.
usage: phase183_vrh_reinforce.py <qwen3|qwen2> [--limit N]
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
LIMIT = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else None
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/data/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/data/phase80a_qwen2vl_proposals.json")}
MODEL_ID, PROP = CFG[WHICH]
OUT = f"{D}/data/phase183_reinforce_{WHICH}{'_sanity' if LIMIT else ''}.jsonl"
B0, W = 300, 0.25
Image.MAX_IMAGE_PIXELS = None
BIAS = {}          # layer_idx -> (1, H, 1, kv) additive tensor, or None


def get_lm_layers(model):
    for path in ["model.language_model.layers", "model.model.language_model.layers",
                 "model.layers", "model.model.layers", "language_model.model.layers"]:
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj, path
    raise RuntimeError("could not locate language-model layers")


def install(model):
    layers, path = get_lm_layers(model)
    print(f"  patched {len(layers)} attention modules at {path}", flush=True)
    for li, lyr in enumerate(layers):
        attn = lyr.self_attn
        orig = attn.forward

        def wrapper(*args, _li=li, _orig=orig, **kw):
            b = BIAS.get(_li)
            if b is not None and kw.get("attention_mask") is not None:
                kw["attention_mask"] = kw["attention_mask"] + b.to(kw["attention_mask"].dtype)
            return _orig(*args, **kw)
        attn.forward = wrapper
    return len(layers)


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    from sklearn.model_selection import GroupKFold
    props = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    img_tok_id = model.config.image_token_id
    opt = [sorted({tok(c, add_special_tokens=False)["input_ids"][-1] for c in [C, f" {C}"]})
           for C in "ABCD"]
    NL = install(model)

    buf = np.load(f"{D}/data/phase170_perhead_{WHICH}.npy")
    meta = json.load(open(f"{D}/data/phase170_perhead_{WHICH}_index.json"))
    H, pidx = meta["n_heads"], meta["items"]
    ref = np.zeros((len(pidx), NL, H), np.float32)
    for i, e in enumerate(pidx):
        q = e["question_id_full"]
        if q not in props: continue
        A = buf[:, e["offset"]:e["offset"]+e["n_cells"]].astype(np.float32).reshape(NL, H, -1)
        gh, gw = e["grid"]; n = e["n_cells"]
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel()[:n], ((xx+.5)/gw).ravel()[:n]
        g = props[q]["gt_box_frac"]
        m = (fx >= g[0]) & (fx <= g[2]) & (fy >= g[1]) & (fy <= g[3])
        ref[i] = A[:, :, m].sum(-1) if m.any() else 0.0
    k = max(1, int(round(0.25*H)))
    HEADSET = {}
    G = np.arange(len(pidx))
    for tr, te in GroupKFold(5).split(G.reshape(-1, 1), G, G):
        sc = ref[tr].mean(0)
        mask = np.zeros((NL, H), bool)
        for l in range(NL):
            mask[l, np.argsort(-sc[l])[:k]] = True
        for i in te:
            HEADSET[pidx[i]["question_id_full"]] = mask

    def chat(text):
        return pr.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=True)

    def build(img, text):
        return pr(images=img, text=chat(text), return_tensors="pt")

    def measure(img):
        return int(sum(g[1]*g[2]//4 for g in build(img, "x")["image_grid_thw"].tolist()))

    def fit(img, target, refine=6, tol=0.06):
        W_, H_ = img.size
        sc = (target/max(measure(img), 1))**0.5
        best = None
        for _ in range(refine):
            cur = img.resize((max(28, int(W_*sc)), max(28, int(H_*sc))), Image.BICUBIC)
            r = measure(cur)
            if best is None or abs(r-target) < abs(best[1]-target): best = (cur, r)
            if r == 0 or abs(r-target)/target <= tol: break
            sc *= (target/r)**0.5
        return best[0]

    def run(inp, heads, keymask, lam):
        BIAS.clear()
        if lam != 0:
            km = torch.tensor(keymask, device=model.device).view(1, 1, 1, -1).float()
            for l in range(NL):
                hm = (torch.ones(H, device=model.device) if heads is None
                      else torch.tensor(heads[l], device=model.device).float())
                BIAS[l] = lam * km * hm.view(1, -1, 1, 1)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        BIAS.clear()
        assert torch.isfinite(lg).all()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[x], 0) for x in opt]), 0)
        return [round(float(v), 6) for v in p.tolist()]

    byq = {f"{e['category']}/{e['question_id']}": e for e in ds}
    items = [q for q in props if q in byq and q in HEADSET]
    if LIMIT: items = items[:LIMIT]
    rng = np.random.default_rng(183)
    t0, done, gate_ok = time.time(), 0, None
    with open(OUT, "w") as fo:
        for qid in items:
            ex = byq[qid]; p = props[qid]
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            inp = build(fit(img, B0), ex["text"])
            g3 = inp["image_grid_thw"][0].tolist(); gh, gw = g3[1]//2, g3[2]//2
            n_img = gh*gw
            ipos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
            base_i = int(ipos[0].item()); kv = inp["input_ids"].shape[1]
            inp = inp.to(model.device)
            yy, xx = np.mgrid[0:gh, 0:gw]
            fy, fx = ((yy+.5)/gh).ravel()[:n_img], ((xx+.5)/gw).ravel()[:n_img]

            def keymask_for(cx, cy):
                x0 = min(max(0, cx-W/2), 1-W); y0 = min(max(0, cy-W/2), 1-W)
                sel = (fx >= x0) & (fx <= x0+W) & (fy >= y0) & (fy <= y0+W)
                km = np.zeros(kv, bool); km[base_i:base_i+n_img] = sel
                return km

            gt = p["gt_box_frac"]
            km_dpr = keymask_for(*p["head"])
            km_orc = keymask_for((gt[0]+gt[2])/2, (gt[1]+gt[3])/2)
            km_rnd = keymask_for(float(rng.uniform(W/2, 1-W/2)), float(rng.uniform(W/2, 1-W/2)))
            hs = HEADSET[qid]

            probs = {"base": run(inp, None, km_dpr, 0.0)}
            if gate_ok is None:
                z = run(inp, hs, km_dpr, 0.0)
                gate_ok = (z == probs["base"])
                print(f"  CORRECTNESS GATE lambda=0 reproduces unpatched: {gate_ok}", flush=True)
                assert gate_ok, "patch is not a no-op at lambda=0"
            for lam in (1.0, 2.0, 4.0):
                probs[f"vrh_dpr_l{int(lam)}"] = run(inp, hs, km_dpr, lam)
            probs["all_dpr_l2"] = run(inp, None, km_dpr, 2.0)
            probs["vrh_rand_l2"] = run(inp, hs, km_rnd, 2.0)
            probs["vrh_oracle_l2"] = run(inp, hs, km_orc, 2.0)
            inp600 = build(fit(img, 2*B0), ex["text"]).to(model.device)
            probs["uniform@600"] = run(inp600, None,
                                       np.zeros(inp600["input_ids"].shape[1], bool), 0.0)

            lab = "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"])
            fo.write(json.dumps({"question_id_full": qid, "category": ex["category"], "label": lab,
                                 "head_cov": p.get("head_cov"), "probs": probs}) + "\n")
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(items)}] {done/(time.time()-t0):.2f} it/s", flush=True)
    print(f"wrote {OUT}  n={done}")


if __name__ == "__main__":
    main()
