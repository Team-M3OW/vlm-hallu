"""
Phase 188: LAYER-WISE (PROGRESSIVE) TOKEN PRUNING, RANKED BY THE DPR HEAD.

Two things this repo has never combined:
  (1) Every pruning arm here is ONE cut at ONE depth (FASTV_K=2, phases 75/83/87/93). No progressive
      schedule -- drop a fraction at L2, again at L8, again at L16 -- exists anywhere in the codebase.
  (2) No pruning arm has ever ranked tokens with the 65-feature GBT head. The closest is phase 75's
      `linear` arm, the §14F signed LINEAR layer combination. The full head is worth +13.6pp of top-1
      coverage over the block mean (39.3 -> 52.9).

WHY THE COMBINATION IS THE INTERESTING ONE
    A progressive schedule needs a ranking AT EACH CUT, and §14L/Track A shows an early read is worse
    than RANDOM (-3.7 / -4.7pp) while a late read beats it by +21.4 / +11.5pp. So any progressive
    method that ranks at its first (early) cut inherits that failure by construction. The DPR head
    reads the whole 28-layer profile, so it can supply a LATE-INFORMED ranking to an EARLY cut --
    the principled version of what phase 94's blind-early-cut two-stage arm was groping toward.
    §13B's objection does not apply: pruning adds no pixels, so "attention cannot create tokens" is
    not a bound here. This is a pure ranking problem, which is the one thing the head is good at.

HOW IT WOULD ENHANCE DPR (stated so the result is scored against the right thing)
    Not accuracy -- pruning removes tokens, it does not add pixels. DPR costs localise@300 +
    crop@300 = 600, which is exactly why its bar is uniform@600. If the head can prune the crop pass
    with no accuracy loss, DPR's cost drops BELOW its own bar, and the method wins on compute where
    it currently only ties. That is the claim this is set up to test.

ARMS (all prune the SAME token count, both passes counted, so the contrast is RANKING QUALITY only)
    none                                   no pruning -- upper bound
    single cut @L2:   rand | layer2 | blockmean | head
    progressive L2/L8/L16 (equal thirds):  rand | blockmean | head
KEEP FRACTIONS: 10%, 25%.
PRE-REGISTERED
    P1  head - blockmean at 10% keep, SINGLE cut, CI clear of zero on BOTH models -> the head is a
        better pruning ranker, and Track A's read-depth claim extends from raw attention to a
        learned read-out.
    P2  progressive-head - single-cut-head, both models -> a late-informed ranking rescues the
        schedule that early reads cannot support.
    GUARD  every arm must beat `rand` at the same keep fraction, else the setup is not measuring
        ranking quality at all.
Head scores are read from phase 188a (out-of-fold, grouped by item). No fitting happens in this file.
usage: phase188_layerwise_prune.py <qwen3|qwen2> [--limit N]
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
MODEL_ID = {"qwen3": "Qwen/Qwen3-VL-2B-Instruct", "qwen2": "Qwen/Qwen2-VL-7B-Instruct"}[WHICH]
OUT = f"{D}/data/phase188_prune_{WHICH}{'_sanity' if LIMIT else ''}.jsonl"
B0, FASTV_K, BLOCK = 300, 2, list(range(16, 27))
STAGES = [2, 8, 16]
KEEP = [0.10, 0.25]
Image.MAX_IMAGE_PIXELS = None
PB = {}          # layer_idx -> additive bias over kv positions


def install(model):
    for path in ["model.language_model.layers", "model.model.language_model.layers",
                 "model.layers", "model.model.layers"]:
        obj = model
        try:
            for a in path.split("."): obj = getattr(obj, a)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList): layers = obj; break
    else:
        raise RuntimeError("no layers")
    for li, lyr in enumerate(layers):
        attn = lyr.self_attn
        orig = attn.forward

        def wrapper(*args, _li=li, _orig=orig, **kw):
            b = PB.get(_li)
            if b is not None and kw.get("attention_mask") is not None:
                kw["attention_mask"] = kw["attention_mask"] + b.to(kw["attention_mask"].dtype)
            return _orig(*args, **kw)
        attn.forward = wrapper
    return len(layers)


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    HS = json.load(open(f"{D}/data/phase188a_headscores_{WHICH}.json"))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    tok = pr.tokenizer
    itid = model.config.image_token_id
    opt = [sorted({tok(c, add_special_tokens=False)["input_ids"][-1] for c in [C, f" {C}"]})
           for C in "ABCD"]
    NL = install(model)

    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False,
                                                          add_generation_prompt=True),
                  return_tensors="pt")

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

    def run(inp, schedule=None, want_attn=False):
        """schedule: list of (from_layer, drop_indices). Bias is cumulative from each stage on."""
        PB.clear()
        if schedule:
            n = inp["input_ids"].shape[1]
            cum = torch.zeros(n, device=model.device)
            bounds = [s[0] for s in schedule] + [NL]
            for si, (fl, drop) in enumerate(schedule):
                if len(drop):
                    cum = cum.clone()
                    cum[torch.as_tensor(np.asarray(drop), device=model.device)] = -1e4
                for li in range(bounds[si], bounds[si+1]):
                    PB[li] = cum.view(1, 1, 1, -1)
        with torch.no_grad():
            out = model(**inp, output_attentions=want_attn)
        lg = out.logits[0, -1].float()
        assert torch.isfinite(lg).all()
        p = torch.softmax(torch.stack([torch.logsumexp(lg[i], 0) for i in opt]), 0)
        A = None
        if want_attn:
            A = np.stack([out.attentions[L][0, :, -1, :].float().mean(0).cpu().numpy()
                          for L in range(len(out.attentions))])
        del out; PB.clear(); torch.cuda.empty_cache()
        return [round(float(v), 6) for v in p.tolist()], A

    byq = {f"{e['category']}/{e['question_id']}": e for e in ds}
    items = [q for q in HS if q in byq]
    if LIMIT: items = items[:LIMIT]
    rng = np.random.default_rng(188)
    t0, done = time.time(), 0
    with open(OUT, "w") as fo:
        for qid in items:
            ex = byq[qid]
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img = Image.open(ip).convert("RGB")
            inp = build(fit(img, B0), ex["text"]).to(model.device)
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            base, ntok = int(pos[0].item()), int(len(pos))
            hs = np.asarray(HS[qid], float)
            if len(hs) != ntok:      # grid drift between extraction and now
                continue
            p_none, A = run(inp, want_attn=True)
            Ai = A[:, base:base+ntok]
            Ai = Ai/np.maximum(Ai.sum(1, keepdims=True), 1e-12)
            scores = {"rand": rng.random(ntok), "layer2": Ai[FASTV_K],
                      "blockmean": Ai[BLOCK].mean(0), "head": hs}
            rec = {"question_id_full": qid, "category": ex["category"],
                   "label": "ABCD".index(ex["label"]) if isinstance(ex["label"], str) else int(ex["label"]),
                   "n_img": ntok, "probs": {"none": p_none}}
            for kf in KEEP:
                keep_n = max(1, int(round(kf*ntok)))
                for nm, s in scores.items():
                    order = np.argsort(-s)
                    drop = (base + order[keep_n:]).tolist()
                    rec["probs"][f"single_{nm}@{kf}"], _ = run(inp, [(FASTV_K, drop)])
                # progressive: three equal cuts down to the same final keep count
                for nm in ["rand", "blockmean", "head"]:
                    order = np.argsort(-scores[nm])
                    sched, prev = [], ntok
                    for si, L in enumerate(STAGES):
                        frac = (ntok/keep_n)**((si+1)/len(STAGES))
                        tgt = max(keep_n, int(round(ntok/frac)))
                        sched.append((L, (base + order[tgt:prev]).tolist()))
                        prev = tgt
                    rec["probs"][f"prog_{nm}@{kf}"], _ = run(inp, sched)
            fo.write(json.dumps(rec) + "\n")
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(items)}] {done/(time.time()-t0):.2f} it/s", flush=True)
    print(f"wrote {OUT}  n={done}")


if __name__ == "__main__":
    main()
