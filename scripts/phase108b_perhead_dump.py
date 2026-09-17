"""
Phase 108b: dump PER-HEAD attention maps so head selection can be done OUT-OF-FOLD offline.

Phase 108 aggregated inside the GPU script, so its CoRe arms selected heads using each item's OWN
ground-truth box -- in-sample leakage. CoRe itself selects heads on a held-out subset and transfers;
that protocol cannot be reconstructed from aggregated maps. This dumps the raw per-head maps once
(float16, ~51MB) so every head rule, including a fold-honest CoRe, is an offline decision from here
on and no further GPU time is needed to change the protocol.

Their diagnosis is that heads divide functionally: some attend mostly to visual tokens ("perception
heads"), others mostly to text ("reasoning heads"), and treating all heads uniformly dilutes both.
They compute the modality attention ratio S_v(h) = the fraction of a head's attention mass that lands
on visual tokens, threshold it (tau_perc = 0.22), and find ~6.4% of heads qualify -- a deliberately
sparse set.

EVERY LOCATOR IN THIS PROJECT AVERAGES OVER HEADS. Phase 105 showed that fixing the LAYER rule (max
instead of mean) is worth +7 to +12pp of coverage; this is the same move one axis over.

⚠ BUT CoRe (Tran et al., arXiv 2510.02219) shows the ABSOLUTE criterion is the wrong one. Their
"QR head" baseline selects heads by absolute attention to the positive document -- structurally
identical to selecting by S_v here -- and they prove (Prop. 4.2) it can pick heads that also flood
irrelevant content, and measure top-8 QR heads DEGRADING re-ranking below the all-heads baseline.
Their fix is a head-level CONTRASTIVE score, a softmax of the positive against the negatives, which
they show is a head-level InfoNCE objective.

The vision port is direct, and should matter MORE here than in text: our serialisation sink absorbs
41-45% of the selected mass (SS5), so a head that floods the whole image scores high under S_v and
low under a contrastive rule. Both criteria are computed, so the comparison CoRe makes in text is
made here in vision.

Saves per-layer maps for six head-selection rules so the layer rule (mean vs max vs max_win4) can be
crossed with the head rule off-line, and stores S_v itself for the analysis of how sparse the useful
set really is.

    allheads      mean over all heads                       <- what we have always done
    top10/25/50   top q% of heads by S_v, within each layer   <- Lu et al. / QR-style, ABSOLUTE
    sv_weighted   heads averaged with weights proportional to S_v
    tau022        Lu et al.'s own threshold, S_v >= 0.22
    core10/25     top q% of heads by the CONTRASTIVE score    <- CoRe-style
    core_w        heads averaged with weights proportional to the contrastive score

CONTRASTIVE HEAD SCORE (the vision analogue of CoRe eq. 6). For head h at layer l, with E the cells
covering the GT box and E~ the ring-masked cells outside it:

    S_core(h) = softmax over {mean attention on E, mean attention on E~} at temperature t,

i.e. how much this head prefers evidence cells to non-evidence cells, NOT how much of its mass is on
the image. Selection uses the GT box, so -- exactly as in CoRe, which selects heads on a small NQ
subset and transfers -- the head set must be chosen OUT-OF-FOLD in the analysis step and never on
the items it is evaluated on.

Any q or threshold used for a headline number must be chosen OUT-OF-FOLD in the analysis step.
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
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/phase80a_qwen2vl_proposals.json")}
MODEL_ID, PROP = CFG[WHICH]
OUT = f"{D}/phase108b_perhead_{WHICH}.npz"
B0, TAU = 300, 0.22
Image.MAX_IMAGE_PIXELS = None


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    props = json.load(open(PROP))
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    img_tok_id = model.config.image_token_id

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

    t0, n = time.time(), 0
    store, meta = {}, []
    if True:
        for ex in ds:
            qid = f"{ex['category']}/{ex['question_id']}" if "question_id" in ex else ex.get("question_id_full")
            if qid not in props: continue
            ip = os.path.join(root, ex["image"])
            if not os.path.exists(ip): continue
            img, _ = fit(Image.open(ip).convert("RGB"), B0)
            inp = build(img, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1]//2, g[2]//2
            n_img = gh*gw
            pos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
            base = int(pos[0].item())
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            A = torch.stack([a[0].float() for a in out.attentions])        # (NL, H, n, n)
            del out
            last = A.shape[-1] - 1
            vis = A[:, :, last, base:base+n_img]                            # (NL, H, n_img)
            # Lu et al.'s modality attention ratio: share of this head's mass on visual tokens
            Sv = (vis.sum(-1) / A[:, :, last, :].sum(-1).clamp_min(1e-12))  # (NL, H)
            v = vis / vis.sum(-1, keepdim=True).clamp_min(1e-12)
            NL, H = Sv.shape
            gt = props[qid]["gt_box_frac"]
            store[f"m{n}"] = v.cpu().numpy().astype(np.float16)      # (NL, H, n_img)
            store[f"sv{n}"] = Sv.cpu().numpy().astype(np.float32)
            meta.append({"question_id_full": qid, "category": ex["category"], "grid": [gh, gw],
                         "n_img": int(n_img), "n_heads": int(H), "gt_box_frac": gt})
            n += 1
            if n % 20 == 0:
                el = time.time()-t0
                print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
            del A; torch.cuda.empty_cache()
    np.savez_compressed(OUT, **store)
    json.dump(meta, open(OUT.replace(".npz", "_meta.json"), "w"))
    print(f"Done. {n} items -> {OUT}", flush=True)


main()
