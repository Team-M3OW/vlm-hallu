"""
Phase 109: REPLICATING Orgad et al., "LLMs Know More Than They Show" (ICLR 2025) in VLMs.

Their central, most-cited claim is about WHERE you probe, not what you probe with:

  "truthfulness information is concentrated in the exact answer tokens" -- e.g. "Hartford" in
  "The capital of Connecticut is Hartford" -- and "recognizing this nuance significantly improves
  error detection". Probing the last generated token, or a mean, misses it.

Phase 107 probed ATTENTION STATISTICS and failed (1 of 2, ordering reversed). That is not their
method. The VLM analogue of an exact answer token is the **exact evidence token**: the visual tokens
whose patch overlaps the annotated evidence region. This probes there.

POSITIONS (per layer, all 28)
    last      hidden state at the final prompt position     <- what everyone does
    evid      MEAN over visual tokens overlapping the GT box  <- Orgad's move, ported
    evid_max  MAX-pooled over the same tokens
    rand      mean over a size-matched random region        <- the control phase 15 showed is
                                                               essential: without it the probe can
                                                               read "annotated object" vs "random
                                                               rectangle" and look spectacular
    imgmean   mean over ALL visual tokens

TARGETS  err (pass-1 wrong) and encfail (pass-1 wrong AND oracle crop right).

THEIR SECOND CLAIM, also tested: detectors are "skill-specific" and do not generalise across
datasets. The analysis step stratifies by tokens-on-target, which is the VLM version of a skill
boundary -- and our own prediction is sharper than theirs: probes should work ABOVE the encoding
cliff (their regime, where the evidence is in the tokens) and collapse BELOW it (phase 66: on those
items the answer is decodable at NO layer).

NOTE the evid/evid_max/rand positions use the GT box. They are therefore DIAGNOSTIC, not deployable,
and must never be reported as a detector -- only as a test of whether the information is present.
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
OUT = f"{D}/phase109_probe_{WHICH}.npz"
B0 = 300
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

    POSN = ["last", "evid", "evid_max", "rand", "imgmean"]
    feats, ids, tot = [], [], []
    rng = np.random.default_rng(109)
    t0, n = time.time(), 0
    for ex in ds:
        qid = f"{ex['category']}/{ex['question_id']}" if "question_id" in ex else ex.get("question_id_full")
        if qid not in props: continue
        ip = os.path.join(root, ex["image"])
        if not os.path.exists(ip): continue
        gt = props[qid]["gt_box_frac"]
        img, _ = fit(Image.open(ip).convert("RGB"), B0)
        inp = build(img, ex["text"])
        g = inp["image_grid_thw"][0].tolist()
        gh, gw = g[1]//2, g[2]//2
        n_img = gh*gw
        pos = (inp["input_ids"][0] == img_tok_id).nonzero().flatten()
        base = int(pos[0].item())
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        Hs = torch.stack([h[0].float() for h in out.hidden_states[1:]])   # (NL, n, d)
        del out
        yy, xx = np.mgrid[0:gh, 0:gw]
        fy, fx = ((yy+.5)/gh).ravel(), ((xx+.5)/gw).ravel()
        inbox = ((fx >= gt[0]) & (fx <= gt[2]) & (fy >= gt[1]) & (fy <= gt[3]))
        if inbox.sum() == 0:                      # sub-cell target: take the nearest cell
            d = (fx-(gt[0]+gt[2])/2)**2 + (fy-(gt[1]+gt[3])/2)**2
            inbox = np.zeros(n_img, bool); inbox[int(d.argmin())] = True
        k = int(inbox.sum())
        ridx = rng.choice(n_img, size=k, replace=False)
        V = Hs[:, base:base+n_img, :]
        row = {
            "last":     Hs[:, -1, :],
            "evid":     V[:, inbox, :].mean(1),
            "evid_max": V[:, inbox, :].max(1).values,
            "rand":     V[:, ridx, :].mean(1),
            "imgmean":  V.mean(1),
        }
        feats.append(np.stack([row[p].cpu().numpy() for p in POSN]).astype(np.float16))
        ids.append(qid)
        tot.append(float(((gt[2]-gt[0])*(gt[3]-gt[1])) * n_img))   # tokens on target
        n += 1
        if n % 20 == 0:
            el = time.time()-t0
            print(f"  [{n}] {n/el:.2f} it/s eta={(191-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
        del Hs, V; torch.cuda.empty_cache()
    np.savez_compressed(OUT, X=np.stack(feats), ids=np.array(ids),
                        tokens_on_target=np.array(tot), positions=np.array(POSN))
    print(f"Done. {n} items, X={np.stack(feats).shape} -> {OUT}", flush=True)


main()
