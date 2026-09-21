"""
Phase 170: dump PER-HEAD attention maps once, so head-selection criteria can be crossed with the
re-ranking head off-line.

Phase 108 stored only per-RULE aggregates (allheads / top-q by S_v / CoRe-contrastive) plus the
per-head scalars S_v and S_core. Testing a new head criterion therefore needed a new GPU pass every
time. This stores the raw object: for every item, the final token's attention over the image cells
at every (layer, head), L1-normalised per head.

    out: (n_items, 28, H, n_cells) float16   +  an index json
    Qwen3-VL-2B  H=16 ->  51 MB      Qwen2-VL-7B  H=28 ->  90 MB

PROMPT CONVENTION -- load-bearing, do not change. `ex["text"]` is V0: question + options +
answer instruction, so the final token IS the answer-emission point (SS16A / phase 121). VRH
(arXiv 2608.27417) finds that scoring heads from OUTPUT query tokens rather than input query tokens
is the single most important design choice in head selection; reading the final token under V0 is
that convention already, so a criterion computed from this dump tests their rule and not a degraded
version of it.

Everything else (bf16 weights, eager attention, fit-to-300 by measured image_grid_thw) is phase 108's
machinery verbatim, so the maps are comparable to what is already on disk.
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
CFG = {"qwen3": ("Qwen/Qwen3-VL-2B-Instruct", f"{D}/phase71a_head_proposals.json"),
       "qwen2": ("Qwen/Qwen2-VL-7B-Instruct", f"{D}/phase80a_qwen2vl_proposals.json")}
MODEL_ID, PROP = CFG[WHICH]
OUT = f"{D}/phase170_perhead_{WHICH}"
B0 = 300
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

    maps, index, t0 = [], [], time.time()
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
        A = torch.stack([a[0].float() for a in out.attentions])          # (NL, H, n, n)
        del out
        last = A.shape[-1] - 1
        vis = A[:, :, last, base:base+n_img]                             # (NL, H, n_img)
        v = vis / vis.sum(-1, keepdim=True).clamp_min(1e-12)
        maps.append(v.half().cpu().numpy())
        index.append({"question_id_full": qid, "category": ex["category"],
                      "grid": [gh, gw], "n_img": n_img})
        del A, vis, v
        torch.cuda.empty_cache()
        if len(maps) % 25 == 0:
            print(f"  [{len(maps)}] {len(maps)/(time.time()-t0):.2f} it/s", flush=True)

    # Grids are RAGGED: fit-to-300 lands each image on its own grid (286-315 cells observed), the
    # same per-item `grid` phase 70 already carries. Store one (NL*H, sum n_i) buffer plus per-item
    # column offsets rather than a padded cube -- no pickle, no padding, exact reconstruction.
    NL, H = maps[0].shape[0], maps[0].shape[1]
    assert all(m.shape[:2] == (NL, H) for m in maps), "layer/head count is not constant"
    off, cur = [], 0
    for m, e in zip(maps, index):
        e["offset"], e["n_cells"] = cur, m.shape[2]
        cur += m.shape[2]
    buf = np.concatenate([m.reshape(NL*H, m.shape[2]) for m in maps], axis=1)
    np.save(f"{OUT}.npy", buf)
    json.dump({"n_layers": NL, "n_heads": H, "items": index}, open(f"{OUT}_index.json", "w"))
    print(f"\nwrote {OUT}.npy  {buf.shape} {buf.dtype}  {buf.nbytes/1e6:.0f} MB  "
          f"({len(index)} items, {NL} layers x {H} heads, cells {min(e['n_cells'] for e in index)}"
          f"-{max(e['n_cells'] for e in index)})")


if __name__ == "__main__":
    main()
