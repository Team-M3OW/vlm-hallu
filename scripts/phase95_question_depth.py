"""
Phase 95: WHERE DOES ATTENTION START DEPENDING ON THE QUESTION? -- a label-free depth locator.

TWO JOBS IN ONE EXPERIMENT
--------------------------
(1) MECHANISM. Phase 87 found a threshold: pruning by attention read anywhere in L0-L14 is
    catastrophic (-15 to -24pp, at or below random) while L16+ is safe, with a +13.1pp step between
    them. We measured it but cannot explain it. The obvious hypothesis: early layers cannot rank
    tokens by relevance because the attention there does not yet depend on WHAT WAS ASKED.

(2) A DEPLOYABLE TOOL. "Read at ~57% of depth" is useless to someone with a different model -- they
    would need ground-truth boxes to locate their own threshold. If question-dependence marks the
    same place, it can be measured with NO LABELS AT ALL: run one image with several different
    questions and find the layer where the attention maps start to diverge.

METHOD
    For each image, run the SAME image with K different questions (the item's own, plus questions
    borrowed from other items -- genuinely about different things). At each layer, measure how much
    the ring-masked attention maps differ across questions:
        divergence(L) = mean over question pairs of (1 - cosine similarity of their maps)
    Early layers should be near-identical across questions (divergence ~ 0). The layer where
    divergence rises is the predicted threshold.

VALIDATION, fixed before the run
    the divergence rise coincides with the pruning threshold (L14-L16 on Qwen3-VL)
        -> mechanism explained AND the locator is validated against a known answer
    divergence rises somewhere else
        -> question-conditioning is NOT what the threshold is about; report the dissociation and
           do not ship the locator

Run on Qwen3-VL-2B (where the pruning threshold is known) and Qwen2-VL-7B (where it is not, so the
locator makes a PREDICTION that phase 96 can then test).
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODELS = [("Qwen3-VL-2B", "Qwen/Qwen3-VL-2B-Instruct"),
          ("Qwen2-VL-7B", "Qwen/Qwen2-VL-7B-Instruct")]
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase95_question_depth.json"
B0, N_IMG, K_Q = 300, 40, 4
Image.MAX_IMAGE_PIXELS = None


def run(tag, mid, items, root):
    model = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(mid)
    itid = model.config.image_token_id
    MS = getattr(pr.image_processor, "merge_size", 2)
    nL = model.config.get_text_config().num_hidden_layers
    print(f"  {tag}: {nL} layers", flush=True)

    def build(img, text):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return pr(images=img, text=pr.apply_chat_template(m, tokenize=False,
                                                          add_generation_prompt=True),
                  return_tensors="pt")

    def measure(img):
        i = build(img, "x")
        return int(sum(g[1] * g[2] // (MS * MS) for g in i["image_grid_thw"].tolist()))

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
        return best[0]

    div = [[] for _ in range(nL)]
    t0 = 0
    for n, (ip, qs) in enumerate(items):
        img = fit(Image.open(ip).convert("RGB"), B0)
        maps = []
        for q in qs:
            inp = build(img, q)
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            if len(pos) < 16:
                maps = []
                break
            base, ntok = int(pos[0].item()), int(len(pos))
            with torch.no_grad():
                o = model(**inp.to(model.device), output_attentions=True)
            A = np.stack([o.attentions[L][0, :, -1, base:base + ntok].float().mean(0).cpu().numpy()
                          for L in range(nL)])
            del o
            torch.cuda.empty_cache()
            maps.append(A / np.maximum(A.sum(1, keepdims=True), 1e-12))
        if len(maps) < 2:
            continue
        m = min(x.shape[1] for x in maps)
        for L in range(nL):
            ds = []
            for i in range(len(maps)):
                for j in range(i + 1, len(maps)):
                    a, b = maps[i][L][:m], maps[j][L][:m]
                    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    ds.append(1.0 - c)
            div[L].append(float(np.mean(ds)))
        if (n + 1) % 10 == 0:
            print(f"    [{n+1}/{len(items)}]", flush=True)
    del model
    torch.cuda.empty_cache()
    return [float(np.mean(d)) if d else float("nan") for d in div]


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]
    rows = [e for e in ds if os.path.exists(os.path.join(root, e["image"]))]
    qs_all = [e["text"] for e in rows]
    items = []
    for i, e in enumerate(rows[:N_IMG]):
        # the item's own question, plus questions borrowed from OTHER items -- genuinely about
        # different things, so any divergence is question-driven and not paraphrase noise
        qs = [e["text"]] + [qs_all[(i * 37 + j * 91) % len(qs_all)] for j in range(1, K_Q)]
        items.append((os.path.join(root, e["image"]), qs))
    print(f"{len(items)} images x {K_Q} questions each\n", flush=True)

    out = {}
    for tag, mid in MODELS:
        out[tag] = run(tag, mid, items, root)
        d = out[tag]
        nL = len(d)
        base = float(np.mean(d[:max(2, nL // 6)]))
        rises = [L for L in range(1, nL) if d[L] > 3 * base]
        print(f"\n  {tag}: divergence by layer")
        for L in range(nL):
            if L % 2 == 0 or (rises and L in (rises[0], rises[0] - 1)):
                mark = "  <-- rise" if rises and L == rises[0] else ""
                print(f"    L{L:<3d} {d[L]:.4f}{mark}")
        print(f"  {tag}: early baseline {base:.4f}, first layer above 3x baseline: "
              f"L{rises[0] if rises else '-'}  ({100*(rises[0]/nL):.0f}% of depth)" if rises
              else f"  {tag}: no rise detected")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print("\nPruning threshold for reference: Qwen3-VL L14->L16 (57% of depth).")


if __name__ == "__main__":
    main()
