"""
Phase 44: is the layer block [0.55L, 0.95L] actually right for Qwen2-VL, or just right for Qwen3-VL?

THE ASSUMPTION BEING CHECKED
----------------------------
The relative block [0.55L, 0.95L] was reverse-engineered to reproduce Qwen3-VL's L16-26 of 28 --
the block a layer sweep picked out there. §4.3 is explicit that this is a REGIME, not a depth trend:
L0-15 give gt_pct 0.153-0.398, L16-26 give 0.027-0.119, and L27 collapses back to 0.408. A curve
shaped like that is exactly what you cannot safely rescale to another architecture, yet Phase 41,
42 and 43 all carry it over and call it "architecture-agnostic". The existing invariant test checks
the arithmetic of the rescaling, not the assumption behind it.

THE MEASUREMENT
---------------
Sweep EVERY layer of Qwen2-VL-7B on the same primary metric §4.3 used: gt_pct, the rank of the
GT-centre cell as a fraction of n_img, whose chance value is 0.500 EXACTLY by construction. Lower is
better. Secondary: P(argmax inside GT), chance = GT area share.

    Qwen2-VL's best regime falls near [0.55L, 0.95L]  -> the transfer in §42 is VALIDATED, and its
                                                         +7.9pp gate / +19.9pp attn-rand stand as
                                                         measured.
    it falls elsewhere                                -> §42's numbers are a LOWER BOUND under a
                                                         transferred hyper-parameter, must be
                                                         labelled as such, and likely explain why
                                                         attn-rand is +19.9pp here against Qwen3-VL's
                                                         +40.0pp.

Either way this is reported; the point is to stop assuming. No accuracy arm is involved, so nothing
here can be tuned to flatter the method.
"""
import json
import os
import statistics as st

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_CACHE", "/media/kavinder/hdd2/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUT = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase44_qwen2vl_layersweep.json"
B0 = 300
N_ITEMS = 120


def main():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    root = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench")["test"]

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval()
    pr = AutoProcessor.from_pretrained(MODEL_ID)
    itid = model.config.image_token_id
    L = model.config.get_text_config().num_hidden_layers
    MS = getattr(pr.image_processor, "merge_size", 2)
    print(f"{MODEL_ID}: {L} layers, merge {MS}", flush=True)

    def build(img, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = pr.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return pr(images=img, text=chat, return_tensors="pt")

    def measure(img):
        return int(sum(g[1] * g[2] // (MS * MS) for g in build(img, "x")["image_grid_thw"].tolist()))

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

    per_layer = {li: {"gt_pct": [], "in_gt": []} for li in range(L)}
    n = 0
    for ex in ds:
        if n >= N_ITEMS:
            break
        ip = os.path.join(root, ex["image"])
        ap = os.path.splitext(ip)[0] + ".json"
        if not os.path.exists(ap):
            continue
        ann = json.load(open(ap))
        if not ann.get("bbox"):
            continue
        img = Image.open(ip).convert("RGB")
        W_, H_ = img.size
        gx0 = min(b[0] for b in ann["bbox"]) / W_; gy0 = min(b[1] for b in ann["bbox"]) / H_
        gx1 = max(b[0] + b[2] for b in ann["bbox"]) / W_; gy1 = max(b[1] + b[3] for b in ann["bbox"]) / H_
        try:
            small = fit(img, B0)
            inp = build(small, ex["text"])
            g = inp["image_grid_thw"][0].tolist()
            gh, gw = g[1] // MS, g[2] // MS
            n_img = gh * gw
            pos = (inp["input_ids"][0] == itid).nonzero().flatten()
            base = int(pos[0].item())
            inp = inp.to(model.device)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            assert torch.isfinite(out.attentions[0]).all(), "non-finite attention"
            cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            gr, gc = min(gh - 1, int(cy * gh)), min(gw - 1, int(cx * gw))
            gi = gr * gw + gc
            for li in range(L):
                a = out.attentions[li][0, :, -1, base:base + n_img].float().mean(0)
                m = a.reshape(gh, gw).clone()
                mm = torch.full_like(m, -1.0)
                if gh > 2 and gw > 2:
                    mm[1:-1, 1:-1] = m[1:-1, 1:-1]      # same ring mask as the deployed localizer
                else:
                    mm = m.clone()
                f = mm.flatten()
                per_layer[li]["gt_pct"].append(float((f > f[gi]).sum()) / n_img)
                j = int(f.argmax().item())
                px, py = ((j % gw) + .5) / gw, ((j // gw) + .5) / gh
                per_layer[li]["in_gt"].append(1.0 * (gx0 <= px <= gx1 and gy0 <= py <= gy1))
            del out
            torch.cuda.empty_cache()
            n += 1
            if n % 20 == 0:
                print(f"  [{n}]", flush=True)
        except Exception as e:
            print(f"  skip {type(e).__name__}: {str(e)[:60]}")
            torch.cuda.empty_cache()

    res = {str(li): {"gt_pct": st.median(v["gt_pct"]), "in_gt": st.mean(v["in_gt"])}
           for li, v in per_layer.items() if v["gt_pct"]}
    json.dump({"model": MODEL_ID, "n": n, "layers": res}, open(OUT, "w"), indent=1)

    print(f"\nn = {n} items.  PRIMARY: median gt_pct, chance 0.500 EXACTLY, LOWER is better.")
    print(f"  {'layer':<8}{'med gt_pct':>12}{'argmax-in-GT':>15}")
    for li in range(L):
        if str(li) not in res:
            continue
        r = res[str(li)]
        mark = "  <- transferred block" if int(0.55 * L) <= li <= int(0.95 * L) else ""
        print(f"  L{li:<7}{r['gt_pct']:>12.3f}{100*r['in_gt']:>14.1f}%{mark}")

    order = sorted(res, key=lambda k: res[k]["gt_pct"])
    best = [int(k) for k in order[:8]]
    tb = list(range(int(0.55 * L), int(0.95 * L) + 1))
    inblock = sum(1 for b in best if b in tb)
    print(f"\n  best 8 layers by gt_pct: {sorted(best)}")
    print(f"  transferred block: L{tb[0]}-L{tb[-1]}")
    print(f"  {inblock}/8 of the best layers fall inside it")
    bb = st.median([res[str(li)]["gt_pct"] for li in tb if str(li) in res])
    ob = st.median([res[str(li)]["gt_pct"] for li in range(L)
                    if li not in tb and str(li) in res])
    print(f"  median gt_pct inside block {bb:.3f}   outside {ob:.3f}")
    print("\n  " + ("=> VALIDATED: the transferred block sits on this model's own best regime."
                    if inblock >= 5 else
                    "=> NOT VALIDATED: Qwen2-VL's best regime is elsewhere. Phase 42's numbers are a\n"
                    "     LOWER BOUND under a transferred hyper-parameter and must be labelled so."))


if __name__ == "__main__":
    main()
