"""
Phase 100: Qwen2-VL pass-1 attention features -- the input the frozen sizer needs.

No GPU. data/phase74_Qwen2_VL_7B_Instruct.jsonl already stores all 28 layers of last-token attention
over the image tokens for all 191 V*Bench items, so the six features in PREREG_DCR_SIZER.md can be
computed off disk.

The definitions are transcribed from phase32_conditional_allocator.py:141-159 -- per-layer
sum-normalised, averaged over the deployed block, ring-masked for `peak` and `peak_over_median`
while the distributional features use the unmasked map. Any drift here would silently change the
frozen configuration, so this file must not "improve" them.

BLOCK rescaled to the same fraction of the stack as Qwen3-VL's L16-26, exactly as phase 80a does:
list(range(int(.57*28), int(.93*28)+1)) = L15..L26.
"""
import json, math
import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
SRC, OUT = f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", f"{D}/phase100_qwen2vl_feats.json"
NL = 28
BLOCK = list(range(int(.57 * NL), int(.93 * NL) + 1))

out = {}
for line in open(SRC):
    r = json.loads(line)
    gh, gw = r["grid"]
    n_img = r["n_img_tokens"]
    A = np.stack([np.asarray(r["attn"][f"L{i}"], dtype=float) for i in range(NL)])
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    acc = A[BLOCK].mean(0).reshape(gh, gw)

    masked = np.full_like(acc, -1.0)
    if gh > 2 and gw > 2:
        masked[1:-1, 1:-1] = acc[1:-1, 1:-1]
    else:
        masked = acc.copy()

    flat = acc.flatten()
    srt = np.sort(flat)[::-1]
    s = flat.sum() + 1e-12
    pn = flat / s
    ent = float(-(pn * np.log(pn + 1e-12)).sum())
    out[r["question_id_full"]] = {
        "peak": float(masked.max()),
        "peak_over_median": float(masked.max() / (np.median(flat) + 1e-12)),
        "top1_frac": float(srt[0] / s),
        "top5_frac": float(srt[:5].sum() / s),
        "entropy": ent,
        "entropy_norm": ent / math.log(float(n_img)),
        "n_img": n_img,
    }
json.dump(out, open(OUT, "w"))
k = list(out)[0]
print(f"wrote {len(out)} items -> {OUT}")
print(f"  example {k}: " + "  ".join(f"{a}={out[k][a]:.4g}" for a in
      ["peak", "peak_over_median", "top1_frac", "top5_frac", "entropy_norm"]))
vals = {a: [v[a] for v in out.values()] for a in ["peak", "top1_frac", "entropy_norm"]}
for a, v in vals.items():
    print(f"  {a:>13}: min {min(v):.4g}  median {np.median(v):.4g}  max {max(v):.4g}")
