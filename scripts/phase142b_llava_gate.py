"""Phase 142b: apply the divergence gate (>=0.5*max, the pre-registered constant) to LLaVA's stored raw
per-layer maps and evaluate top-1 coverage at W=0.25 against the deployed mean-block and raw-max rules."""
import json, sys, numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase140_trainfree_ladder as L
DIV = json.load(open(f"{L.D}/phase142_llava_divergence.json"))
for tag, label in [("llavanext", "LLaVA-NeXT-7B"), ("onevision", "LLaVA-OneVision-7B")]:
    if tag not in DIV: print(label, "no divergence curve"); continue
    items, (b0, b1), _ = L.load_llava(tag); div = np.array(DIV[tag]); n = len(items)
    print(f"\n=== {label} divergence: " + " ".join(f"L{i}:{v:.3f}" for i, v in enumerate(div)))
    rng = np.random.default_rng(142)
    def hits(maps): return np.array([float(it["cov"][int(np.argmax(np.where(it["rm"], m, -1e9)))] >= L.COV) for it, m in zip(items, maps)])
    def ci(d):
        b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)]); return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
    dep = hits([it["raw"][b0:b1].mean(0) for it in items]); rmx = hits([it["raw"][b0:b1].max(0) for it in items])
    print(f"  deployed mean block {dep.mean()*100:.1f}%   raw max block {rmx.mean()*100:.1f}%")
    for g in [0.3, 0.5, 0.7]:
        gate = div >= g*div.max(); ls = np.where(gate)[0]
        for agg, nm in [(lambda A: A[gate].max(0), "div-gated max"), (lambda A: A[gate].mean(0), "div-gated mean")]:
            h = hits([agg(it["raw"]) for it in items]); m, lo, hi = ci(h-dep)
            print(f"  gate {g} layers {ls.tolist()}  {nm:>14} {h.mean()*100:5.1f}%  vs deployed {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{'*' if lo > 0 else ''}")
