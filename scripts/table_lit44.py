"""
The literature 4x4: 4 models x the 4 benchmarks the attention-cropping literature evaluates on
(V*Bench, HR-Bench 4K, GQA, TextVQA -- the set ViRGo uses; CV-Bench/RealworldQA stay as the
adverse-condition evidence, not here).

Cell = DWA (transferred ridge) - block (published read-out), cost-fair: both arms pay one localise
pass and one crop at the same budget. Stars = 95% CI excludes zero (paired item bootstrap).

GQA is a deterministic 2000-item subsample of testdev_balanced (12578); the other three are full
splits. TextVQA validation (5000). Nothing is type-filtered anywhere.
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks as B
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)
MODELS = ["qwen3_2b", "qwen2_7b", "internvl3_8b", "llava_ov"]
NAMES = {"qwen3_2b": "Qwen3-VL-2B", "qwen2_7b": "Qwen2-VL-7B", "internvl3_8b": "InternVL3-8B", "llava_ov": "LLaVA-OV-7B"}
BENCH = [("vstar", "V*Bench"), ("hr4k", "HR-Bench 4K"), ("gqa", "GQA"), ("textvqa", "TextVQA"), ("docvqa", "DocVQA")]


def load(m, b):
    f = f"{D}/data/phase225_{m}_{b}.jsonl"
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def has(rows, a):
    return [r for r in rows if a in r.get("probs", {}) or a in r.get("preds", {})]


def acc(rows, a):
    v = [B.item_correct(r, a) for r in has(rows, a)]
    return 100 * float(np.mean(v)) if v else float("nan")


def delta(rows, a, b, Bn=10000):
    d = [B.item_correct(r, a) - B.item_correct(r, b) for r in rows
         if (a in r.get("probs", {}) or a in r.get("preds", {})) and (b in r.get("probs", {}) or b in r.get("preds", {}))]
    if len(d) < 8:
        return None
    d = np.array(d); m = d[rng.integers(0, len(d), (Bn, len(d)))].mean(1)
    lo, hi = np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100
    return d.mean() * 100, lo, hi, len(d)


print("=" * 100)
print("DWA - block (cost-fair placement), 4 models x 4 literature benchmarks")
print("=" * 100)
print(f"{'model':15s}" + "".join(f"{nm:>22s}" for _, nm in BENCH))
for m in MODELS:
    row = f"{NAMES[m]:15s}"
    for b, nm in BENCH:
        rows = load(m, b)
        r = delta(rows, "dwa_t", "block") if rows else None
        row += f"{('%.1f [%.1f,%.1f]' % (r[0], r[1], r[2])):>22s}" if r else f"{'-- running --':>22s}"
    print(row)
print()
print()
print("=" * 100)
print("DWA - uniform@600 (method vs the equal-compute bar)")
print("=" * 100)
print(f"{'model':15s}" + "".join(f"{nm:>22s}" for _, nm in BENCH))
for m in MODELS:
    row = f"{NAMES[m]:15s}"
    for b, nm in BENCH:
        rows = load(m, b)
        r = delta(rows, "dwa_t", "uniform@lo") if rows else None
        row += f"{('%.1f [%.1f,%.1f]' % (r[0], r[1], r[2])):>22s}" if r else f"{'-- running --':>22s}"
    print(row)
print()
print("=" * 100)
print("accuracy levels (bar / block / DWA), same cells")
print("=" * 100)
for b, nm in BENCH:
    print(f"-- {nm}")
    for m in MODELS:
        rows = load(m, b)
        if not rows:
            print(f"   {NAMES[m]:15s} -- running --"); continue
        print(f"   {NAMES[m]:15s} n={len(rows):5d}  bar {acc(rows,'uniform@lo'):5.1f}  block {acc(rows,'block'):5.1f}  DWA {acc(rows,'dwa_t'):5.1f}")
