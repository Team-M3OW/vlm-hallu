"""Phase 242 (CPU-only): five cliff analyses for the small-details section.

Old-1  does the cliff move with budget?  Sliding threshold on target extent at B=300 vs B~588,
       single-region items, two models; threshold in merged tokens and in image-side fraction.
Old-2  the budget formula.  Collapse test: does accuracy depend on tokens-on-target (S*B) rather
       than on S or B separately?  Table of accuracy vs tokens-on-target at the two budgets.
New-1  required magnification law.  For each item, the smallest oracle window W that answers it,
       against target size: does W* track sqrt(S)?
New-3  localisation invariance.  Coverage of the ground-truth box below vs above the cliff.
New-6  patch dilution.  Target side in pixels vs the encoder patch / merged-token footprint; where
       the cliff sits in that arithmetic.

Data: phase78_w_sweep.jsonl + phase30c maps (Qwen3), phase97m_merged_qwen2vl.jsonl + phase74 maps
(Qwen2).  No model runs."""
import json, numpy as np
from scipy.stats import spearmanr
D = "."
rng = np.random.default_rng(242)


def load(f): return [json.loads(l) for l in open(f)]


q3 = load("data/phase78_w_sweep.jsonl"); q2 = load("data/phase97m_merged_qwen2vl.jsonl")
m3 = {r["question_id_full"]: r for r in load("data/phase30c_attn_maps_all.jsonl")}
m2 = {r["question_id_full"]: r for r in load("data/phase74_Qwen2_VL_7B_Instruct.jsonl")}
for rows, maps in ((q3, m3), (q2, m2)):
    for r in rows:
        if r.get("gt_area_frac") is None:
            x0, y0, x1, y1 = maps[r["question_id_full"]]["gt_box_frac"]
            r["gt_area_frac"] = (x1 - x0) * (y1 - y0)
        r["grid"] = maps[r["question_id_full"]]["grid"]
        r["img_wh"] = maps[r["question_id_full"]].get("img_wh")


def correct(r, arm):
    p = r["probs"].get(arm)
    return None if p is None else float(int(np.argmax(p)) == r["label"])


def cliff(rows, arm, B, min_side=10, grid=np.linspace(0.05, 2.0, 40)):
    xs = np.array([r["gt_area_frac"] * B for r in rows])
    ys = np.array([correct(r, arm) for r in rows], float)
    best = None
    for t in grid:
        lo = xs < t; hi = ~lo
        if lo.sum() < min_side or hi.sum() < min_side: continue
        step = ys[hi].mean() - ys[lo].mean()
        if best is None or step > best[0]: best = (step, t, ys[lo].mean(), ys[hi].mean(), int(lo.sum()), int(hi.sum()))
    return best


print("=" * 100)
print("OLD-1  cliff vs budget (single-region items, category=direct_attributes)")
for name, rows in (("qwen3", q3), ("qwen2", q2)):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    for arm, B in (("uniform@300", 300.0), ("uniform@600", rows[0]["realized_tokens"]["uniform@600"])):
        step, t, ab, aa, nb, na = cliff(sr, arm, B)
        side = np.sqrt(t / B) * 100
        print(f"  {name:6s} {arm:12s} B={B:5.0f}  t*={t:.3f} tok  below {ab*100:5.1f}% (n={nb:3d})  above {aa*100:5.1f}% (n={na:3d})  step {step*100:+5.1f}  |  t*/B={t/B*100:.4f}% area = {side:.2f}% side")

print("=" * 100)
print("OLD-2  collapse test: accuracy vs tokens-on-target (S*B), by budget")
bins = [(0, 0.15), (0.15, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
for name, rows in (("qwen3", q3), ("qwen2", q2)):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    print(f"  {name}:")
    print(f"    {'ext bin':14s} {'n@300':>6s} {'acc@300':>8s} {'n@588':>6s} {'acc@588':>8s}")
    for lo, hi in bins:
        e3 = np.array([r["gt_area_frac"] * 300 for r in sr]); e6 = np.array([r["gt_area_frac"] * sr[0]["realized_tokens"]["uniform@600"] for r in sr])
        y3 = np.array([correct(r, "uniform@300") for r in sr], float); y6 = np.array([correct(r, "uniform@600") for r in sr], float)
        m3_ = (e3 >= lo) & (e3 < hi); m6_ = (e6 >= lo) & (e6 < hi)
        s3 = f"{y3[m3_].mean()*100:7.1f}" if m3_.sum() >= 5 else "      -"
        s6 = f"{y6[m6_].mean()*100:7.1f}" if m6_.sum() >= 5 else "      -"
        print(f"    [{lo:4.2f},{hi if hi<1e8 else 9:4.2f}) {m3_.sum():6d} {s3:>8s} {m6_.sum():6d} {s6:>8s}")

print("=" * 100)
print("NEW-1  required magnification: smallest oracle W that answers, vs target size")
Ws = [0.15, 0.25, 0.35, 0.5, 0.7]
for name, rows in (("qwen3", q3), ("qwen2", q2)):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    Wstar = []
    for r in sr:
        w = None
        for W in Ws:
            v = correct(r, f"oracle@{W}")
            if v == 1.0: w = W; break
        Wstar.append(w if w is not None else 1.0)  # 1.0 = unresolved
    S = np.array([r["gt_area_frac"] for r in sr])
    rho, p = spearmanr(Wstar, np.sqrt(S))
    print(f"  {name}: unresolved by W=0.7: {sum(1 for w in Wstar if w==1.0)}/{len(Wstar)}   Spearman(W*, sqrt(S)) = {rho:+.3f} (p={p:.2g})")
    qs = np.quantile(S, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (S >= qs[i]) & (S <= qs[i + 1])
        w = np.array(Wstar)[m]
        print(f"    S in [{qs[i]:.5f},{qs[i+1]:.5f}]  n={m.sum():3d}  median W*={np.median(w):.2f}  resolved={np.mean(w<1.0)*100:4.0f}%")

print("=" * 100)
print("NEW-3  localisation invariance: coverage below vs above the cliff (t*=0.25 tok at B=300)")
for name, rows in (("qwen3", q3), ("qwen2", q2)):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    xs = np.array([r["gt_area_frac"] * 300 for r in sr])
    for field in ("argmax_cov", "head_cov"):
        v = np.array([r.get(field, np.nan) for r in sr], float)
        lo = xs < 0.25; hi = ~lo
        def ci(x):
            m = x[~np.isnan(x)]
            if len(m) < 5: return (np.nan, np.nan, np.nan)
            b = m[rng.integers(0, len(m), (4000, len(m)))].mean(1)
            return m.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
        ml, ll, hl = ci(v[lo]); mh, lh, hh = ci(v[hi])
        print(f"  {name} {field:10s} below {ml:.3f} [{ll:.3f},{hl:.3f}]  above {mh:.3f} [{lh:.3f},{hh:.3f}]")

print("=" * 100)
print("NEW-6  patch dilution: target side vs encoder footprint (300-token pass)")
for name, rows in (("qwen3", q3), ("qwen2", q2)):
    sr = [r for r in rows if r["category"] == "direct_attributes"]
    ratios, tok_targets = [], []
    for r in sr:
        gh, gw = r["grid"]
        iw, ih = r["img_wh"] if r["img_wh"] else m3[r["question_id_full"]]["img_wh"]
        S = r["gt_area_frac"]
        target_side = np.sqrt(S * iw * ih)
        patch_side = iw / gw
        merged_side = 2 * patch_side
        ratios.append(target_side / merged_side)
        tok_targets.append(S * gh * gw)
    ratios = np.array(ratios); tok_targets = np.array(tok_targets)
    print(f"  {name}: median target side = {np.median(ratios):.3f} merged tokens;  target < 1 merged token on {np.mean(ratios<1)*100:.0f}% of items")
    print(f"         median tokens-on-target = {np.median(tok_targets):.3f};  below the 0.25-token cliff: {np.mean(tok_targets<0.25)*100:.0f}%")
