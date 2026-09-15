"""
Phase 69: ADAPTIVE BUDGET FROM A NON-SIZE SIGNAL. Does anything free at pass 1 predict how many
tokens an item needs?

WHY THIS EXPERIMENT EXISTS
--------------------------
SS14A closed the size-based budget rule: with a PERFECT ground-truth area_fraction, the rule
b = t*/area_frac loses to flat uniform in 11 of 12 cells. But that same re-analysis measured real
headroom -- an oracle controller reaches the plateau's 84.8% at 2418 mean tokens instead of 7990,
a 3.3x saving -- and showed WHY size fails: the budget an item actually needs is nearly flat in
target size (3212/2146/2318/1615 tokens across a 20x range of area fraction).

So the family is not dead; the SIGNAL was wrong. This file asks whether any pass-1-free signal
predicts required budget. Precedent: `peak` predicts coverage at AUROC 0.788 for free (SS11A), so
"a cheap internal statistic carries information about a downstream quantity" is established here.

NO GPU. Every feature is read from artifacts already on disk:
    phase30c_attn_maps_all.jsonl   28 layers x 300 cells of image attention, per item
    phase60_logit_lens.jsonl       28 layers x 4 answer probabilities, per item
    phase27_exchange_results.jsonl the 7-rung uniform ladder -- supplies the LABEL
Joined on (category, index); the join is verified by requiring gt_area_frac to agree to 1e-9,
because the three files spell question ids differently (`direct_attributes/0` vs
`direct_attributes/sa_4690.jpg::0`) and a positional join would fail silently.

THE LABEL
---------
For each item, the smallest ladder rung from which it is correct AT THAT RUNG AND EVERY RUNG ABOVE.
Monotone stability is required because SS14A showed 22.5% of items flip correct->wrong going up the
ladder; regressing on "cheapest rung that happened to be right" would fit noise and inflate every
number downstream. Items never stably correct are labelled with the top rung (you cannot buy them).

THE BAR, AND WHY IT IS THE ONLY HONEST ONE
------------------------------------------
A budget controller emits a DISTRIBUTION of budgets, so the >10% budget-gate does not apply. Each
controller is scored against the uniform ladder interpolated at the controller's OWN mean token
spend. This is exactly the mismatch that once turned +3.9pp [-0.3,+8.2] into a spurious
"+5.2pp [+1.1,+9.4] SIGNIFICANT" (SS8), so it is enforced here rather than remembered.

CONTROLS, FIXED BEFORE THE RUN
------------------------------
    shuffled-label     the identical pipeline with y permuted. Must land ON the uniform ladder.
                       If it does not, the evaluation itself is leaking and no result is readable.
    constant rung      predicting one rung for everybody IS the uniform ladder, by construction.
    size-only          area_fraction as the single feature -- SS14A's refuted rule, re-run inside
                       this harness so the comparison is apples to apples.
All predictions are OUT-OF-FOLD (repeated stratified 5-fold, 20 repeats). Nothing is scored on an
item that was in its own training fold.

READING
-------
    curve clears the ladder by >2pp at matched tokens, shuffled control does not  -> SIGNAL EXISTS
    curve tracks the ladder within noise                                          -> the last member
        of the adaptive-budget family is closed, and SS14A + this file close it on measured
        evidence at both the oracle level and the achievable level.
"""
import json
import math
import os
import statistics as st

import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
RUNGS = ["uniform@150", "uniform@300", "uniform@600", "uniform@1200",
         "uniform@2400", "uniform@4800", "uniform@8000"]
OUT = os.path.join(D, "phase69_budget_predictor.json")
BLOCK = list(range(16, 27))          # the deployed read-out block


def ent(p):
    p = np.asarray(p, dtype=float)
    p = p / max(p.sum(), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def load():
    A = {r["question_id_full"]: r for r in map(json.loads,
         open(f"{D}/phase30c_attn_maps_all.jsonl"))}
    L = {r["question_id_full"]: r for r in map(json.loads, open(f"{D}/phase60_logit_lens.jsonl"))}
    P = [json.loads(l) for l in open(f"{D}/phase27_exchange_results.jsonl")]
    rows = []
    for p in P:
        qid = p["question_id_full"]
        cat, idx = qid.split("/")[0], qid.split("::")[-1]
        k = f"{cat}/{idx}"
        if k not in A or k not in L:
            continue
        a, l = A[k], L[k]
        # verify the join rather than trusting the id convention
        assert abs(a["gt_area_frac"] - p["bbox_area_frac"]) < 1e-9, f"join mismatch on {k}"
        assert abs(l["gt_area_frac"] - p["bbox_area_frac"]) < 1e-6, f"lens join mismatch on {k}"
        rows.append((p, a, l))
    assert len(rows) == len(P), f"joined {len(rows)}/{len(P)}"
    return rows


def features(p, a, l):
    gh, gw = a["grid"]
    n = a["n_img_tokens"]
    # ---- attention map, deployed block, ring-masked exactly as the proposer masks it
    acc = np.zeros(n)
    for Lx in BLOCK:
        v = np.asarray(a["attn"][f"L{Lx}"], dtype=float)
        acc += v / max(v.sum(), 1e-12)
    acc /= len(BLOCK)
    M = acc.reshape(gh, gw)
    ring = M.copy()
    if gh > 2 and gw > 2:
        inner = ring[1:-1, 1:-1].flatten()
    else:
        inner = ring.flatten()
    s = np.sort(inner)[::-1]
    tot = max(inner.sum(), 1e-12)
    # spatial spread of attention mass (concentrated vs diffuse)
    yy, xx = np.mgrid[0:gh, 0:gw]
    w = M.flatten() / max(M.sum(), 1e-12)
    cx, cy = float((xx.flatten() * w).sum()), float((yy.flatten() * w).sum())
    spread = float(np.sqrt(((xx.flatten() - cx) ** 2 + (yy.flatten() - cy) ** 2).dot(w)))
    # ---- logit lens under uniform: how settled is the answer, and how early
    U = np.asarray(l["uniform"], dtype=float)       # 28 layers x 4
    fin = U[-1]
    order = np.sort(fin)[::-1]
    fa = int(np.argmax(fin))
    agree = float(np.mean([int(np.argmax(U[i])) == fa for i in range(len(U))]))
    lateagree = float(np.mean([int(np.argmax(U[i])) == fa for i in range(len(U) - 8, len(U))]))
    first = next((i for i in range(len(U)) if all(int(np.argmax(U[j])) == fa
                                                  for j in range(i, len(U)))), len(U))
    W_, H_ = p["img_wh"]
    return {
        # --- pass-1 attention geometry (free)
        "peak": float(s[0]), "top5": float(s[:5].sum() / tot), "top20": float(s[:20].sum() / tot),
        "ent": ent(inner), "gini": gini(inner), "spread": spread,
        "peak_over_mean": float(s[0] / max(inner.mean(), 1e-12)),
        "ratio_1_5": float(s[0] / max(s[4], 1e-12)),
        # --- pass-1 answer state (free)
        "conf": float(order[0]), "margin": float(order[0] - order[1]), "aent": ent(fin),
        "lens_agree": agree, "lens_lateagree": lateagree, "lens_settle": float(first) / len(U),
        # --- free metadata
        "logpix": math.log10(W_ * H_), "aspect": float(max(W_, H_) / min(W_, H_)),
        "is_rel": 1.0 * (p["category"] == "relative_position"),
    }


def label_and_ladder(p):
    hit = [1 * (p["pred"][a] == p["label"]) for a in RUNGS]
    tk = [p["realized_tokens"][a] for a in RUNGS]
    k = next((i for i in range(len(hit)) if all(hit[i:])), None)
    return (k if k is not None else len(RUNGS) - 1), hit, tk


def ladder_at(mean_tok, HIT, TK):
    """Uniform-ladder accuracy interpolated at a given mean token spend -- the matched bar."""
    xs = [st.mean([t[i] for t in TK]) for i in range(len(RUNGS))]
    ys = [st.mean([h[i] for h in HIT]) for i in range(len(RUNGS))]
    if mean_tok <= xs[0]:
        return ys[0]
    if mean_tok >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= mean_tok <= xs[i + 1]:
            f = (mean_tok - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]


def controller(pred_rung, offset, HIT, TK):
    """Apply a global offset to the predicted rung; traces the cost-accuracy CURVE, not a point."""
    acc, cost = [], []
    for j, r in enumerate(pred_rung):
        k = int(min(max(r + offset, 0), len(RUNGS) - 1))
        acc.append(HIT[j][k]); cost.append(TK[j][k])
    return st.mean(acc), st.mean(cost)


def oof(X, y, seeds=20):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    P = np.zeros(len(y))
    for s in range(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=s)
            m.fit(sc.transform(X[tr]), y[tr])
            P[te] += m.predict(sc.transform(X[te]))
    return P / seeds


def main():
    rows = load()
    print(f"joined {len(rows)} items (gt_area_frac verified)\n")
    F = [features(p, a, l) for p, a, l in rows]
    names = list(F[0].keys())
    X = np.array([[f[k] for k in names] for f in F])
    Y, HIT, TK = [], [], []
    for p, _, _ in rows:
        k, h, t = label_and_ladder(p)
        Y.append(k); HIT.append(h); TK.append(t)
    Y = np.array(Y)
    print("required-rung label distribution:",
          {RUNGS[i].split('@')[1]: int((Y == i).sum()) for i in range(len(RUNGS))})
    plateau = st.mean([h[-1] for h in HIT])
    orc_a = st.mean([1.0 * HIT[j][Y[j]] for j in range(len(Y))])
    orc_c = st.mean([TK[j][Y[j]] for j in range(len(Y))])
    print(f"oracle controller {100*orc_a:.1f}% @ {orc_c:.0f} tok | "
          f"plateau {100*plateau:.1f}% @ {st.mean([t[-1] for t in TK]):.0f} tok\n")

    # ---- univariate screen: does ANY single free feature correlate with required budget?
    print("univariate |Spearman rho| with required rung:")
    from scipy.stats import spearmanr
    rs = sorted(((abs(spearmanr(X[:, i], Y).statistic), names[i]) for i in range(len(names))),
                reverse=True)
    for r, nm in rs[:8]:
        print(f"   {nm:16} {r:.3f}")
    print(f"   [size-only, for reference] area_frac "
          f"{abs(spearmanr([p['bbox_area_frac'] for p,_,_ in rows], Y).statistic):.3f}")

    res = {"n": len(Y), "plateau_acc": plateau, "oracle_acc": orc_a, "oracle_tok": orc_c,
           "univariate": {nm: float(r) for r, nm in rs}, "curves": {}}

    rng = np.random.default_rng(69)
    variants = {
        "internals (all free features)": X,
        "size only (SS14A's refuted rule)": np.array(
            [[p["bbox_area_frac"]] for p, _, _ in rows]),
        "shuffled labels (CONTROL)": X,
    }
    for nm, Xi in variants.items():
        yy = rng.permutation(Y) if nm.startswith("shuffled") else Y
        pr = oof(Xi, yy)
        print(f"\n=== {nm} ===")
        print(f"  {'offset':>7}{'acc':>8}{'mean tok':>11}{'ladder@tok':>12}{'delta':>9}")
        curve = []
        for off in range(-2, 5):
            a, c = controller(np.rint(pr).astype(int), off, HIT, TK)
            bar = ladder_at(c, HIT, TK)
            curve.append({"offset": off, "acc": a, "tok": c, "bar": bar, "delta": a - bar})
            print(f"  {off:+7d}{100*a:7.1f}%{c:11.0f}{100*bar:11.1f}%{100*(a-bar):+8.1f}pp")
        res["curves"][nm] = curve
        best = max(curve, key=lambda d: d["delta"])
        print(f"  best delta vs matched ladder: {100*best['delta']:+.1f}pp "
              f"at {best['tok']:.0f} tok")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

    print("\n" + "=" * 70)
    ci = max(res["curves"]["internals (all free features)"], key=lambda d: d["delta"])
    cs = max(res["curves"]["shuffled labels (CONTROL)"], key=lambda d: d["delta"])
    print(f"VERDICT  internals {100*ci['delta']:+.1f}pp | shuffled control {100*cs['delta']:+.1f}pp")
    if ci["delta"] > 0.02 and ci["delta"] - cs["delta"] > 0.02:
        print("  => SIGNAL EXISTS. A free pass-1 statistic predicts required budget.")
    else:
        print("  => NO SIGNAL. The adaptive-budget family is closed at the achievable level too,")
        print("     alongside SS14A closing it at the oracle-size level.")


if __name__ == "__main__":
    main()
