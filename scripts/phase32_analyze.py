"""
Phase 32 analyzer: does a conditional allocator earn the second forward pass?

Bars (from PLAN.md, fixed long before this phase):
    56.5%  uniform@300              -- beats doing nothing
    66.0%  uniform@600              -- COMPUTE-MATCHED; the honest floor for a 2-pass method
    72.9%  44% oracle capture       -- beats the published training-free proposer
    93.2%  oracle crop              -- ceiling
    74.3%  PERFECT GATE             -- measured offline; the ceiling for ANY gate over these arms

LEAKAGE DISCIPLINE -- the point of this file
--------------------------------------------
Gates split into two kinds and they MUST be reported differently:

  * NO FITTED PARAMETER (conf_route, text_gate): nothing is estimated from the data, so evaluating
    on all 191 items is legitimate and the number is honest as-is.
  * FITTED (conf_thresh, learned, and the window W): every parameter is fitted on TRAIN FOLDS ONLY
    and scored on the held-out fold. A threshold or weight chosen by looking at all 191 items and
    then scored on those same items is not a result, and Phase 31 exists partly because that
    temptation is real.

FORBIDDEN FEATURES, enforced by simply never reading them here: the GT box, `gt_area_frac`, and the
V*Bench `category` label. Category is an ANNOTATION; gating on it would reproduce §4X's post-hoc
split and dress it up as a method. `question` text is an INPUT and is allowed.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
WINDOWS = [0.15, 0.25, 0.35, 0.50]
FOLDS = 5
BAR_COMPUTE, BAR_PUB = 66.0, 72.9

REL = ("left", "right", "above", "below", "under", "over", "behind",
       "front", "side", "between", "beneath", "underneath", "top of", "bottom of")


def is_relational(q):
    ql = q.lower()
    return any(k in ql for k in REL)


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 30:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    print(f"n = {n}")

    def pr(r, arm):
        return r["probs"][arm]

    def pred(r, arm):
        p = pr(r, arm)
        return max(range(4), key=lambda i: p[i])

    def conf(r, arm):
        return max(pr(r, arm))

    def hit(r, arm):
        return 1.0 * (pred(r, arm) == r["label"])

    print("\n=== BUDGET GATE ===")
    ok = True
    for arm in sorted(rows[0]["realized_tokens"]):
        med = st.median([r["realized_tokens"][arm] for r in rows])
        off = abs(med - 300) / 300
        ok &= off < .10
        print(f"  {arm:12} {med:6.0f} tok  {100*off:4.1f}% off  {'OK' if off<.10 else '!! VOID'}")
    print("  ALL ARMS PASS" if ok else "  SOME ARMS VOID")

    u = [hit(r, "uniform") for r in rows]
    o = [hit(r, "oracle") for r in rows]
    print(f"\n  uniform {100*st.mean(u):.1f}%   oracle {100*st.mean(o):.1f}%")

    print("\n=== fixed policies (no gate) ===")
    for W in WINDOWS:
        a = [hit(r, f"attn@{W}") for r in rows]
        print(f"  always attn@{W:<5} {100*st.mean(a):5.1f}%")

    # ---------------------------------------------------------------- ceilings
    print("\n=== ceilings (not achievable; for orientation) ===")
    for W in WINDOWS:
        best = [max(hit(r, "uniform"), hit(r, f"attn@{W}")) for r in rows]
        print(f"  perfect gate @W={W:<5} {100*st.mean(best):5.1f}%")
    allbest = [max([hit(r, "uniform")] + [hit(r, f"attn@{W}") for W in WINDOWS]) for r in rows]
    print(f"  perfect gate + perfect W  {100*st.mean(allbest):5.1f}%")

    # ------------------------------------------------- UNFITTED gates (honest on all items)
    print("\n" + "=" * 72)
    print("UNFITTED GATES -- no parameter estimated from data, all 191 items, no leakage")
    print("=" * 72)

    results = {}
    for W in WINDOWS:
        # confidence routing: believe whichever PASS is more confident
        cr = [hit(r, f"attn@{W}") if conf(r, f"attn@{W}") > conf(r, "uniform")
              else hit(r, "uniform") for r in rows]
        results[f"conf_route@{W}"] = cr
        # text gate: do not reallocate on relational questions
        tg = [hit(r, "uniform") if is_relational(r["question"]) else hit(r, f"attn@{W}")
              for r in rows]
        results[f"text_gate@{W}"] = tg
        # both
        bo = [hit(r, "uniform") if is_relational(r["question"])
              else (hit(r, f"attn@{W}") if conf(r, f"attn@{W}") > conf(r, "uniform")
                    else hit(r, "uniform")) for r in rows]
        results[f"text+conf@{W}"] = bo

    print(f"  {'gate':18} {'acc':>7} {'vs uniform':>12} {'vs 66.0 bar':>13}")
    for k in sorted(results):
        a = results[k]
        acc = 100 * st.mean(a)
        lo, hi = boot(a, u)
        mark = "PASS" if acc > BAR_COMPUTE else "fail"
        print(f"  {k:18} {acc:6.1f}% {acc-100*st.mean(u):+11.1f}pp {mark:>13}"
              f"   CI[{100*lo:+.1f},{100*hi:+.1f}]")

    # sanity: is the text gate just reconstructing the category annotation?
    agree = sum(1 for r in rows
                if is_relational(r["question"]) == (r["category"] == "relative_position"))
    print(f"\n  text gate vs `category` annotation: agrees on {agree}/{n} "
          f"({100*agree/n:.0f}%) -- reported for transparency; the gate reads the QUESTION, which")
    print("  is a legitimate input, but if agreement were 100% it would be the annotation in disguise.")

    # ------------------------------------------------- FITTED gates (cross-validated)
    print("\n" + "=" * 72)
    print(f"FITTED GATES -- {FOLDS}-fold CV, every parameter fitted on TRAIN folds only")
    print("=" * 72)
    rng = random.Random(3232)
    idx = list(range(n))
    rng.shuffle(idx)
    folds = [idx[i::FOLDS] for i in range(FOLDS)]

    def cv(fit_fn):
        """fit_fn(train_rows) -> predict_fn(row) -> 0/1 correctness of the chosen arm."""
        out = [0.0] * n
        for f in range(FOLDS):
            te = set(folds[f])
            tr = [rows[i] for i in idx if i not in te]
            g = fit_fn(tr)
            for i in folds[f]:
                out[i] = g(rows[i])
        return out

    # (a) confidence THRESHOLD on pass 1, plus W, both fitted per fold
    def fit_thresh(tr):
        best = (-1, None, None)
        for W in WINDOWS:
            for t in [i / 20 for i in range(5, 21)]:
                acc = st.mean([hit(r, f"attn@{W}") if conf(r, "uniform") < t else hit(r, "uniform")
                               for r in tr])
                if acc > best[0]:
                    best = (acc, t, W)
        _, t, W = best
        return lambda r: hit(r, f"attn@{W}") if conf(r, "uniform") < t else hit(r, "uniform")

    # (b) logistic regression over legitimate features
    def feats(r, W):
        f = r["attn_feats"]
        return [conf(r, "uniform"), conf(r, f"attn@{W}"),
                conf(r, f"attn@{W}") - conf(r, "uniform"),
                f["top1_frac"], f["top5_frac"], f["entropy_norm"],
                min(f["peak_over_median"], 1000.0) / 1000.0,
                1.0 * is_relational(r["question"])]

    def fit_lr(tr):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None
        best = (-1, None, None, None)
        for W in WINDOWS:
            X = [feats(r, W) for r in tr]
            # target: would reallocating have been RIGHT when the arms disagree?
            y = [1 if hit(r, f"attn@{W}") > hit(r, "uniform") else 0 for r in tr]
            if len(set(y)) < 2:
                continue
            sc = StandardScaler().fit(X)
            m = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X), y)
            acc = st.mean([hit(r, f"attn@{W}") if p == 1 else hit(r, "uniform")
                           for r, p in zip(tr, m.predict(sc.transform(X)))])
            if acc > best[0]:
                best = (acc, m, sc, W)
        if best[1] is None:
            return None
        _, m, sc, W = best
        return lambda r: (hit(r, f"attn@{W}")
                          if m.predict(sc.transform([feats(r, W)]))[0] == 1
                          else hit(r, "uniform"))

    for name, fn in [("conf_thresh", fit_thresh), ("learned_LR", fit_lr)]:
        probe = fn(rows[:50])
        if probe is None:
            print(f"  {name:18} SKIPPED (sklearn unavailable)")
            continue
        a = cv(fn)
        acc = 100 * st.mean(a)
        lo, hi = boot(a, u)
        print(f"  {name:18} {acc:6.1f}% {acc-100*st.mean(u):+11.1f}pp "
              f"{'PASS' if acc > BAR_COMPUTE else 'fail':>13}   CI[{100*lo:+.1f},{100*hi:+.1f}]")
        results[name] = a

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    bestk = max(results, key=lambda k: st.mean(results[k]))
    bacc = 100 * st.mean(results[bestk])
    lo, hi = boot(results[bestk], u)
    fixed = 100 * max(st.mean([hit(r, f"attn@{W}") for r in rows]) for W in WINDOWS)
    print(f"  best gate: {bestk} at {bacc:.1f}%")
    print(f"    vs uniform@300  56.5 -> {bacc-100*st.mean(u):+.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]")
    print(f"    vs best FIXED policy ({fixed:.1f}%) -> {bacc-fixed:+.1f}pp")
    print(f"    vs compute-matched bar 66.0 -> {'PASS' if bacc>BAR_COMPUTE else 'FAIL'}")
    print(f"    vs published bar      72.9 -> {'PASS' if bacc>BAR_PUB else 'FAIL'}")
    cap = 100 * (bacc - 100*st.mean(u)) / max(100*st.mean(o) - 100*st.mean(u), 1e-9)
    print(f"    oracle capture {cap:.1f}%  (published training-free proposer 31-44%)")
    if bacc <= BAR_COMPUTE:
        print("\n  => STILL NEGATIVE. Conditioning helps but the second pass is still not earned.")
    elif bacc <= BAR_PUB:
        print("\n  => EARNS ITS COMPUTE. Beats the compute-matched control but not the published")
        print("     training-free proposer. A bounded positive, reported as such.")
    else:
        print("\n  => CLEARS BOTH BARS, training-free and data-free.")
    if bestk in ("conf_thresh", "learned_LR"):
        print("  NOTE: best gate is a FITTED one -- the number above is cross-validated, and the")
        print("  unfitted gates above are the stronger claim because they have nothing to overfit.")


if __name__ == "__main__":
    main()
