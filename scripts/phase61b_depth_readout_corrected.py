"""
Phase 61b -- RE-RUN of Phase 61 with the CORRECT final layer.

WHY THIS EXISTS
---------------
Phase 61 asked "why do the last layers destroy signal, and can a free rule exploit it?" Its premise
was Phase 60's finding that decoding at L24 beats the final layer by +5.1pp on sub-token items.

That premise was an artefact: the lens computed every layer as lm_head(final_norm(h_i)), which is
right for intermediate layers but WRONG for the last one, because HuggingFace returns
hidden_states[-1] with the final norm already applied. Only the final layer was corrupted
(max logit error 23.47 vs 0.06), so it looked worse than it is.

This file redoes the whole analysis with the final layer taken from the model's own logits:
    intermediate layers  <- Phase 60 lens (correct)
    final layer          <- Phase 64 `baseline` arm (model.logits)
A join check asserts the two runs agree on an intermediate layer before anything is reported.

WHAT IS BEING RE-ASKED
----------------------
  (A) Is there ANY read-out gap at all -- does any layer, chosen out of fold, beat the final layer?
  (B) Do the four Phase-61 hypotheses survive once the baseline is correct?
      H1 evidence-dependent prior fallback · H2 evidence-independent degradation
      H3 sharpening artefact · H4 noise
  (C) Do any of the free depth-selection rules beat reading at the final layer?

If (A) is zero the rest is moot, and that is the expected outcome -- but it is re-derived here
rather than asserted.
"""
import json
import random
import statistics as st
from collections import Counter

P60 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"
P64 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase64_component_ablation.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    a = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P60))}
    b = {r["question_id_full"]: r for r in (json.loads(l) for l in open(P64))}
    k = sorted(set(a) & set(b))
    nL = len(a[k[0]]["uniform"])
    am = lambda p: max(range(4), key=lambda i: p[i])

    def lay(q, cond):
        return a[q][cond][:nL - 1] + [b[q]["probs"][cond]["baseline"]]

    # --- join check: the two runs must agree on an INTERMEDIATE layer (which was never corrupted)
    agree = sum(1 for q in k if am(a[q]["uniform"][24]) ==
                am(b[q]["probs"]["uniform"]["trunc_L24"])) / len(k)
    assert agree > 0.99, f"join unsafe: intermediate-layer agreement only {agree:.3f}"
    print(f"n = {len(k)};  join check: intermediate-layer agreement {100*agree:.1f}%  OK")
    print("final layer taken from model.logits, NOT from norm(hidden_states[-1])\n")

    sub = [q for q in k if a[q]["tokens_on_target"] < 1.0]
    res_ = [q for q in k if a[q]["tokens_on_target"] >= 1.0]

    # ---------------------------------------------------------------- (A)
    print("=" * 74)
    print("(A) IS THERE ANY READ-OUT GAP? layer chosen OUT OF FOLD")
    print("=" * 74)
    print(f"  {'stratum':<16}{'n':>5}{'final':>9}{'CV best layer':>15}{'delta':>9}{'95% CI':>17}")
    for nm, g in (("ALL ITEMS", k), ("SUB-TOKEN", sub), ("resolvable", res_)):
        if len(g) < 20:
            continue
        idx = list(range(len(g)))
        random.Random(11).shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        fin = [1.0 * (am(lay(g[i], "uniform")[-1]) == a[g[i]]["label"]) for i in range(len(g))]
        oof = [0.0] * len(g)
        picked = []
        for f in range(5):
            te = set(folds[f])
            tr = [i for i in idx if i not in te]
            best = max(range(nL), key=lambda li: st.mean(
                1.0 * (am(lay(g[i], "uniform")[li]) == a[g[i]]["label"]) for i in tr))
            picked.append(best)
            for i in te:
                oof[i] = 1.0 * (am(lay(g[i], "uniform")[best]) == a[g[i]]["label"])
        d = [x - y for x, y in zip(oof, fin)]
        lo, hi = boot(d)
        print(f"  {nm:<16}{len(g):>5}{100*st.mean(fin):>8.1f}%{100*st.mean(oof):>14.1f}%"
              f"{100*st.mean(d):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
              f"{'  SIG' if (lo > 0 or hi < 0) else ''}   layers {picked}")

    # ---------------------------------------------------------------- (B)
    print("\n" + "=" * 74)
    print("(B) DO THE FOUR HYPOTHESES SURVIVE? (they explained a gap that does not exist)")
    print("=" * 74)
    BEST = 24
    print(f"\n  H2 -- evidence-dependence of the L{BEST}-vs-final difference")
    print(f"  {'condition':<12}{'stratum':<14}{'L'+str(BEST):>8}{'final':>8}{'diff':>8}")
    for cond in ("uniform", "oracle"):
        for nm, g in (("sub-token", sub), ("resolvable", res_)):
            if len(g) < 20:
                continue
            bb = st.mean(1.0 * (am(lay(q, cond)[BEST]) == a[q]["label"]) for q in g)
            ff = st.mean(1.0 * (am(lay(q, cond)[-1]) == a[q]["label"]) for q in g)
            print(f"  {cond:<12}{nm:<14}{100*bb:>7.1f}%{100*ff:>7.1f}%{100*(bb-ff):>+7.1f}")

    print(f"\n  H1 -- when L{BEST} is right and the final layer is wrong, what does the final say?")
    for nm, g in (("sub-token", sub), ("resolvable", res_)):
        if len(g) < 20:
            continue
        flip = [q for q in g if am(lay(q, "uniform")[BEST]) == a[q]["label"]
                and am(lay(q, "uniform")[-1]) != a[q]["label"]]
        if not flip:
            print(f"  {nm:<12} n_flip=0 -- nothing to explain")
            continue
        c = Counter(am(lay(q, "uniform")[-1]) for q in flip)
        base = Counter(am(lay(q, "uniform")[-1]) for q in g)
        tot = sum(base.values())
        top, ct = c.most_common(1)[0]
        print(f"  {nm:<12} n_flip={len(flip):<4} " + " ".join(f"{'ABCD'[x]}={v}" for x, v in sorted(c.items()))
              + f"   most common '{'ABCD'[top]}' {100*ct/len(flip):.0f}% vs base {100*base[top]/tot:.0f}%")

    print(f"\n  H4 -- churn symmetry L{BEST} -> final")
    for nm, g in (("sub-token", sub), ("resolvable", res_)):
        if len(g) < 20:
            continue
        w2r = sum(1 for q in g if am(lay(q, "uniform")[BEST]) != a[q]["label"]
                  and am(lay(q, "uniform")[-1]) == a[q]["label"])
        r2w = sum(1 for q in g if am(lay(q, "uniform")[BEST]) == a[q]["label"]
                  and am(lay(q, "uniform")[-1]) != a[q]["label"])
        print(f"  {nm:<12} right->wrong {r2w:<4} wrong->right {w2r:<4} net {w2r-r2w:+d}")

    # ---------------------------------------------------------------- (C)
    print("\n" + "=" * 74)
    print("(C) DO ANY FREE DEPTH RULES BEAT THE FINAL LAYER?")
    print("=" * 74)
    for nm, g in (("ALL ITEMS", k), ("SUB-TOKEN", sub)):
        n = len(g)
        idx = list(range(n))
        random.Random(11).shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        fin = [1.0 * (am(lay(g[i], "uniform")[-1]) == a[g[i]]["label"]) for i in range(n)]
        RULES = {
            "fixed_cv": lambda q, tr: max(range(nL), key=lambda li: st.mean(
                1.0 * (am(lay(x, "uniform")[li]) == a[x]["label"]) for x in tr)),
            "max_conf": lambda q, tr: max(range(nL), key=lambda li: max(lay(q, "uniform")[li])),
            "max_conf_late": lambda q, tr: max(range(nL // 2, nL),
                                               key=lambda li: max(lay(q, "uniform")[li])),
        }
        print(f"\n  {nm} (n={n})")
        print(f"    {'rule':<16}{'acc':>8}{'vs final':>11}{'95% CI':>17}")
        print(f"    {'final (baseline)':<16}{100*st.mean(fin):>7.1f}%{0.0:>+10.1f}   {'[+0.0,+0.0]':>15}")
        for rn, fn in RULES.items():
            acc = [0.0] * n
            for f in range(5):
                te = set(folds[f])
                tr = [g[i] for i in idx if i not in te]
                for i in te:
                    li = fn(g[i], tr)
                    acc[i] = 1.0 * (am(lay(g[i], "uniform")[li]) == a[g[i]]["label"])
            d = [x - y for x, y in zip(acc, fin)]
            lo, hi = boot(d)
            print(f"    {rn:<16}{100*st.mean(acc):>7.1f}%{100*st.mean(d):>+10.1f}"
                  f"   [{100*lo:+.1f},{100*hi:+.1f}]{'  SIG' if (lo > 0 or hi < 0) else ''}")
        orc = [max(1.0 * (am(lay(q, "uniform")[li]) == a[q]["label"]) for li in range(nL)) for q in g]
        print(f"    {'oracle_layer':<16}{100*st.mean(orc):>7.1f}%"
              f"{100*(st.mean(orc)-st.mean(fin)):>+10.1f}   (ceiling, not a method)")


if __name__ == "__main__":
    main()
