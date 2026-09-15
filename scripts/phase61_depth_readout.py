"""
Phase 61: (A) WHY do the last layers destroy signal, and (B) can a free rule exploit it?
Offline on Phase 60's per-layer ABCD distributions. No GPU.

(A) THE DIAGNOSIS
-----------------
§12A found that on sub-token items, decoding at L24 beats the final layer by +5.1pp [+0.7,+10.3]
out of fold, with L24 chosen in all 5 folds. Something in L25-27 discards answer-relevant signal.
Four hypotheses, each with a distinct offline signature:

  H1 OPTION-POSITION PRIOR. Late layers fall back on a letter prior when visual evidence is weak.
     Signature: on items where L24 is right and the final layer is wrong, the final layer's answers
     concentrate on ONE letter far above its base rate, and the effect is stronger on sub-token
     (visually weak) items than on resolvable ones.
  H2 EVIDENCE-INDEPENDENT DEGRADATION. Late layers hurt everywhere.
     Signature: the same drop under the ORACLE crop (strong evidence) as under uniform.
  H3 SHARPENING ARTIFACT. The argmax flips only where the top two options were nearly tied.
     Signature: flipped items have tiny L24 margins; confident items never flip.
  H4 NOISE. No systematic structure; flips are symmetric (as many wrong->right as right->wrong).
     Signature: net change ~0 with large gross churn.

(B) THE METHOD
--------------
Reading at L24 is LAYER TRUNCATION: it also skips 3 of 28 layers, so it is cheaper as well as more
accurate. Per-item depth selection could do better, if a LABEL-FREE signal picks the layer. Rules
tested, all free and all evaluated OUT OF FOLD where they have any fitted quantity:

    fixed_final      the deployed read-out (baseline)
    fixed_cv         one layer chosen on training folds
    max_conf         the layer whose ABCD distribution is most confident
    stable_k         the earliest layer whose argmax then holds for k consecutive layers
    agree_final      last layer agreeing with the majority vote over the final block
    oracle_layer     per-item best layer -- the CEILING, not a method

Depth is reported alongside accuracy, since a rule that reads at L24 costs 3/28 = 11% less compute.
"""
import json
import random
import statistics as st
from collections import Counter

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    nL = len(rows[0]["uniform"])
    am = lambda p: max(range(4), key=lambda i: p[i])
    hit = lambda p, y: 1.0 * (am(p) == y)
    sub = [r for r in rows if r["tokens_on_target"] < 1.0]
    res_ = [r for r in rows if r["tokens_on_target"] >= 1.0]
    print(f"n = {len(rows)}  (sub-token {len(sub)}, resolvable {len(res_)});  {nL} layers\n")

    print("=" * 74)
    print("(A) WHY DO THE LAST LAYERS HURT?")
    print("=" * 74)
    BEST = 24
    print(f"\n  H2 -- is the drop evidence-INDEPENDENT? (L{BEST} vs final, per condition/stratum)")
    print(f"  {'condition':<12}{'stratum':<14}{'L'+str(BEST):>8}{'final':>8}{'drop':>8}")
    for cond in ("uniform", "oracle"):
        for nm, g in (("sub-token", sub), ("resolvable", res_)):
            if len(g) < 20:
                continue
            b = st.mean(hit(r[cond][BEST], r["label"]) for r in g)
            f = st.mean(hit(r[cond][-1], r["label"]) for r in g)
            print(f"  {cond:<12}{nm:<14}{100*b:>7.1f}%{100*f:>7.1f}%{100*(b-f):>+7.1f}")
    print("  H2 predicts equal drops everywhere; H1 predicts the drop concentrates where evidence")
    print("  is weak (uniform + sub-token).")

    print(f"\n  H1 -- when L{BEST} is right and the final layer is wrong, WHAT does the final say?")
    for nm, g in (("sub-token", sub), ("resolvable", res_)):
        if len(g) < 20:
            continue
        flip = [r for r in g if hit(r["uniform"][BEST], r["label"]) == 1
                and hit(r["uniform"][-1], r["label"]) == 0]
        if not flip:
            continue
        c = Counter(am(r["uniform"][-1]) for r in flip)
        base = Counter(am(r["uniform"][-1]) for r in g)
        tot = sum(base.values())
        top, ct = c.most_common(1)[0]
        print(f"  {nm:<12} n_flip={len(flip):<4} final-layer answers: "
              + " ".join(f"{'ABCD'[k]}={v}" for k, v in sorted(c.items()))
              + f"   most common '{'ABCD'[top]}' {100*ct/len(flip):.0f}% "
              f"vs its base rate {100*base[top]/tot:.0f}%")
    print("  (a letter far above its base rate = the late layers falling back on a position prior)")

    print(f"\n  H3 -- do flips happen only where L{BEST} was nearly tied?")
    for nm, g in (("sub-token", sub), ("resolvable", res_)):
        if len(g) < 20:
            continue
        mar = lambda r: (lambda p: sorted(p, reverse=True)[0] - sorted(p, reverse=True)[1])(
            r["uniform"][BEST])
        fl = [mar(r) for r in g if am(r["uniform"][BEST]) != am(r["uniform"][-1])]
        st_ = [mar(r) for r in g if am(r["uniform"][BEST]) == am(r["uniform"][-1])]
        if fl and st_:
            print(f"  {nm:<12} median L{BEST} margin: flipped {st.median(fl):.3f}  "
                  f"unflipped {st.median(st_):.3f}   (n_flip={len(fl)})")

    print(f"\n  H4 -- is the churn symmetric? (gross flips vs net change)")
    for nm, g in (("sub-token", sub), ("resolvable", res_)):
        if len(g) < 20:
            continue
        w2r = sum(1 for r in g if hit(r["uniform"][BEST], r["label"]) == 0
                  and hit(r["uniform"][-1], r["label"]) == 1)
        r2w = sum(1 for r in g if hit(r["uniform"][BEST], r["label"]) == 1
                  and hit(r["uniform"][-1], r["label"]) == 0)
        print(f"  {nm:<12} L{BEST}->final:  right->wrong {r2w}   wrong->right {w2r}   "
              f"net {w2r - r2w:+d}")

    print("\n" + "=" * 74)
    print("(B) CAN A FREE RULE EXPLOIT IT? -- depth selection, out of fold")
    print("=" * 74)
    for nm, g in (("ALL ITEMS", rows), ("SUB-TOKEN", sub)):
        n = len(g)
        idx = list(range(n))
        random.Random(11).shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        fin = [hit(r["uniform"][-1], r["label"]) for r in g]

        def ev(pick):
            acc, dep = [0.0] * n, [0.0] * n
            for f in range(5):
                te = set(folds[f])
                tr = [i for i in idx if i not in te]
                for i in te:
                    li = pick(g[i], [g[j] for j in tr])
                    acc[i] = hit(g[i]["uniform"][li], g[i]["label"])
                    dep[i] = (li + 1) / nL
            return acc, dep

        RULES = {
            "fixed_final": lambda r, tr: nL - 1,
            "fixed_cv": lambda r, tr: max(range(nL), key=lambda li: st.mean(
                hit(x["uniform"][li], x["label"]) for x in tr)),
            "max_conf": lambda r, tr: max(range(nL), key=lambda li: max(r["uniform"][li])),
            "max_conf_late": lambda r, tr: max(range(nL // 2, nL),
                                               key=lambda li: max(r["uniform"][li])),
            "stable_3": lambda r, tr: next(
                (li for li in range(nL // 2, nL - 3)
                 if len({am(r["uniform"][li + k]) for k in range(4)}) == 1), nL - 1),
        }
        print(f"\n  {nm} (n={n})")
        print(f"    {'rule':<16}{'acc':>8}{'vs final':>11}{'95% CI':>17}{'mean depth':>12}")
        for rn, fn in RULES.items():
            a, d = ev(fn)
            dd = [x - y for x, y in zip(a, fin)]
            lo, hi = boot(dd)
            print(f"    {rn:<16}{100*st.mean(a):>7.1f}%{100*st.mean(dd):>+10.1f}"
                  f"   [{100*lo:+.1f},{100*hi:+.1f}]{100*st.mean(d):>11.0f}%"
                  f"{'  SIG' if (lo > 0 or hi < 0) else ''}")
        orc = [max(hit(r["uniform"][li], r["label"]) for li in range(nL)) for r in g]
        print(f"    {'oracle_layer':<16}{100*st.mean(orc):>7.1f}%"
              f"{100*(st.mean(orc)-st.mean(fin)):>+10.1f}   (ceiling, not a method)")


if __name__ == "__main__":
    main()
