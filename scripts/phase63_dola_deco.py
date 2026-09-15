"""
Phase 63: DoLa and DeCo as baselines, on our per-layer read-outs. Offline, no GPU.

WHY
---
The literature survey found our logit-lens territory is occupied. Two methods in particular:

  DoLa (arXiv 2309.03883, ICLR 2024): contrast a MATURE (final) layer against a PREMATURE layer,
      score = log p_final - log p_premature, restricted by an adaptive plausibility constraint.
      Premature layer chosen per token by max Jensen-Shannon divergence from the final, within a
      bucket. Applied to multiple choice by likelihood scoring, so our MCQ setting is NOT a
      differentiator.
  DeCo (arXiv 2410.11779, ICLR 2025): MLLMs recognise objects in preceding layers and later layers
      suppress that recognition under language priors. Corrects with
      logits = phi(h_final) + alpha * max_prob(anchor) * phi(h_anchor), anchor chosen dynamically
      from a late band (layers 20-28 of 32 in their setup).

A third data point from the survey: ICLA (2603.00437) reports DoLa COLLAPSES on Qwen2.5-VL and DeCo
degrades, i.e. layer methods tuned on LLaVA may not transfer to Qwen VLMs -- which is our backbone
family. So running them here is informative either way.

WHAT THIS IS AND IS NOT
-----------------------
These are the SCORING RULES of DoLa and DeCo applied to our 4-way MCQ read-out, not the full
generation-time methods. Both papers do apply their rule to multiple choice, so this is a fair
comparison of the decision rule -- but it is a reimplementation, and any negative result is about
the rule in this setting, not a refutation of the papers. Their free hyper-parameters (bucket,
alpha) are SWEPT and the best is reported, which favours them over our fixed L24.

Within a 4-option softmax, log p differs from the raw logits by a constant shared across the four
options, so log p is a valid logit surrogate for ranking. Both rules are rank-based, so this is exact.
"""
import json
import math
import random
import statistics as st

P60 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"
P62 = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase62_combine.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def am(p):
    return max(range(4), key=lambda i: p[i])


def jsd(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda a, b: sum(x * math.log(max(x, 1e-12) / max(y, 1e-12)) for x, y in zip(a, b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dola(layers, bucket, alpha=0.1, static=None):
    """score = log p_final - log p_premature, with the adaptive plausibility constraint."""
    pf = layers[-1]
    if static is not None:
        j = static
    else:
        j = max(bucket, key=lambda li: jsd(layers[li], pf))
    mx = max(pf)
    ok = [c for c in range(4) if pf[c] >= alpha * mx] or list(range(4))
    sc = {c: math.log(max(pf[c], 1e-12)) - math.log(max(layers[j][c], 1e-12)) for c in ok}
    return max(sc, key=sc.get)


def deco(layers, band, alpha):
    """logits = final + alpha * max_prob(anchor) * anchor, anchor = argmax max_prob over the band."""
    pf = layers[-1]
    a = max(band, key=lambda li: max(layers[li]))
    w = max(layers[a])
    sc = [math.log(max(pf[c], 1e-12)) + alpha * w * math.log(max(layers[a][c], 1e-12))
          for c in range(4)]
    return max(range(4), key=lambda c: sc[c])


def main():
    rows = [json.loads(l) for l in open(P60)]
    nL = len(rows[0]["uniform"])
    sub = [r for r in rows if r["tokens_on_target"] < 1.0]
    print(f"n = {len(rows)} (sub-token {len(sub)}); {nL} layers. Arm = uniform@300.")
    print("DoLa/DeCo hyper-parameters are SWEPT and their BEST is reported -- this favours them.\n")

    for nm, g in (("ALL ITEMS", rows), ("SUB-TOKEN", sub)):
        n = len(g)
        fin = [1.0 * (am(r["uniform"][-1]) == r["label"]) for r in g]
        print(f"=== {nm} (n={n}) ===")
        print(f"  {'method':<34}{'acc':>8}{'vs final':>11}{'95% CI':>18}")
        print(f"  {'final layer (baseline)':<34}{100*st.mean(fin):>7.1f}%{0.0:>+10.1f}"
              f"   {'[+0.0,+0.0]':>16}")

        BUCKETS = {"lower half [0,L/2)": list(range(0, nL // 2)),
                   "upper half [L/2,L-1)": list(range(nL // 2, nL - 1)),
                   "all [0,L-1)": list(range(0, nL - 1))}
        best_d = None
        for bn, bk in BUCKETS.items():
            v = [1.0 * (dola(r["uniform"], bk) == r["label"]) for r in g]
            d = [x - y for x, y in zip(v, fin)]
            lo, hi = boot(d)
            if best_d is None or st.mean(d) > best_d[0]:
                best_d = (st.mean(d), bn, st.mean(v), lo, hi)
            print(f"  {'DoLa dynamic, ' + bn:<34}{100*st.mean(v):>7.1f}%{100*st.mean(d):>+10.1f}"
                  f"   [{100*lo:+.1f},{100*hi:+.1f}]{'  SIG' if (lo>0 or hi<0) else ''}")
        bs = None
        for li in range(nL - 1):
            v = [1.0 * (dola(r["uniform"], None, static=li) == r["label"]) for r in g]
            if bs is None or st.mean(v) > bs[0]:
                bs = (st.mean(v), li)
        d = [x - y for x, y in zip(
            [1.0 * (dola(r["uniform"], None, static=bs[1]) == r["label"]) for r in g], fin)]
        lo, hi = boot(d)
        print(f"  {'DoLa-static, best L' + str(bs[1]) + ' (oracle-picked)':<34}"
              f"{100*bs[0]:>7.1f}%{100*st.mean(d):>+10.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
              f"{'  SIG' if (lo>0 or hi<0) else ''}")

        bd = None
        for band_lo in (int(0.6 * nL), int(0.7 * nL)):
            band = list(range(band_lo, nL - 1))
            for al in (0.2, 0.5, 1.0, 2.0):
                v = [1.0 * (deco(r["uniform"], band, al) == r["label"]) for r in g]
                if bd is None or st.mean(v) > bd[0]:
                    bd = (st.mean(v), band_lo, al)
        band = list(range(bd[1], nL - 1))
        v = [1.0 * (deco(r["uniform"], band, bd[2]) == r["label"]) for r in g]
        d = [x - y for x, y in zip(v, fin)]
        lo, hi = boot(d)
        print(f"  {'DeCo, band L' + str(bd[1]) + '+, alpha=' + str(bd[2]) + ' (best)':<34}"
              f"{100*st.mean(v):>7.1f}%{100*st.mean(d):>+10.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
              f"{'  SIG' if (lo>0 or hi<0) else ''}")

        v = [1.0 * (am(r["uniform"][24]) == r["label"]) for r in g]
        d = [x - y for x, y in zip(v, fin)]
        lo, hi = boot(d)
        print(f"  {'OURS: read at L24 (fixed)':<34}{100*st.mean(v):>7.1f}%{100*st.mean(d):>+10.1f}"
              f"   [{100*lo:+.1f},{100*hi:+.1f}]{'  SIG' if (lo>0 or hi<0) else ''}")
        print()

    # ---- and against ALLOCATION, which is the actual method
    r2 = [json.loads(l) for l in open(P62)]
    s2 = [r for r in r2 if r["tokens_on_target"] < 1.0]
    fin = [1.0 * (am(r["uniform"][-1]) == r["label"]) for r in s2]
    print("=" * 74)
    print(f"CONTEXT: the same contrast for ALLOCATION (sub-token, n={len(s2)})")
    print("=" * 74)
    for lbl, v in (("multi4 @ final layer", [1.0 * (am(r["multi4"][-1]) == r["label"]) for r in s2]),
                   ("multi4 @ L24", [1.0 * (am(r["multi4"][24]) == r["label"]) for r in s2])):
        d = [x - y for x, y in zip(v, fin)]
        lo, hi = boot(d)
        print(f"  {lbl:<34}{100*st.mean(v):>7.1f}%{100*st.mean(d):>+10.1f}"
              f"   [{100*lo:+.1f},{100*hi:+.1f}]{'  SIG' if lo > 0 else ''}")
    print("\n  Every layer-based rule operates on ONE forward pass of the uniform image and cannot")
    print("  add information the pass never encoded (§10). Allocation changes what is encoded.")


if __name__ == "__main__":
    main()
