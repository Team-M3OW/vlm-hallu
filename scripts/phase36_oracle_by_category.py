"""
Phase 36: the test that decides whether §6C is about REGION COUNT or about WINDOW QUALITY.
Offline, no GPU -- every arm needed was already logged by Phase 32/33.

THE OBJECTION
-------------
§6C claims reallocation fails on multi-region questions because a crop destroys the second region.
The obvious alternative: our LOCALIZER simply picks the wrong one of the two needed regions, and a
better proposal would fix it. Phase 32 already logged the arm that separates these -- `oracle`,
which crops to the GROUND-TRUTH BOX. That is a perfect single-region proposal, by construction.

    oracle ALSO LOSES on relative_position  -> even a PERFECT crop of a correct region fails when
                                               the question needs two. Region count confirmed; no
                                               localizer-quality objection survives.
    oracle WINS on relative_position        -> the failure is proposal quality, not region count.
                                               §6C collapses to "our localizer picks one of two".

THE SECOND, FINER VERSION
-------------------------
Phase 32 logged W in {0.15,0.25,0.35,0.5} and Phase 33 logged W in {0.15,0.25}. If
relative_position recovers toward zero as the window GROWS, the boundary is not "counting regions"
but "the window must cover the EVIDENCE SET" -- a sharper and more defensible statement of the same
thing. If it stays negative even at W=0.5 while oracle also loses, region count survives as stated.

`rand@W` travels with every window on V*Bench, so placement is separable from window size at every
point on the sweep.
"""
import json
import random
import statistics as st
from collections import defaultdict

VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
HR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33_hrbench_transfer.jsonl"


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(VS)]
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])

    print("=" * 80)
    print("THE DECIDER: does a PERFECT (oracle GT-box) crop also lose on relative_position?")
    print("=" * 80)
    print(f"  {'category':<20}{'n':>4}{'uniform':>9}{'oracle':>9}{'delta':>8}{'95% CI':>18}")
    for cat in ("direct_attributes", "relative_position"):
        g = [r for r in rows if r["category"] == cat]
        u = [hit(r, "uniform") for r in g]
        o = [hit(r, "oracle") for r in g]
        d = [x - y for x, y in zip(o, u)]
        lo, hi = boot(d)
        print(f"  {cat:<20}{len(g):>4}{100*st.mean(u):>8.1f}%{100*st.mean(o):>8.1f}%"
              f"{100*st.mean(d):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n" + "=" * 80)
    print("WINDOW SWEEP, V*Bench: does relative_position recover as the window GROWS?")
    print("=" * 80)
    Ws = [w for w in ("0.15", "0.25", "0.35", "0.5") if f"attn@{w}" in rows[0]["probs"]]
    print(f"  {'category':<20}" + "".join(f"{'W='+w:>12}" for w in Ws) + f"{'oracle':>12}")
    for cat in ("direct_attributes", "relative_position"):
        g = [r for r in rows if r["category"] == cat]
        u = st.mean([hit(r, "uniform") for r in g])
        cells = []
        for w in Ws:
            cells.append(100 * (st.mean([hit(r, f"attn@{w}") for r in g]) - u))
        o = 100 * (st.mean([hit(r, "oracle") for r in g]) - u)
        print(f"  {cat:<20}" + "".join(f"{c:>+11.1f}" for c in cells) + f"{o:>+11.1f}")
    print("  (deltas vs uniform@300, same realized budget)")

    print("\n  placement control at every window (attn - rand):")
    print(f"  {'category':<20}" + "".join(f"{'W='+w:>12}" for w in Ws))
    for cat in ("direct_attributes", "relative_position"):
        g = [r for r in rows if r["category"] == cat]
        cells = []
        for w in Ws:
            a = st.mean([hit(r, f"attn@{w}") for r in g])
            b = st.mean([hit(r, f"rand@{w}") for r in g])
            cells.append(100 * (a - b))
        print(f"  {cat:<20}" + "".join(f"{c:>+11.1f}" for c in cells))

    print("\n" + "=" * 80)
    print("WINDOW SWEEP, HR-Bench 4k (W=0.15 vs 0.25)")
    print("=" * 80)
    hr = [json.loads(l) for l in open(HR)]
    hw = [w for w in ("0.15", "0.25") if f"attn@{w}" in hr[0]["probs"]]
    print(f"  {'category':<10}{'n':>5}{'uniform':>9}" + "".join(f"{'W='+w:>12}" for w in hw))
    for cat in ("single", "cross"):
        g = [r for r in hr if r["category"] == cat]
        u = st.mean([hit(r, "uniform") for r in g])
        cells = [100 * (st.mean([hit(r, f"attn@{w}") for r in g]) - u) for w in hw]
        print(f"  {cat:<10}{len(g):>5}{100*u:>8.1f}%" + "".join(f"{c:>+11.1f}" for c in cells))

    print("\n" + "=" * 80)
    print("MAIN EFFECTS, reported SEPARATELY (the 2x2 interaction is underpowered)")
    print("=" * 80)
    de, re_ = [], []
    for r in rows:
        v = hit(r, "attn@0.15") - hit(r, "uniform")
        (de if r["category"] == "direct_attributes" else re_).append(v)
    hs, hc = [], []
    for r in hr:
        v = hit(r, "attn@0.15") - hit(r, "uniform")
        (hs if r["category"] == "single" else hc).append(v)
    pooled_single = de + hs
    pooled_multi = re_ + hc
    for nm, d in [("single-region (V*+HR)", pooled_single), ("multi-region (V*+HR)", pooled_multi)]:
        lo, hi = boot(d)
        print(f"  {nm:<26}n={len(d):<5}delta {100*st.mean(d):>+6.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]")
    diff = st.mean(pooled_single) - st.mean(pooled_multi)
    rng = random.Random(1)
    bs = sorted(
        (sum(pooled_single[rng.randrange(len(pooled_single))] for _ in pooled_single)/len(pooled_single)
         - sum(pooled_multi[rng.randrange(len(pooled_multi))] for _ in pooled_multi)/len(pooled_multi))
        for _ in range(10000))
    print(f"  SEPARATION single - multi  {100*diff:+.1f}pp  CI[{100*bs[250]:+.1f},{100*bs[9750]:+.1f}]")
    print("  (this is the REGION-COUNT MAIN EFFECT; the size main effect is RePOPE's n=600 curve)")


if __name__ == "__main__":
    main()
