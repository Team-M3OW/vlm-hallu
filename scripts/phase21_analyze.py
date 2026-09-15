"""
Phase 21 analyzer: WITHIN-BENCHMARK test of the boundary condition (RePOPE-clean, stratified).

WHY THIS IS THE STRONGEST FORM OF THE BOUNDARY CLAIM
----------------------------------------------------
§5.5 says: allocation pays when the evidence is smaller than the budget can resolve, and costs when
it is not. So far that rests on comparing V*Bench (71% sub-token, allocation +12.0pp) against
HR-Bench 4k (resolvable, allocation -8.8pp) -- two benchmarks differing in model prompt, question
format, image scale and annotation. Any of those could carry the effect instead.

RePOPE-clean spans BOTH regimes inside ONE benchmark: the offline screen puts 28% of items below
one merged token at B0=300 and 72% above it, and Phase 21 sampled it STRATIFIED on exactly that
axis -- 120 items in each of [0,0.5) [0.5,2) [2,8) [8,32) [32,inf) merged tokens on object, plus 200
negatives. Same model, same yes/no prompt, same images, same budget. Only target size varies.

THREE THINGS THAT COULD FAKE A CONFIRMATION, AND HOW EACH IS HANDLED
--------------------------------------------------------------------
1. FLOOR/CEILING. Small targets are harder irrespective of allocation, so `uniform` is low on the
   sub-token strata and high on the resolvable ones. A gain that appears only where there is
   headroom is a floor effect, not the boundary condition. So `uniform`'s OWN level is printed
   beside every delta, and if uniform is already >90% on a resolvable stratum that stratum is
   declared UNINFORMATIVE rather than counted as confirmation.
2. THE SIGN, NOT THE SIZE. A smaller-but-positive gain on large targets is consistent with a floor
   effect. The boundary condition predicts `alloc_query - uniform < 0` there. That is the
   pre-registered discriminator and it is evaluated as such below.
3. CHANCE IS 50%, NOT 25%. This is yes/no, scored as P(yes)>0.5 on positives. These accuracies are
   NOT comparable to the 4-way MCQ numbers in §2/§4/§5 and are never quoted beside them.

`alloc_random` travels with every stratum, so placement-vs-cropping is separable along the whole
curve -- the same contrast that decided V*Bench (§4) and HR-Bench (§5B).
"""
import json
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase21_crossing_results.jsonl"
ORDER = ["[0,0.5)", "[0.5,2)", "[2,8)", "[8,32)", "[32,1000000000.0)"]
NICE = {"[0,0.5)": "<0.5 tok", "[0.5,2)": "0.5-2 tok", "[2,8)": "2-8 tok",
        "[8,32)": "8-32 tok", "[32,1000000000.0)": ">32 tok"}


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    arms = list(rows[0]["arms"])
    pos = [r for r in rows if r["group"] == "positive"]
    neg = [r for r in rows if r["group"] != "positive"]
    print(f"n = {len(rows)}  ({len(pos)} positives stratified on target size, {len(neg)} negatives)")
    print("READ-OUT: yes/no, correct = P(yes)>0.5 on positives.  CHANCE = 50%, not 25%.")
    print("These numbers are NOT comparable to the 4-way MCQ accuracies in §2/§4/§5.")

    print("\n=== BUDGET GATE (all arms must sit within 10% of each other) ===")
    med = {a: st.median([r["realized_tokens"][a] for r in rows]) for a in arms}
    lo, hi = min(med.values()), max(med.values())
    for a in arms:
        print(f"  {a:16}{med[a]:6.0f} tok")
    spread = (hi - lo) / lo
    print(f"  spread {100*spread:.1f}%  -> {'OK' if spread < .10 else '!! CONTRAST VOID'}")
    if spread >= .10:
        print("  refusing to report contrasts on a voided budget gate.")
        return

    corr = lambda r, a: 1.0 * (r["arms"][a] > 0.5)

    print("\n=== THE CROSSING CURVE (positives, by merged tokens on target) ===")
    print(f"  {'stratum':<12}{'n':>4}{'med tok':>9}{'uniform':>9}"
          f"{'alloc_query':>13}{'delta':>9}{'95% CI':>18}{'vs random':>11}")
    deltas = {}
    for s in ORDER:
        g = [r for r in pos if r["stratum"] == s]
        if not g:
            continue
        u = [corr(r, "uniform") for r in g]
        q = [corr(r, "alloc_query") for r in g]
        rd = [corr(r, "alloc_random") for r in g]
        d = [x - y for x, y in zip(q, u)]
        lo_, hi_ = boot(d)
        deltas[s] = (st.mean(u), st.mean(d), lo_, hi_)
        mt = st.median([r["tokens_on_object"] for r in g])
        flag = "  CEILING" if st.mean(u) > 0.90 else ""
        print(f"  {NICE[s]:<12}{len(g):>4}{mt:>9.2f}{100*st.mean(u):>8.1f}%"
              f"{100*st.mean(q):>12.1f}%{100*st.mean(d):>+8.1f}"
              f"   [{100*lo_:+.1f},{100*hi_:+.1f}]{100*(st.mean(q)-st.mean(rd)):>+10.1f}{flag}")

    print("\n=== THE 25/75 SPLIT ARM: is a crossing about ALLOCATION or about the 50/50 division? ===")
    print(f"  {'stratum':<12}{'alloc_query(50/50)':>20}{'alloc_query25(25/75)':>22}")
    for s in ORDER:
        g = [r for r in pos if r["stratum"] == s]
        if not g:
            continue
        u = st.mean([corr(r, "uniform") for r in g])
        a1 = st.mean([corr(r, "alloc_query") for r in g]) - u
        a2 = st.mean([corr(r, "alloc_query25") for r in g]) - u
        print(f"  {NICE[s]:<12}{100*a1:>+19.1f}{100*a2:>+21.1f}")

    print("\n=== FALSE POSITIVES (negatives; allocation must not simply inflate `yes`) ===")
    print(f"  {'arm':<16}{'FP rate':>9}")
    for a in arms:
        print(f"  {a:<16}{100*st.mean([corr(r, a) for r in neg]):>8.1f}%")
    print("  (a method that wins on positives by saying `yes` more is not localizing anything)")

    print("\n" + "=" * 78)
    print("VERDICT vs the PRE-REGISTERED boundary condition")
    print("=" * 78)
    sub = [s for s in ORDER[:2] if s in deltas]
    res = [s for s in ORDER[2:] if s in deltas]
    sm = st.mean([deltas[s][1] for s in sub]) if sub else float("nan")
    print(f"  sub-token strata (<2 tok):  mean delta {100*sm:+.1f}pp")
    for s in res:
        u, d, lo_, hi_ = deltas[s]
        info = "UNINFORMATIVE (uniform at ceiling)" if u > 0.90 else (
            "SIGN FLIP -- boundary condition confirmed" if hi_ < 0 else
            "negative but n.s." if d < 0 else "still positive")
        print(f"  {NICE[s]:<12} uniform {100*u:5.1f}%  delta {100*d:+6.1f}pp  -> {info}")
    flips = [s for s in res if deltas[s][0] <= 0.90 and deltas[s][3] < 0]
    weak = [s for s in res if deltas[s][0] <= 0.90 and deltas[s][1] < 0 and deltas[s][3] >= 0]
    if flips:
        print(f"\n  => WITHIN-BENCHMARK CONFIRMATION. Allocation significantly HURTS on "
              f"{len(flips)} resolvable stratum/strata\n     while helping on sub-token targets, "
              f"with model, prompt, images and budget all held fixed.")
    elif weak:
        print(f"\n  => DIRECTIONALLY CONSISTENT but not significant on {len(weak)} stratum/strata.")
        print("     The within-benchmark test is SUGGESTIVE ONLY; §5.5 continues to rest on the")
        print("     cross-benchmark contrast. Report it that way.")
    else:
        print("\n  => NO WITHIN-BENCHMARK CROSSING. Allocation does not go negative on resolvable")
        print("     targets here. §5.5's boundary condition is NOT reproduced within RePOPE and")
        print("     must be stated as a cross-benchmark observation only.")


if __name__ == "__main__":
    main()
