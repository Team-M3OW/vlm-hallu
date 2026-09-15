"""
Phase 77 analyzer: does evidence-grounded contrastive decoding work?

DECISION ORDER, fixed before the data existed. Read TOP-DOWN and stop at the first failure.

 1. Does masking do ANYTHING? contrast_l1 for the oracle region must exceed the random region.
    If not, image-region masking is inert (SS10A's finding) and every arm below is noise.
 2. CEILING: cd_oracle must beat baseline. The oracle region is the best any localiser can target,
    so if contrast there is flat the method is dead regardless of proposal quality -- stop.
 3. METHOD: cd_head must beat both baseline AND cd_rand. Beating baseline alone proves nothing;
    any logit sharpening shifts accuracy on a 4-way MCQ.
 4. FALSIFICATION: cd_head's gain must be larger where the head's window COVERS the evidence than
    where it misses. If the gain does not track coverage, it is not about evidence -- void the arm
    whatever its accuracy.

BAR: two LM passes at B0 each, so uniform@600 (63.9% on these items, SS14D) is the compute-matched
bar, not uniform@300's 56.5%.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase77_contrastive_decode.jsonl"
BAR_600 = 0.639


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n)]


def main():
    R = [json.loads(l) for l in open(PATH)]
    if len(R) < 40:
        print(f"only {len(R)} rows; wait")
        return
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    print(f"n = {len(R)}   median realized tokens {st.median([r['realized_tokens'] for r in R]):.0f}")
    print(f"  cells masked: " + ", ".join(
        f"{k} {st.median([r['n_masked'][k] for r in R]):.0f}" for k in R[0]["n_masked"]))

    print("\n=== STEP 1: does masking do anything at all? (L1 logit shift) ===")
    for k in R[0]["contrast_l1"]:
        print(f"  {k:8} {st.mean([r['contrast_l1'][k] for r in R]):8.4f}")
    orc = st.mean([r["contrast_l1"]["oracle"] for r in R])
    rnd = st.mean([r["contrast_l1"]["rand"] for r in R])
    print(f"  oracle/random ratio = {orc/max(rnd,1e-9):.2f}x "
          f"{'(evidence region matters more)' if orc > 1.2*rnd else '<< INERT: masking evidence is no different from masking anything'}")

    base = [hit(r, "baseline") for r in R]
    print(f"\n=== ACCURACY  (baseline {100*st.mean(base):.1f}%, bar uniform@600 {100*BAR_600:.1f}%) ===")
    arms = sorted({k for r in R for k in r["probs"] if k.startswith("cd_")})
    print(f"  {'arm':22}{'acc':>8}{'vs baseline':>14}")
    print(f"  {'mask_only_head':22}{100*st.mean([hit(r,'mask_only_head') for r in R]):7.1f}%")
    for a in arms:
        v = [hit(r, a) for r in R]
        lo, hi = boot(v, base)
        print(f"  {a:22}{100*st.mean(v):7.1f}%{100*(st.mean(v)-st.mean(base)):+11.1f}pp  "
              f"CI[{lo:+.1f},{hi:+.1f}]")

    def best(pfx):
        c = [a for a in arms if a.startswith(pfx)]
        return max(c, key=lambda a: st.mean([hit(r, a) for r in R])) if c else None

    bo, bh, br = best("cd_oracle"), best("cd_head"), best("cd_rand")
    print("\n" + "=" * 70)
    ao = st.mean([hit(r, bo) for r in R]) if bo else 0
    ah = st.mean([hit(r, bh) for r in R]) if bh else 0
    ar = st.mean([hit(r, br) for r in R]) if br else 0
    ab = st.mean(base)
    print(f"STEP 2 ceiling : best cd_oracle {100*ao:.1f}% vs baseline {100*ab:.1f}% "
          f"= {100*(ao-ab):+.1f}pp")
    if ao - ab < 0.02:
        print("  => DEAD. Contrast on the GROUND-TRUTH region does not move the answer. No")
        print("     localiser can rescue this; a seventh internal-intervention null.")
        return
    print(f"STEP 3 method  : best cd_head {100*ah:.1f}%  vs rand {100*ar:.1f}% "
          f"= {100*(ah-ar):+.1f}pp")
    if ah - ar < 0.02:
        print("  => NOT EVIDENCE-SPECIFIC. Any region gives the same lift; this is logit")
        print("     sharpening, not grounding.")
        return
    print("\nSTEP 4 falsification: does the gain track COVERAGE?")
    cov = [r for r in R if r["head_cov"] >= .5]
    mis = [r for r in R if r["head_cov"] < .5]
    for nm, S in [("head COVERS", cov), ("head MISSES", mis)]:
        if not S:
            continue
        v = [hit(r, bh) for r in S]
        b2 = [hit(r, "baseline") for r in S]
        print(f"  {nm:14} n={len(S):4d}  {100*(st.mean(v)-st.mean(b2)):+6.1f}pp vs baseline")
    print(f"\n  vs compute-matched bar (uniform@600): {100*(ah-BAR_600):+.1f}pp")


if __name__ == "__main__":
    main()
