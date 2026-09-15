"""
Phase 81 analyzer: does RANK-ONLY multi-crop clear the bar that single-crop DCR could not?

CONTEXT. SS80 withdrew the claim that DCR beats the compute-matched budget baseline: pooled it is
+4.7pp [-3.7,+13.1] on Qwen3-VL and +3.1pp [-5.2,+11.0] on Qwen2-VL, CI spanning zero on BOTH. That
is the paper's central weakness. This run is the attempt to fix it using only the operation that
has been shown to work (ranking) and none of the operations that have failed (predicting budget,
scale, or window size).

PRE-REGISTERED PREDICTION, fixed before the data existed
    Union coverage of the head's top-4 separated cells, at 75 tok each, that is ALSO above the
    SS13B cliff: 63.4%, against single-crop's 52.9% -- a +10.5pp coverage gain at IDENTICAL total
    budget. At SS6D's exchange rate (~52.7pp per coverage flip) that predicts ~+5.5pp over
    dcr_single, i.e. ~74% against the uniform@600 bar's 63.9%.

DECISION ORDER. Stop at the first failure; do not read downstream numbers.
 1. BUDGET GATE. Every arm within 10% of 300 realized tokens (600 for the bar). A multi-crop arm
    that quietly spends more than its single-crop counterpart invalidates the entire comparison.
 2. Is multi-crop better than single-crop AT ALL? dcr_multi vs dcr_single.
 3. Is it OUR RANKING or just multi-crop? dcr_multi must beat BOTH argmax_multi and rand_multi.
    If rand_multi alone explains the gain, the finding is "show the model several crops", which is
    prior art (phase 58) and not a contribution of the read-out work.
 4. THE BAR. dcr_multi vs uniform@600, with the CI. This is the number SS80 withdrew.

A gain that does not survive step 3 must be reported as multi-crop, not as DCR.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase81_rank_only.jsonl"
PREDICTED = 5.5


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n)]


def main():
    R = [json.loads(l) for l in open(PATH)]
    if len(R) < 40:
        print(f"only {len(R)} rows; wait"); return
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    ARMS = list(R[0]["probs"].keys())
    print(f"n = {len(R)}")

    print("\n=== [1] BUDGET GATE (measured; multi-crop must not overspend) ===")
    void = set()
    for a in ARMS:
        med = st.median([r["realized_tokens"][a] for r in R])
        tgt = 600 if "600" in a else 300
        off = abs(med - tgt) / tgt
        if off >= .10:
            void.add(a)
        print(f"  {a:18}{med:6.0f} tok (target {tgt})  {100*off:4.1f}%  "
              f"{'OK' if off < .10 else '!! VOID'}")
    if void:
        print(f"  !! VOIDED ARMS: {sorted(void)} -- contrasts involving them are not readable")

    acc = {a: [hit(r, a) for r in R] for a in ARMS}
    print(f"\n=== ACCURACY ===")
    for a in ARMS:
        print(f"  {a:18}{100*st.mean(acc[a]):6.1f}%{'  [VOID]' if a in void else ''}")

    def c(lo, hi, why):
        d = 100 * (st.mean(acc[hi]) - st.mean(acc[lo]))
        l, h = boot(acc[hi], acc[lo])
        flag = " [VOID]" if (lo in void or hi in void) else ""
        sig = "SIG" if l > 0 else ("neg" if h < 0 else "n.s.")
        print(f"  {hi:16} - {lo:16}{d:+7.1f}pp CI[{l:+.1f},{h:+.1f}] {sig:4} {why}{flag}")
        return d, l, h

    print(f"\n=== [2] is multi-crop better at all? (predicted {PREDICTED:+.1f}pp) ===")
    d2, l2, _ = c("dcr_single@300", "dcr_multi_k4", "THE STEP")
    print("\n=== [3] is it OUR RANKING, or just multi-crop? ===")
    d3a, l3a, _ = c("argmax_multi_k4", "dcr_multi_k4", "vs deployed ranking, same k")
    d3b, l3b, _ = c("rand_multi_k4", "dcr_multi_k4", "vs random placement, same k")
    _ = c("dcr_single@300", "argmax_multi_k4", "(does multi-crop alone explain it?)")
    print("\n=== [4] THE BAR -- the claim SS80 withdrew ===")
    d4, l4, h4 = c("uniform@600", "dcr_multi_k4", "<< DECIDES THE PAPER")
    _ = c("uniform@300", "dcr_multi_k4", "vs vanilla")
    _ = c("dcr_multi_k4", "oracle@300", "headroom left")

    print("\n" + "=" * 72)
    if void:
        print("VERDICT WITHHELD: a voided budget makes these contrasts unreadable.")
    elif l2 <= 0:
        print("=> NO STEP. Multi-crop does not beat single-crop; rank-only allocation fails and")
        print("   the paper keeps SS80's withdrawn claim withdrawn.")
    elif l3a <= 0 or l3b <= 0:
        print("=> MULTI-CROP, NOT DCR. The gain is not attributable to our ranking (it does not")
        print("   beat the deployed ranking and/or random placement at the same k). Report as")
        print("   multi-crop -- which is phase 58's prior result, not a new contribution.")
    elif l4 > 0:
        print("=> ★ CLEARS THE BAR. Rank-only allocation beats spending the same budget uniformly,")
        print("   attributable to the learned ranking. This is the method result the paper needs.")
        print("   NEXT, BEFORE CLAIMING IT: replicate on Qwen2-VL (the standard is >1 model).")
    else:
        print("=> BEATS SINGLE-CROP AND ITS CONTROLS, BUT NOT THE BAR. Honest statement: better")
        print("   allocation, still not better than more tokens. SS80's withdrawal stands.")


if __name__ == "__main__":
    main()
