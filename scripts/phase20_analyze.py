"""Phase 20 analysis: budget-matched allocation on V*Bench (4-way MCQ).

Verdict rules, fixed in the phase20 docstring before results:
  * `alloc_random` is the decider -- beating `uniform` alone would only show that non-uniform
    layouts help, not that QUERY-CONDITIONAL placement helps.
  * Budget matching verified on REALIZED tokens; a comparison across mismatched budgets is void.
  * 4-way argmax accuracy is intrinsically immune to the bias-shift trap (Phase 16 `rand_last`),
    so no false-positive arm is needed here -- uniformly inflating an option cannot raise argmax
    accuracy.
"""
import json
import math
import random
import statistics as st

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BUDGETS = [300, 600, 1200]
ARMS = ["uniform", "alloc_query", "alloc_random", "crop_only"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def paired(a, b, nb=5000, seed=19):
    rng = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in ix) / n - sum(b[i] for i in ix) / n)
    d.sort()
    return d[int(.025 * nb)], d[int(.975 * nb)], sum(1 for x in d if x > 0) / nb


def main():
    r = [json.loads(l) for l in open(f"{DATA}/phase20_vstar_results.jsonl")]
    seen, u = set(), []
    for x in r:
        if x["question_id_full"] not in seen:
            seen.add(x["question_id_full"]); u.append(x)
    r = u
    print(f"V*Bench n={len(r)}")
    print(f"  median target area fraction: {st.median([x['bbox_area_frac'] for x in r]):.6f}")
    print(f"  median image: {int(st.median([x['img_wh'][0] for x in r]))}x"
          f"{int(st.median([x['img_wh'][1] for x in r]))}")
    from collections import Counter
    print(f"  categories: {dict(Counter(x['category'] for x in r))}")

    print("\n" + "=" * 84)
    print("BUDGET MATCH (realized merged tokens, median)")
    print("=" * 84)
    for B in BUDGETS:
        d = {a: int(st.median([x["realized_tokens"][f"{a}@{B}"] for x in r])) for a in ARMS}
        sp = (max(d.values()) - min(d.values())) / min(d.values())
        print(f"  B={B:<5} " + "  ".join(f"{a}={v}" for a, v in d.items())
              + f"   spread {100*sp:.1f}%" + ("  OK" if sp < .08 else "  !! TOO WIDE"))

    hits = {}
    print("\n" + "=" * 84)
    print("ACCURACY (4-way MCQ) at matched budget")
    print("=" * 84)
    for B in BUDGETS:
        print(f"\n  --- B={B} ---")
        for a in ARMS:
            h = [1 if x["pred"][f"{a}@{B}"] == x["label"] else 0 for x in r]
            hits[f"{a}@{B}"] = h
            lo, hi = wilson(sum(h), len(h))
            print(f"    {a:<14}{100*sum(h)/len(h):6.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]")

    print("\n" + "=" * 84)
    print("DECIDING CONTRASTS (paired, same items, matched realized budget)")
    print("=" * 84)
    for B in BUDGETS:
        print(f"\n  --- B={B} ---")
        for a, b, lab in [("alloc_query", "uniform", "allocation vs UNIFORM"),
                          ("alloc_query", "alloc_random", "allocation vs RANDOM REGION  <-- DECIDER"),
                          ("crop_only", "alloc_query", "crop-only vs allocation (upper bound)")]:
            ha, hb = hits[f"{a}@{B}"], hits[f"{b}@{B}"]
            lo, hi, fr = paired(ha, hb)
            d = 100 * (sum(ha) - sum(hb)) / len(ha)
            sig = "  SIGNIFICANT" if lo > 0 else ("  (reversed)" if hi < 0 else "  (spans 0)")
            print(f"    {lab:<44}{d:+6.1f}pp  CI [{100*lo:+6.1f},{100*hi:+6.1f}] {100*fr:5.1f}%{sig}")

    print("\n" + "=" * 84)
    print("CROSS-BUDGET: does query-placed allocation beat uniform at a LARGER budget?")
    print("=" * 84)
    for Ba in BUDGETS:
        for Bb in BUDGETS:
            if Bb <= Ba:
                continue
            ha, hb = hits[f"alloc_query@{Ba}"], hits[f"uniform@{Bb}"]
            lo, hi, fr = paired(ha, hb)
            ta = int(st.median([x["realized_tokens"][f"alloc_query@{Ba}"] for x in r]))
            tb = int(st.median([x["realized_tokens"][f"uniform@{Bb}"] for x in r]))
            tag = "  <-- FEWER TOKENS, BETTER ACCURACY" if lo > 0 else ""
            print(f"  alloc_query@{Ba} ({ta} tok, {100*sum(ha)/len(ha):.1f}%) vs "
                  f"uniform@{Bb} ({tb} tok, {100*sum(hb)/len(hb):.1f}%): "
                  f"CI [{100*lo:+.1f},{100*hi:+.1f}]{tag}")

    print("\n" + "=" * 84)
    print("NATIVELY-WRONG SUBSET (closest analogue to Phase 17's recovery metric)")
    print("=" * 84)
    for B in BUDGETS:
        wrong = [i for i, x in enumerate(r) if x["pred"][f"uniform@{B}"] != x["label"]]
        if not wrong:
            continue
        for a in ["alloc_query", "alloc_random", "crop_only"]:
            k = sum(1 for i in wrong if r[i]["pred"][f"{a}@{B}"] == r[i]["label"])
            print(f"  B={B:<5} {a:<14} recovers {k:3d}/{len(wrong):<3d} = {100*k/len(wrong):5.1f}% "
                  f"of items uniform@{B} got wrong")
        print()


if __name__ == "__main__":
    main()
