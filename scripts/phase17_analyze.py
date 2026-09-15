"""Phase 17 (E1) analysis: does query-conditional allocation beat uniform AT EQUAL TOKEN BUDGET?

Verdict rules, fixed before results (see phase17 docstring):
  * `alloc_random` is the decider. alloc_query beating uniform is NOT enough -- a non-uniform layout
    might help regardless of where it points. alloc_query must beat alloc_random at matched budget.
  * Budget matching is verified on REALIZED tokens (image_grid_thw), reported per arm. If realized
    totals differ by more than a few percent the comparison is void.
  * Every recovery number carries its false-positive rate; verdicts read on discrimination.
"""
import json
import math
import random
import statistics as st

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BUDGETS = [150, 300, 600]
ARMS = ["uniform", "alloc_query", "alloc_random", "crop_only"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def paired(a, b, nb=5000, seed=17):
    rng = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in ix) / n - sum(b[i] for i in ix) / n)
    d.sort()
    return d[int(.025 * nb)], d[int(.975 * nb)], sum(1 for x in d if x > 0) / nb


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/phase17_budget_results.jsonl")]
    seen, u = set(), []
    for r in rows:
        if r["uid"] not in seen:
            seen.add(r["uid"]); u.append(r)
    rows = u
    P = [r for r in rows if r["group"] == "denial"]
    N = [r for r in rows if r["group"] == "negative"]
    print(f"RePOPE-clean confident denials n={len(P)}   clean negatives n={len(N)}")
    if not N:
        print("!! NEGATIVE ARM MISSING -- recovery numbers below are UNINTERPRETABLE alone.")

    print("\n" + "=" * 92)
    print("BUDGET MATCH CHECK (realized merged tokens, median) -- the comparison is void without it")
    print("=" * 92)
    for B in BUDGETS:
        cells = {a: int(st.median([r["realized_tokens"][f"{a}@{B}"] for r in rows])) for a in ARMS}
        spread = (max(cells.values()) - min(cells.values())) / max(1, min(cells.values()))
        print(f"  target B={B:<5} " + "  ".join(f"{a}={v}" for a, v in cells.items())
              + f"   spread {100*spread:.1f}%" + ("  OK" if spread < .08 else "  !! TOO WIDE"))

    print("\n" + "=" * 92)
    print("RECOVERY on confident denials / FP on true absences / DISCRIMINATION")
    print("=" * 92)
    hits = {}
    for B in BUDGETS:
        print(f"\n  --- budget B={B} ---")
        for a in ARMS:
            k = f"{a}@{B}"
            hp = [1 if r["arms"][k] > 0.5 else 0 for r in P]
            hn = [1 if r["arms"][k] > 0.5 else 0 for r in N] if N else [0]
            hits[k] = (hp, hn)
            rec, fp = sum(hp) / len(hp), sum(hn) / len(hn)
            lo, hi = wilson(sum(hp), len(hp))
            print(f"    {a:<14}{100*rec:6.1f}% [{100*lo:5.1f},{100*hi:5.1f}]   FP {100*fp:5.1f}%"
                  f"   disc {100*(rec-fp):+6.1f}")

    print("\n" + "=" * 92)
    print("THE DECIDING CONTRASTS (paired, same items, matched realized budget)")
    print("=" * 92)
    for B in BUDGETS:
        print(f"\n  --- B={B} ---")
        for a, b, lab in [("alloc_query", "uniform", "allocation vs UNIFORM (same budget)"),
                          ("alloc_query", "alloc_random", "allocation vs RANDOM REGION  <-- DECIDER"),
                          ("crop_only", "alloc_query", "crop-only vs allocation (upper bound)")]:
            ha, hb = hits[f"{a}@{B}"][0], hits[f"{b}@{B}"][0]
            lo, hi, fr = paired(ha, hb)
            d = 100 * (sum(ha) - sum(hb)) / len(ha)
            sig = "  SIGNIFICANT" if lo > 0 else ("  (reversed)" if hi < 0 else "  (spans 0)")
            print(f"    {lab:<44}{d:+6.1f}pp  CI [{100*lo:+6.1f},{100*hi:+6.1f}] {100*fr:5.1f}%{sig}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    wins = []
    for B in BUDGETS:
        lo, _, _ = paired(hits[f"alloc_query@{B}"][0], hits[f"alloc_random@{B}"][0])
        if lo > 0:
            wins.append(B)
    if wins:
        print(f"  Query-conditional allocation beats a size-matched RANDOM region at matched budget")
        print(f"  at B={wins}. The win is ALLOCATION, not merely a non-uniform layout.")
    else:
        print("  alloc_query does NOT beat alloc_random at any budget. Per the pre-registration,")
        print("  any advantage over uniform is then 'non-uniform layouts happen to help', NOT a")
        print("  query-conditional allocation effect. Report as such.")


if __name__ == "__main__":
    main()
