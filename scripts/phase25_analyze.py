"""Phase 25 analysis: query placement vs NATIVE AnyRes, on the model AnyRes was invented for.

Verdict rules, fixed before results:
  * Budget gate on REALIZED tokens comes first. LLaVA-NeXT's count is a STEP function of input
    size (it snaps to an `image_grid_pinpoints` entry), so exact matching is impossible; a contrast
    is admissible only if the realized spread is under 10%, and void otherwise. No exceptions --
    this is the check that caught bug #18 and bug #20.
  * `alloc_random` is the decider. Beating `anyres` alone would show that spending the budget on a
    sub-region helps, not that the QUESTION is what should choose the region.
  * `uniform` is an UNMATCHED reference row (see the phase25 docstring) and never enters a paired
    contrast.
  * 4-way argmax is immune to the bias-shift trap, so no false-positive arm is required.
  * This phase can RETRACT Phase 23 section 4K. If `anyres >= alloc_query` here, 4K's tiling rows
    were a Qwen format artifact and must be withdrawn, keeping only the grid-alignment finding.
"""
import json
import math
import random
import statistics as st

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
MATCHED = ["anyres", "alloc_query", "alloc_random"]
ALL_ARMS = ["uniform"] + MATCHED


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
    return d[int(.025 * nb)], d[int(.975 * nb)]


def main():
    r = [json.loads(l) for l in open(f"{DATA}/phase25_llavanext_results.jsonl")]
    seen, u = set(), []
    for x in r:
        if x["question_id_full"] not in seen:
            seen.add(x["question_id_full"]); u.append(x)
    r = u
    budgets = ["native"]   # per-item native AnyRes cost; see the phase25 docstring
    print(f"LLaVA-NeXT-vicuna-7b on V*Bench, n={len(r)}   budgets={budgets}")
    print(f"  median target area fraction: {st.median([x['bbox_area_frac'] for x in r]):.6f}"
          f"   median native AnyRes cost: {int(st.median([x['native_tokens'] for x in r]))} tokens")

    print("\n" + "=" * 88)
    print("GATE: BUDGET MATCH across the matched trio (realized image tokens, median)")
    print("=" * 88)
    # PER-ITEM gate. A median-based gate is NOT sufficient and this is not a nicety: on the real
    # V*Bench run the medians agreed exactly (spread 0.0%) while **39.3% of individual items** were
    # mismatched by more than 10% (p90 spread 46.4%, max 115.2%). Aggregating over those items mixes
    # matched and unmatched comparisons into one number. We therefore restrict every contrast to the
    # items whose own realized counts agree, and report how many were dropped.
    admissible = {}
    for B in budgets:
        d = {a: int(st.median([x["realized_tokens"][f"{a}@{B}"] for x in r])) for a in ALL_ARMS}
        keep = []
        for x in r:
            t = [x["realized_tokens"][f"{a}@{B}"] for a in MATCHED]
            if (max(t) - min(t)) / min(t) <= .10:
                keep.append(x)
        admissible[B] = len(keep) >= 30
        print(f"  PER-ITEM matched: {len(keep)}/{len(r)} items "
              f"({100*(len(r)-len(keep))/len(r):.1f}% dropped as unmatched)")
        globals()["_KEEP"] = keep
        m = {a: d[a] for a in MATCHED}
        sp = (max(m.values()) - min(m.values())) / min(m.values())
        print(f"  {B:<8} " + "  ".join(f"{a}={v}" for a, v in d.items())
              + f"   trio spread {100*sp:.1f}%" + ("  OK" if admissible[B] else "  !! VOID"))

    hits = {}
    print("\n" + "=" * 88)
    print("ACCURACY (4-way MCQ)")
    print("=" * 88)
    for B in budgets:
        print(f"\n  --- {B} " + ("" if admissible[B] else " [budget VOID: rows not comparable]"))
        rr = _KEEP
        for a in ALL_ARMS:
            h = [1 if x["pred"][f"{a}@{B}"] == x["label"] else 0 for x in rr]
            hits[f"{a}@{B}"] = h
            lo, hi = wilson(sum(h), len(h))
            tag = "   (unmatched reference)" if a == "uniform" else ""
            print(f"    {a:<14}{100*sum(h)/len(h):6.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]{tag}")

    print("\n" + "=" * 88)
    print("DECIDING CONTRASTS -- native format, single image, no image-count confound")
    print("=" * 88)
    for B in budgets:
        if not admissible[B]:
            print(f"\n  --- {B}: VOID (budget spread over 10%)")
            continue
        print(f"\n  --- {B} ---")
        for a, b, lab in [
                ("alloc_query", "anyres", "query placement vs NATIVE ANYRES   <-- THE HEADLINE"),
                ("alloc_query", "alloc_random", "query vs size-matched random region  <-- DECIDER"),
                ("alloc_random", "anyres", "any sub-region vs tiling (placement-free benefit)")]:
            ha, hb = hits[f"{a}@{B}"], hits[f"{b}@{B}"]
            lo, hi = paired(ha, hb)
            d = 100 * (sum(ha) - sum(hb)) / len(ha)
            sig = "  SIGNIFICANT" if lo > 0 else ("  (reversed)" if hi < 0 else "  (spans 0)")
            print(f"    {lab:<52}{d:+6.1f}pp  CI [{100*lo:+6.1f},{100*hi:+6.1f}]{sig}")

    print("\n" + "=" * 88)
    print("VERDICT ON PHASE 23 SECTION 4K")
    print("=" * 88)
    ok = [B for B in budgets if admissible[B]]
    if not ok:
        print("  No admissible budget -- 4K neither confirmed nor retracted by this run.")
    else:
        B = max(ok)
        aq = sum(hits[f"alloc_query@{B}"]) / len(r)
        an = sum(hits[f"anyres@{B}"]) / len(r)
        ar = sum(hits[f"alloc_random@{B}"]) / len(r)
        print(f"  At {B}: alloc_query {100*aq:.1f}%  anyres {100*an:.1f}%  alloc_random {100*ar:.1f}%")
        if aq > an and aq > ar:
            print("  => Query-conditional placement beats the SHIPPED high-resolution pathway on")
            print("     that pathway's own model and own format. Phase 23's confounded tiling")
            print("     contrast becomes a real claim; 4K's caveat can be narrowed.")
        elif an >= aq:
            print("  => RETRACT 4K's tiling rows: Phase 23's result was a Qwen FORMAT ARTIFACT.")
            print("     Keep only the grid-alignment finding (alloc_query vs anyres_pick).")
        else:
            print("  => alloc_query does not beat alloc_random: placement is not the operative")
            print("     variable on this architecture. C1 does not transfer -- report that.")


if __name__ == "__main__":
    main()
