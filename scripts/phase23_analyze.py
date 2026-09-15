"""Phase 23 analysis: does query-conditional placement survive against REAL AnyRes tiling?

Verdict rules, fixed before results (same discipline as Phase 20):
  * Budget matching is checked on REALIZED tokens FIRST. A contrast across mismatched budgets is
    void -- this is the bug that faked Phase 17 v1 (19% budget advantage to the favoured arm).
  * The headline contrast is `alloc_query` vs `anyres`, NOT vs `uniform`. `uniform` is reported
    only to show how much of the previously-claimed gap was against a weak baseline.
  * `anyres_pick` separates PLACEMENT from CROP TIGHTNESS; without it the two are confounded.
  * 4-way argmax is immune to the bias-shift trap, so no false-positive arm is needed.
"""
import json
import math
import random
import statistics as st
from collections import Counter

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BUDGETS = [300, 600, 1200]
ARMS = ["uniform", "anyres", "anyres_pick", "alloc_random", "alloc_query"]

# MEASURED CONSTRAINT (not a calibration bug -- see FINDINGS bug #21):
# Qwen3-VL's processor gives EVERY image a floor of 64 merged tokens, however small it is
# (28x28 px -> 64 tokens; 112x112 px -> 64 tokens). A 2x2 AnyRes grid plus a thumbnail is 5
# images, so the tiling arm cannot cost less than 5*64 = 320 tokens. B=300 is therefore
# INFEASIBLE for `anyres`, which is why it pins at 350 no matter how it is calibrated.
# This is a real property of the tiling pathway -- tiling has a MINIMUM budget of
# n_tiles x 64 tokens -- and it is reported as a result, not silently dropped.
PER_IMAGE_TOKEN_FLOOR = 64
INFEASIBLE = {("anyres", 300)}


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
    r = [json.loads(l) for l in open(f"{DATA}/phase23_anyres_results.jsonl")]
    seen, u = set(), []
    for x in r:
        if x["question_id_full"] not in seen:
            seen.add(x["question_id_full"]); u.append(x)
    r = u
    print(f"V*Bench n={len(r)}")
    print(f"  grids chosen: {dict(Counter(tuple(x['grid']) for x in r))}")
    print(f"  median target area fraction: {st.median([x['bbox_area_frac'] for x in r]):.6f}")

    print("\n" + "=" * 88)
    print("GATE: BUDGET MATCH (realized merged tokens, median). Mismatched contrasts are VOID.")
    print("=" * 88)
    ok = True
    for B in BUDGETS:
        d = {a: int(st.median([x["realized_tokens"][f"{a}@{B}"] for x in r])) for a in ARMS}
        sp = (max(d.values()) - min(d.values())) / min(d.values())
        feas = {a: v for a, v in d.items() if (a, B) not in INFEASIBLE}
        spf = (max(feas.values()) - min(feas.values())) / min(feas.values())
        good = spf < .10
        ok &= good
        note = "".join(f"  [{a}@{B} INFEASIBLE: floor {PER_IMAGE_TOKEN_FLOOR}/img]"
                       for a, b in INFEASIBLE if b == B)
        print(f"  B={B:<5} " + "  ".join(f"{a}={v}" for a, v in d.items())
              + f"   spread(feasible) {100*spf:.1f}%" + ("  OK" if good else "  !! TOO WIDE") + note)
    if not ok:
        print("\n  !! At least one budget is mismatched. Treat those rows as uninterpretable.")

    hits = {}
    print("\n" + "=" * 88)
    print("ACCURACY (4-way MCQ) at matched budget")
    print("=" * 88)
    for B in BUDGETS:
        print(f"\n  --- B={B} ---")
        for a in ARMS:
            h = [1 if x["pred"][f"{a}@{B}"] == x["label"] else 0 for x in r]
            hits[f"{a}@{B}"] = h
            lo, hi = wilson(sum(h), len(h))
            print(f"    {a:<14}{100*sum(h)/len(h):6.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]")

    print("\n" + "=" * 88)
    print("DECIDING CONTRASTS (paired, same items, matched realized budget)")
    print("=" * 88)
    contrasts = [
        ("alloc_query", "anyres",       "query placement vs ANYRES TILING   <-- THE HEADLINE"),
        ("alloc_query", "uniform",      "query placement vs uniform (old, weak baseline)"),
        ("anyres",      "uniform",      "anyres vs uniform (how weak WAS our baseline?)"),
        ("anyres_pick", "anyres",       "query-PICKED tile vs full grid (placement alone)"),
        ("alloc_query", "anyres_pick",  "tight crop vs grid-aligned tile (tightness alone)"),
        ("alloc_query", "alloc_random", "query vs size-matched random region (old decider)"),
    ]
    for B in BUDGETS:
        print(f"\n  --- B={B} ---")
        for a, b, lab in contrasts:
            if (a, B) in INFEASIBLE or (b, B) in INFEASIBLE:
                print(f"    {lab:<52}  -- VOID: {'/'.join(x for x in (a, b) if (x, B) in INFEASIBLE)}"
                      f"@{B} is below the {PER_IMAGE_TOKEN_FLOOR}-token/image floor")
                continue
            ha, hb = hits[f"{a}@{B}"], hits[f"{b}@{B}"]
            lo, hi, fr = paired(ha, hb)
            d = 100 * (sum(ha) - sum(hb)) / len(ha)
            sig = "  SIGNIFICANT" if lo > 0 else ("  (reversed)" if hi < 0 else "  (spans 0)")
            print(f"    {lab:<52}{d:+6.1f}pp  CI [{100*lo:+6.1f},{100*hi:+6.1f}]{sig}")

    print("\n" + "=" * 88)
    print("HOW MUCH OF THE OLD GAP WAS AGAINST A STRAWMAN?")
    print("  fraction of (alloc_query - uniform) that anyres alone already recovers")
    print("=" * 88)
    for B in BUDGETS:
        if ("anyres", B) in INFEASIBLE:
            continue
        aq = sum(hits[f"alloc_query@{B}"]) / len(r)
        un = sum(hits[f"uniform@{B}"]) / len(r)
        an = sum(hits[f"anyres@{B}"]) / len(r)
        if abs(aq - un) < 1e-9:
            continue
        frac = (an - un) / (aq - un)
        verdict = ("  <-- our uniform baseline WAS a strawman" if frac > 0.5 else
                   "  <-- tiling does not explain the gap")
        print(f"  B={B:<5} uniform {100*un:.1f}%  anyres {100*an:.1f}%  alloc_query {100*aq:.1f}%"
              f"   anyres recovers {100*frac:.0f}% of the gap{verdict}")

    print("\n" + "=" * 88)
    print("CROSS-BUDGET: does query placement beat AnyRes tiling at a LARGER budget?")
    print("  (the production-relevant question -- real systems tile at a big budget)")
    print("=" * 88)
    for Ba in BUDGETS:
        for Bb in BUDGETS:
            if Bb <= Ba or ("anyres", Bb) in INFEASIBLE:
                continue
            ha, hb = hits[f"alloc_query@{Ba}"], hits[f"anyres@{Bb}"]
            lo, hi, fr = paired(ha, hb)
            ta = int(st.median([x["realized_tokens"][f"alloc_query@{Ba}"] for x in r]))
            tb = int(st.median([x["realized_tokens"][f"anyres@{Bb}"] for x in r]))
            tag = "  <-- FEWER TOKENS, BETTER ACCURACY" if lo > 0 else ""
            print(f"  alloc_query@{Ba} ({ta} tok, {100*sum(ha)/len(ha):.1f}%) vs "
                  f"anyres@{Bb} ({tb} tok, {100*sum(hb)/len(hb):.1f}%): "
                  f"CI [{100*lo:+.1f},{100*hi:+.1f}]{tag}")

    print("\n" + "=" * 88)
    print("BY TARGET SIZE (does tiling rescue the small-target regime?)")
    print("=" * 88)
    qs = sorted(x["bbox_area_frac"] for x in r)
    cuts = [qs[int(len(qs) * f)] for f in (1/3, 2/3)]
    for lo_i, (name, sel) in enumerate([
            ("smallest 1/3", lambda x: x["bbox_area_frac"] < cuts[0]),
            ("middle 1/3", lambda x: cuts[0] <= x["bbox_area_frac"] < cuts[1]),
            ("largest 1/3", lambda x: x["bbox_area_frac"] >= cuts[1])]):
        ix = [i for i, x in enumerate(r) if sel(x)]
        row = f"  {name:<14} n={len(ix):<4}"
        for B in [600]:
            for a in ["uniform", "anyres", "alloc_query"]:
                acc = 100 * sum(hits[f"{a}@{B}"][i] for i in ix) / max(1, len(ix))
                row += f"  {a}@{B}={acc:5.1f}%"
        print(row)


if __name__ == "__main__":
    main()
