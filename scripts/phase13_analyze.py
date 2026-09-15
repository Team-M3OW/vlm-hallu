"""Phase 13 analysis: which non-crop interventions actually move the needle?

Reported on ALL items and on the RePOPE-clean subset (Phase 11 showed the crop cohort is 61.7%
not-clean, and that cleaning roughly doubled the interference effect).

Verdicts are read on DISCRIMINATION (recovery - false-positive rate), never raw recovery -- Phase
10's other-scene arm looked like +17.0pp on raw recovery and had a CI spanning zero once its
19.1% FP rate was accounted for.
"""
import json
import math
import random
import sys

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase11_repope_audit import load_repope, verdict

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
ORDER = ["baseline", "crop_alone",
         "black_outside", "blur_outside", "dim_outside", "gray_outside",
         "upscale2x", "upscale3x", "redbox", "prompt_focus", "prompt_coords"]
FAMILY = {
    "baseline": "reference", "crop_alone": "reference",
    "black_outside": "scene removed, RES CONSTANT", "blur_outside": "scene removed, RES CONSTANT",
    "dim_outside": "scene removed, RES CONSTANT", "gray_outside": "scene removed, RES CONSTANT",
    "upscale2x": "RES RAISED, scene kept", "upscale3x": "RES RAISED, scene kept",
    "redbox": "pointing only", "prompt_focus": "pointing only", "prompt_coords": "pointing only",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def dd(pa, na, pb, nb_, n_boot=4000, seed=7):
    """CI on the difference in DISCRIMINATION between two arms."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        pi = [rng.randrange(len(pa)) for _ in range(len(pa))]
        ni = [rng.randrange(len(na)) for _ in range(len(na))]
        da = sum(pa[i] for i in pi) / len(pa) - sum(na[i] for i in ni) / len(na)
        db = sum(pb[i] for i in pi) / len(pb) - sum(nb_[i] for i in ni) / len(nb_)
        out.append(da - db)
    out.sort()
    return out[int(.025 * n_boot)], out[int(.975 * n_boot)]


def table(P, N, title):
    print(f"\n{'='*100}\n{title}   (positives n={len(P)}, negatives n={len(N)})\n{'='*100}")
    print(f"  {'arm':<16}{'family':<28}{'recovery':>10}{'95% CI':>16}{'FP':>8}{'disc':>9}")
    hits = {}
    for a in ORDER:
        hp = [1 if r["arms"].get(a, 0) > 0.5 else 0 for r in P]
        hn = [1 if r["arms"].get(a, 0) > 0.5 else 0 for r in N]
        hits[a] = (hp, hn)
        rec = sum(hp) / len(hp)
        fp = sum(hn) / len(hn)
        lo, hi = wilson(sum(hp), len(hp))
        print(f"  {a:<16}{FAMILY[a]:<28}{100*rec:9.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]"
              f"{100*fp:7.1f}%{100*(rec-fp):+8.1f}")
    print(f"\n  vs crop_alone (difference in discrimination; CI excluding 0 = genuinely different):")
    ca, cn = hits["crop_alone"]
    for a in ORDER:
        if a in ("crop_alone",):
            continue
        lo, hi = dd(hits[a][0], hits[a][1], ca, cn)
        tag = "" if lo <= 0 <= hi else ("  WORSE than crop" if hi < 0 else "  BETTER than crop")
        print(f"    {a:<16} {100*lo:+7.1f},{100*hi:+7.1f} pp{tag}")
    return hits


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/phase13_bfs_results.jsonl")]
    seen, u = set(), []
    for r in rows:
        if r["uid"] not in seen:
            seen.add(r["uid"]); u.append(r)
    rows = u
    P = [r for r in rows if r["group"] == "positive"]
    N = [r for r in rows if r["group"] == "negative"]

    # GATE 1: did the upscale arms actually produce more visual tokens?
    print("=" * 100)
    print("GATE: upscale arms must actually increase visual-token count (FINDINGS bug #5)")
    print("=" * 100)
    for a in ("baseline", "upscale2x", "upscale3x", "crop_alone"):
        toks = [r["grids"][a][1] * r["grids"][a][2] // 4 for r in P if a in r.get("grids", {})]
        toks.sort()
        print(f"  {a:<12} median merged visual tokens = {toks[len(toks)//2]:6d}   "
              f"min {toks[0]:5d}  max {toks[-1]:6d}")
    b = [r["grids"]["baseline"][1]*r["grids"]["baseline"][2] for r in P]
    u2 = [r["grids"]["upscale2x"][1]*r["grids"]["upscale2x"][2] for r in P]
    inc = sum(1 for x, y in zip(b, u2) if y > x)
    print(f"  -> upscale2x produced MORE tokens than baseline on {inc}/{len(b)} items")
    if inc < 0.5 * len(b):
        print("  !! UPSCALE ARM IS LARGELY A NO-OP -- the processor capped it. Do NOT read")
        print("     'resolution does not matter' from these arms; they did not vary resolution.")

    table(P, N, "ALL ITEMS")

    rp = load_repope()
    cp = [r for r in P if verdict(rp, (r["split"], str(r["question_id"]))) == "CLEAN"]
    cn = [r for r in N if verdict(rp, (r["split"], str(r["question_id"]))) == "CLEAN"]
    if cp and cn:
        table(cp, cn, "RePOPE-CLEAN ITEMS ONLY (the trustworthy numbers)")


if __name__ == "__main__":
    main()
