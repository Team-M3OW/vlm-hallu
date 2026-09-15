"""
Phase 30b analyzer. Two questions, asked in order, because the second only matters if the first
passes.

Q1 (THE GATE) -- did Phase 22's attention signal transfer to V*Bench at B0=300?
    Statistic: `gt_pct`, the rank of the grid cell containing the GT centre, as a fraction of
    n_img. Chance is 0.5 BY CONSTRUCTION -- a uniformly random ranking puts the GT cell at a
    uniformly random percentile, mean 0.5. No permutation null is needed; the null is exact.

    Also reported: P(top-1 cell falls inside the GT box). Its chance level is NOT 0.5, it is the
    GT box's share of the image -- median 0.108%, so ~1 in 900. Anything above a couple of percent
    is a large effect.

Q2 (only if Q1 passes) -- does a percentile box recover a usable region?
    `pctl@r` vs `minmax@r`, scored on containment / area_frac / headroom. Headroom = containment /
    area_frac, and 1.0 is what proposing the whole image gets you, so headroom must exceed 1 by a
    clear margin for allocation to pay anything.

Reporting discipline: Phase 30a's min/max result is the PRE-REGISTERED headline and stays the
headline. Everything here is post-hoc repair and is labelled as such.
"""
import json
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase30b_robust_readout.jsonl"


def q(v, p):
    return st.quantiles(v, n=100)[p - 1] if len(v) > 2 else (v[0] if v else float("nan"))


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if not rows:
        print("no rows yet")
        return
    print(f"n = {len(rows)} items   (post-hoc; the pre-registered headline is Phase 30a)")
    bad = [r for r in rows if "note" in r]
    if bad:
        print(f"  WARNING: {len(bad)} token-count mismatches")
    print(f"  grid: median {st.median([r['n_img_tokens'] for r in rows]):.0f} merged tokens")
    gtok = [r["gt_tokens"] for r in rows]
    print(f"  GT target size IN TOKENS: median {st.median(gtok):.2f}  "
          f"p10 {q(gtok,10):.2f}  p90 {q(gtok,90):.2f}  "
          f"({100*sum(1 for g in gtok if g < 1)/len(gtok):.0f}% are SUB-TOKEN)")
    gt_area = [r["gt_area_frac"] for r in rows]
    print(f"  chance P(top-1 in GT) = median GT area share = {st.median(gt_area):.5f} "
          f"(~1 in {1/max(st.median(gt_area),1e-9):.0f})")

    print("\n" + "=" * 72)
    print("Q1 THE GATE: did the Phase 22 signal transfer? (chance gt_pct = 0.500 exactly)")
    print("=" * 72)
    print(f"  {'layer':7} {'med gt_pct':>11} {'mean':>8} {'p10':>8} "
          f"{'top1_in_gt':>11} {'med concen':>11}")
    verdict = {}
    for L in ("L2", "L4", "L8"):
        if L not in rows[0]["layers"]:
            continue
        p = [r["layers"][L]["gt_pct"] for r in rows]
        ing = [r["layers"][L]["top1_in_gt"] for r in rows]
        con = [r["layers"][L]["concentration"] for r in rows]
        verdict[L] = st.median(p)
        print(f"  {L:7} {st.median(p):11.3f} {st.mean(p):8.3f} {q(p,10):8.3f} "
              f"{100*sum(ing)/len(ing):10.1f}% {st.median(con):11.1f}x")
    best = min(verdict, key=verdict.get)
    mp = verdict[best]
    print(f"\n  best layer {best}, median gt_pct {mp:.3f}")
    if mp < 0.35:
        print("  => TRANSFERRED. The GT cell ranks well above chance; sinks are a removable")
        print("     nuisance and a robust read-out is worth testing (Q2).")
    elif mp < 0.45:
        print("  => WEAK. Above chance but not by much. A read-out repair is unlikely to clear the")
        print("     72.9% bar; report as a bounded signal, not a method.")
    else:
        print("  => DID NOT TRANSFER. The GT cell ranks at ~chance at B0=300. Phase 22's RePOPE")
        print("     result does not carry to V*Bench here, and NO read-out repair rescues it.")
        print("     This is the finding: the localizer is not in the attention map at this budget.")

    print("\n" + "=" * 72)
    print("Q2 read-out repair: percentile box vs min/max  (post-hoc)")
    print("=" * 72)
    keys = sorted(rows[0]["layers"]["L2"]["readouts"].keys())
    print(f"  {'layer/readout':22} {'med cont':>9} {'med area':>9} {'med head':>9} {'area<=.25':>10}")
    for L in ("L2", "L4", "L8"):
        if L not in rows[0]["layers"]:
            continue
        for k in keys:
            d = [r["layers"][L]["readouts"][k] for r in rows]
            c = [x["containment"] for x in d]
            a = [x["area_frac"] for x in d]
            h = [x["headroom"] for x in d]
            print(f"  {L+'/'+k:22} {st.median(c):9.3f} {st.median(a):9.3f} {st.median(h):9.2f}"
                  f" {100*sum(1 for x in a if x<=0.25)/len(a):9.1f}%")

    print("\n=== by category ===")
    cats = sorted({r["category"] for r in rows})
    for cat in cats:
        g = [r for r in rows if r["category"] == cat]
        p = [r["layers"][best]["gt_pct"] for r in g]
        ing = [r["layers"][best]["top1_in_gt"] for r in g]
        print(f"  {cat:20} n={len(g):3d}  med gt_pct {st.median(p):.3f}  "
              f"top1_in_gt {100*sum(ing)/len(ing):.1f}%")

    print("\n=== by GT size in tokens (does the signal survive the sub-token regime?) ===")
    srt = sorted(rows, key=lambda r: r["gt_tokens"])
    for i in range(4):
        g = srt[i*len(srt)//4:(i+1)*len(srt)//4]
        p = [r["layers"][best]["gt_pct"] for r in g]
        ing = [r["layers"][best]["top1_in_gt"] for r in g]
        print(f"  gt_tokens {g[0]['gt_tokens']:6.2f}-{g[-1]['gt_tokens']:6.2f} n={len(g):3d}  "
              f"med gt_pct {st.median(p):.3f}  top1_in_gt {100*sum(ing)/len(ing):5.1f}%")


if __name__ == "__main__":
    main()
