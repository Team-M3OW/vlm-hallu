"""
Phase 58 analyzer: does handing the model k candidate windows in ONE pass beat handing it one?

The decisive contrast is `multi4` vs `top1@0.15`: SAME number of passes (2), SAME total tokens
(~580 vs ~585), same localiser, same window size. The only difference is whether the budget buys one
window at B0 or four windows at B0/4. Coverage of the evidence set rises 49.9% -> 67.1% by handing
over four; the question is whether the model can use it.

Every arm is also scored against the uniform sweep measured in-run, and peak-gated variants are
assembled offline at the V*Bench-transferred firing rate.
"""
import json
import math
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase58_multicrop.jsonl"
FIRE = 0.40


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 40:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    conf = lambda r, a: max(r["probs"][a])
    anchors = sorted((st.mean(r["cost"][a]["tokens"] for r in rows),
                      st.mean(hit(r, a) for r in rows))
                     for a in ("uniform@300", "uniform@600", "uniform@1200"))

    def bar(t):
        if t <= anchors[0][0]:
            return anchors[0][1]
        for (t0, a0), (t1, a1) in zip(anchors, anchors[1:]):
            if t <= t1:
                w = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
                return a0 + w * (a1 - a0)
        (t0, a0), (t1, a1) = anchors[-2], anchors[-1]
        return a1 + (a1 - a0) / (math.log(t1) - math.log(t0)) * (math.log(t) - math.log(t1))

    print(f"n = {n} V*Bench items, Qwen3-VL-2B")
    print("  bar (in-run): " + ", ".join(f"{t:.0f}tok->{100*a:.1f}%" for t, a in anchors))
    print(f"  coverage: top-1 window {100*st.mean(r['cand_cov'][0] for r in rows):.1f}%"
          f"  ->  best of 4 {100*st.mean(max(r['cand_cov']) for r in rows):.1f}%\n")

    arms = [a for a in ("uniform@300", "uniform@600", "uniform@1200", "top1@0.15",
                        "multi2", "multi3", "multi4", "multi4_600", "multi3_scene", "oracle")
            if a in rows[0]["probs"]]
    print(f"  {'arm':<14}{'psses':>6}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    res = {}
    for a in arms:
        v = [hit(r, a) for r in rows]
        ps = st.mean(r["cost"][a]["passes"] for r in rows)
        tk = st.mean(r["cost"][a]["tokens"] for r in rows)
        b = bar(tk)
        lo, hi = boot([x - b for x in v])
        res[a] = (st.mean(v), tk, b, st.mean(v) - b, lo, hi)
        note = "  (uses GT)" if a == "oracle" else ""
        print(f"  {a:<14}{ps:>6.2f}{tk:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%"
              f"{100*(st.mean(v)-b):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]{note}")

    print("\n=== THE DECISIVE CONTRAST: same passes, same tokens, 4 windows vs 1 ===")
    for k in ("multi2", "multi3", "multi4"):
        if k not in res:
            continue
        d = [hit(r, k) - hit(r, "top1@0.15") for r in rows]
        lo, hi = boot(d)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {k} - top1@0.15  {100*st.mean(d):>+6.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  "
              f"{sig}   ({res[k][1]:.0f} vs {res['top1@0.15'][1]:.0f} tokens)")

    print("\n=== PEAK-GATED variants (pay the multi-crop pass only when the gate fires) ===")
    vals = sorted((r["peak"] for r in rows), reverse=True)
    tau = vals[max(0, int(FIRE * len(vals)) - 1)]
    print(f"  {'policy':<22}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    best = None
    for a in ("top1@0.15", "multi2", "multi3", "multi4", "multi4_600", "multi3_scene"):
        if a not in res:
            continue
        v, tk = [], []
        for r in rows:
            if r["peak"] >= tau:
                v.append(hit(r, a) if conf(r, a) > conf(r, "uniform@300") else hit(r, "uniform@300"))
                tk.append(r["cost"][a]["tokens"])
            else:
                v.append(hit(r, "uniform@300"))
                tk.append(r["cost"]["uniform@300"]["tokens"])
        t = st.mean(tk); b = bar(t)
        lo, hi = boot([x - b for x in v])
        m = st.mean(v) - b
        if best is None or m > best[0]:
            best = (m, a, st.mean(v), t, lo, hi)
        print(f"  {'gated ' + a:<22}{t:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%"
              f"{100*m:>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== by category ===")
    for cat in sorted({r["category"] for r in rows}):
        g = [r for r in rows if r["category"] == cat]
        print(f"  {cat:<20}n={len(g):<4}" + "  ".join(
            f"{a} {100*st.mean(hit(r,a) for r in g):.1f}%"
            for a in ("uniform@300", "top1@0.15", "multi4")))

    print("\n" + "=" * 74)
    if best and best[4] > 0:
        print(f"  => {best[1]} (gated) BEATS the budget axis: {100*best[0]:+.1f}pp "
              f"CI[{100*best[4]:+.1f},{100*best[5]:+.1f}] at {best[3]:.0f} tokens. SIGNIFICANT.")
    elif best:
        print(f"  => best is gated {best[1]} at {100*best[0]:+.1f}pp "
              f"CI[{100*best[4]:+.1f},{100*best[5]:+.1f}] -- positive but not significant.")
    d = [hit(r, "multi4") - hit(r, "top1@0.15") for r in rows]
    lo, hi = boot(d)
    if lo > 0:
        print(f"     Multi-crop beats single-crop at equal budget by {100*st.mean(d):+.1f}pp "
              f"CI[{100*lo:+.1f},{100*hi:+.1f}] -- the proposal bottleneck IS buyable this way.")
    elif hi < 0:
        print(f"     Multi-crop is WORSE than single-crop at equal budget ({100*st.mean(d):+.1f}pp):")
        print("     the extra candidates are not usable even when handed over directly.")
    else:
        print(f"     Multi-crop vs single-crop at equal budget: {100*st.mean(d):+.1f}pp, n.s.")


if __name__ == "__main__":
    main()
