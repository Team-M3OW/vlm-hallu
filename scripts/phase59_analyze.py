"""
Phase 59 analyzer: does multi-crop transfer to 4K on HR-Bench `single`?

Bar computed on THIS subset (an all-items bar would understate what uniform achieves on these
items -- the error caught in §9D). Reference points, same benchmark and subset:
    top1 crop-only, all categories  -17.3pp      four separate 4K fixes reached at best break-even
    V*Bench single-region multi4    +9.5pp [+0.8,+17.4]
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase59_multicrop_hrbench.jsonl"
FIRE = 0.40


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    rows = [r for r in rows if "uniform@1200" in r["probs"]]
    if len(rows) < 80:
        print(f"only {len(rows)} usable rows; wait")
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

    print(f"n = {n} rows / {len({r['instance'] for r in rows})} instances, "
          f"HR-Bench 4k `single` ({st.median([r['img_wh'][0] for r in rows]):.0f}px)")
    print("  bar (this subset): " + ", ".join(f"{t:.0f}tok->{100*a:.1f}%" for t, a in anchors))
    print("  reference: top1 crop-only on all categories was -17.3pp here; "
          "V*Bench single-region multi4 was +9.5pp\n")

    arms = [a for a in ("uniform@300", "uniform@600", "uniform@1200", "top1@0.15",
                        "multi2", "multi3", "multi4", "multi4_600") if a in rows[0]["probs"]]
    print(f"  {'arm':<14}{'psses':>6}{'tokens':>8}{'acc':>8}{'bar':>8}{'margin':>9}{'95% CI':>17}")
    res = {}
    for a in arms:
        v = [hit(r, a) for r in rows]
        ps = st.mean(r["cost"][a]["passes"] for r in rows)
        tk = st.mean(r["cost"][a]["tokens"] for r in rows)
        b = bar(tk)
        lo, hi = boot([x - b for x in v])
        res[a] = (st.mean(v), tk, b, st.mean(v) - b, lo, hi)
        print(f"  {a:<14}{ps:>6.2f}{tk:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%"
              f"{100*(st.mean(v)-b):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
              f"{'  SIG' if lo > 0 else ''}")

    print("\n=== THE DECISIVE CONTRAST: same passes, same tokens, k windows vs 1 ===")
    for k in ("multi2", "multi3", "multi4"):
        if k not in res:
            continue
        d = [hit(r, k) - hit(r, "top1@0.15") for r in rows]
        lo, hi = boot(d)
        print(f"  {k} - top1@0.15  {100*st.mean(d):>+6.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  "
              f"{'SIG' if (lo>0 or hi<0) else 'n.s.'}   ({res[k][1]:.0f} vs {res['top1@0.15'][1]:.0f} tok)")

    print("\n=== PEAK-GATED variants ===")
    pk = {}
    for r in rows:
        pk.setdefault(r["instance"], r["peak"])
    vals = sorted(pk.values(), reverse=True)
    tau = vals[max(0, int(FIRE * len(vals)) - 1)]
    for a in ("top1@0.15", "multi4", "multi4_600"):
        if a not in res:
            continue
        v, tk = [], []
        for r in rows:
            if pk[r["instance"]] >= tau:
                v.append(hit(r, a) if conf(r, a) > conf(r, "uniform@300") else hit(r, "uniform@300"))
                tk.append(r["cost"][a]["tokens"])
            else:
                v.append(hit(r, "uniform@300"))
                tk.append(r["cost"]["uniform@300"]["tokens"])
        t = st.mean(tk); b = bar(t)
        lo, hi = boot([x - b for x in v])
        print(f"  {'gated ' + a:<20}{t:>8.0f}{100*st.mean(v):>7.1f}%{100*b:>7.1f}%"
              f"{100*(st.mean(v)-b):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
              f"{'  SIG' if lo > 0 else ''}")

    print("\n=== CIRCULAR EVAL (the benchmark's own metric) ===")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["instance"]].append(i)
    full = [k for k, v in by.items() if len(v) >= 4]
    for a in arms:
        c = st.mean([min(hit(rows[i], a) for i in by[k]) for k in full])
        print(f"  {a:<14}{100*c:>7.1f}%")

    print("\n" + "=" * 74)
    m4 = res.get("multi4")
    if m4 and m4[4] > 0:
        print(f"  => MULTI-CROP TRANSFERS TO 4K: multi4 {100*m4[3]:+.1f}pp "
              f"CI[{100*m4[4]:+.1f},{100*m4[5]:+.1f}] vs the budget axis, n={n}. SIGNIFICANT.")
        print("     The method is not V*Bench-bound: it wins at 4032px in the qualifying regime,")
        print("     where every previous version of it failed.")
    elif m4 and m4[3] > 0:
        print(f"  => positive at 4K ({100*m4[3]:+.1f}pp CI[{100*m4[4]:+.1f},{100*m4[5]:+.1f}]) but")
        print("     not significant. Directional transfer; report as such.")
    else:
        print("  => multi-crop does NOT transfer to 4K. The V*Bench result is scale-bound and the")
        print("     paper says so.")


if __name__ == "__main__":
    main()
