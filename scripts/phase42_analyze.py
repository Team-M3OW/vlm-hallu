"""
Phase 42 analyzer: do the MECHANISM and the METHOD hold on a second architecture?

Nothing is fitted. Every hyper-parameter came from Qwen3-VL. This file scores Qwen2-VL-7B against
the three claims that are currently single-model, and prints the Qwen3-VL value beside each so the
comparison is explicit rather than implied.

  CLAIM 1 (§4.3)  the sink-masked attention peak localizes:      attn > rand
  CLAIM 2 (§3.3)  coverage mediates the SIGN of the effect:      dose-response, crossing zero
  CLAIM 3 (§5.3)  the confidence gate is a COVERAGE DETECTOR:    AUROC(conf(attn)) >> 0.5

Each has a pre-registered failure reading, stated at the point of use. Coverage uses the GT box and
is EXPLANATORY only -- the proposal never sees a box, and no hyper-parameter is chosen on it.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase42_qwen2vl_method.jsonl"
REF = {"attn_rand": 40.0, "cov0": -15.6, "cov100": +37.1, "auroc": 0.835, "gate": +12.0}
W = 0.15


def cov(b, cx, cy, W):
    gx0, gy0, gx1, gy1 = b
    x0, x1, y0, y1 = cx - W / 2, cx + W / 2, cy - W / 2, cy + W / 2
    if x0 < 0: x0, x1 = 0.0, W
    if y0 < 0: y0, y1 = 0.0, W
    if x1 > 1: x0, x1 = 1 - W, 1.0
    if y1 > 1: y0, y1 = 1 - W, 1.0
    ix = max(0.0, min(gx1, x1) - max(gx0, x0))
    iy = max(0.0, min(gy1, y1) - max(gy0, y0))
    return ix * iy / max((gx1 - gx0) * (gy1 - gy0), 1e-9)


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def auroc(y, s):
    p = sorted(zip(s, y))
    r, i = {}, 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[j + 1][0] == p[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    sr = sum(r[k] for k in range(len(p)) if p[k][1] == 1)
    return (sr - pos * (pos + 1) / 2) / (pos * neg)


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 40:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    cA = lambda r: max(r["probs"][f"attn@{W}"])
    cU = lambda r: max(r["probs"]["uniform"])
    for r in rows:
        r["_cov"] = cov(r["gt_box_frac"], *r["peak_frac"], W)
        r["_rcov"] = cov(r["gt_box_frac"], *r["rand_frac"], W)
    print(f"Qwen2-VL-7B, V*Bench, n = {n}.  Every hyper-parameter transferred from Qwen3-VL-2B.")

    print("\n=== BUDGET GATE ===")
    arms = list(rows[0]["realized_tokens"])
    med = {a: st.median([r["realized_tokens"][a] for r in rows]) for a in arms}
    for a in arms:
        print(f"  {a:12}{med[a]:6.0f} tok")
    sp = (max(med.values()) - min(med.values())) / min(med.values())
    print(f"  spread {100*sp:.1f}%  -> {'OK' if sp < .10 else '!! VOID'}")
    if sp >= .10:
        return

    u = [hit(r, "uniform") for r in rows]
    print("\n=== ACCURACY (chance 25%) ===")
    print(f"  {'arm':<12}{'acc':>8}{'vs uniform':>13}{'95% CI':>18}")
    for a in arms:
        v = [hit(r, a) for r in rows]
        d = [x - y for x, y in zip(v, u)]
        lo, hi = boot(d)
        print(f"  {a:<12}{100*st.mean(v):>7.1f}%{100*st.mean(d):>+12.1f}pp   [{100*lo:+.1f},{100*hi:+.1f}]")

    print("\n=== CLAIM 1: does the sink-masked localizer TRANSFER? (attn - rand) ===")
    print("  FAIL READING: attn ~= rand means the localizer is Qwen3-VL-specific.")
    for w in (0.15, 0.25):
        a = [hit(r, f"attn@{w}") for r in rows]
        b = [hit(r, f"rand@{w}") for r in rows]
        d = [x - y for x, y in zip(a, b)]
        lo, hi = boot(d)
        tag = "TRANSFERS" if lo > 0 else "does NOT transfer"
        print(f"  W={w}: {100*st.mean(d):+.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  {tag}"
              f"   (Qwen3-VL: {REF['attn_rand']:+.1f}pp)")
    print("  mean coverage achieved by the attention peak: "
          f"{100*st.mean([r['_cov'] for r in rows]):.1f}%   by a RANDOM centre: "
          f"{100*st.mean([r['_rcov'] for r in rows]):.1f}%")

    print("\n=== CLAIM 2: is COVERAGE the mediator here too? ===")
    print("  FAIL READING: no dose-response means §3 is about one VLM, not about VLMs.")
    print(f"  {'coverage bin':<16}{'n':>4}{'uniform':>9}{'attn@0.15':>11}{'delta':>8}{'95% CI':>18}")
    for lo_, hi_, nm in [(-.001, .001, "0% (missed)"), (.001, .25, "0-25%"),
                         (.25, .999, "25-100%"), (.999, 1.01, "100% (full)")]:
        g = [r for r in rows if lo_ < r["_cov"] <= hi_]
        if len(g) < 8:
            print(f"  {nm:<16}{len(g):>4}   too few")
            continue
        uu = [hit(r, "uniform") for r in g]
        aa = [hit(r, f"attn@{W}") for r in g]
        d = [x - y for x, y in zip(aa, uu)]
        l, h = boot(d)
        print(f"  {nm:<16}{len(g):>4}{100*st.mean(uu):>8.1f}%{100*st.mean(aa):>10.1f}%"
              f"{100*st.mean(d):>+8.1f}   [{100*l:+.1f},{100*h:+.1f}]")
    print(f"  (Qwen3-VL: missed {REF['cov0']:+.1f}pp, full {REF['cov100']:+.1f}pp)")

    print("\n=== CLAIM 3: is the confidence gate a COVERAGE DETECTOR here too? ===")
    print("  FAIL READING: AUROC ~0.5 means §5.3's explanation of the method is local.")
    y = [1 if r["_cov"] > .001 else 0 for r in rows]
    for nm, f in [("conf(attn crop)", cA), ("conf(uniform) [control]", cU)]:
        print(f"  {nm:<26} AUROC {auroc(y, [f(r) for r in rows]):.3f}"
              + (f"   (Qwen3-VL: {REF['auroc']:.3f})" if nm.startswith("conf(attn") else ""))
    print(f"  covered {sum(y)} / missed {len(y)-sum(y)}")

    print("\n=== THE DEPLOYED GATE, unmodified ===")
    gate = [hit(r, f"attn@{W}") if cA(r) > cU(r) else hit(r, "uniform") for r in rows]
    orc = [hit(r, f"attn@{W}") if r["_cov"] > .001 else hit(r, "uniform") for r in rows]
    d = [x - y for x, y in zip(gate, u)]
    lo, hi = boot(d)
    capt = (st.mean(gate) - st.mean(u)) / max(st.mean(orc) - st.mean(u), 1e-9)
    print(f"  conf gate {100*st.mean(gate):.1f}%  ({100*st.mean(d):+.1f}pp "
          f"CI[{100*lo:+.1f},{100*hi:+.1f}])   (Qwen3-VL: {REF['gate']:+.1f}pp)")
    print(f"  its coverage ceiling {100*(st.mean(orc)-st.mean(u)):+.1f}pp  -> captures {100*capt:.0f}%")

    print("\n" + "=" * 74)
    print("GENERALITY VERDICT")
    print("=" * 74)
    a15 = [hit(r, "attn@0.15") for r in rows]
    r15 = [hit(r, "rand@0.15") for r in rows]
    l1, _ = boot([x - y for x, y in zip(a15, r15)])
    c1 = l1 > 0
    miss = [r for r in rows if r["_cov"] <= .001]
    full = [r for r in rows if r["_cov"] > .999]
    c2 = (len(miss) >= 8 and len(full) >= 8 and
          st.mean([hit(r, f"attn@{W}") - hit(r, "uniform") for r in full]) >
          st.mean([hit(r, f"attn@{W}") - hit(r, "uniform") for r in miss]))
    c3 = auroc(y, [cA(r) for r in rows]) > 0.65
    for nm, okk in [("CLAIM 1 localizer transfers", c1),
                    ("CLAIM 2 coverage mediates", c2),
                    ("CLAIM 3 gate detects coverage", c3)]:
        print(f"  {nm:<34}{'HOLDS' if okk else 'DOES NOT HOLD'}")
    if c1 and c2 and c3:
        print("\n  => mechanism and method are NOT Qwen3-VL-specific. §3/§4/§5 may be stated")
        print("     across architectures, with per-model numbers reported.")
    else:
        print("\n  => at least one claim is architecture-specific. Report which, and scope §3-§5")
        print("     to the models where it holds.")


if __name__ == "__main__":
    main()
