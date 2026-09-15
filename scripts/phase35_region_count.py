"""
Phase 35: re-testing the boundary condition. Is it TARGET SIZE or NUMBER OF REGIONS?

WHY THE CURRENT BOUNDARY CLAIM IS IN TROUBLE
--------------------------------------------
§5.5 states the boundary as target size: "allocation pays when the evidence is smaller than the
budget can resolve, and costs when it is not." Two results now contradict that as stated.

  RePOPE (Phase 21, within-benchmark, 400x size range): the predictor works on the POSITIVE half --
  +16.7pp at <0.5 tok decaying monotonically to -0.8pp at >32 tok -- but the negative half is
  untestable because `uniform` saturates at 97.5-99.2% on every resolvable stratum. No crossing can
  be observed where there is no headroom.

  HR-Bench 4k: at B0=300 a 4032x4032 image gives ~233px per merged token, so HR-Bench is DEEPLY
  sub-token. The size predictor therefore says allocation should win there by a wide margin.
  It LOST by 8.8pp. The predictor gets the one benchmark it was invented to explain backwards.

THE ALTERNATIVE ON THE TABLE
----------------------------
HR-Bench's own category split points somewhere else entirely:

    cross  (compare objects in different parts of the image)   gate 48.8%  uniform 51.5%   -2.7pp
    single (one object, one place)                             gate 58.0%  uniform 53.5%   +4.5pp

A crop concentrates one region and discards the rest. That is free when the answer lives in one
region and fatal when the question needs two, IRRESPECTIVE of how small either is. Under this
account the boundary is REGION COUNT, and target size is a correlate that happens to track it on
V*Bench and RePOPE (both single-region tasks) while dissociating on HR-Bench.

The two accounts make OPPOSITE predictions on HR-Bench `single`, which is sub-token AND
single-region: size says win (it is tiny), region-count says win (it is one region) -- agreeing --
and on HR-Bench `cross`, sub-token AND multi-region: size says WIN, region-count says LOSE.
That is the discriminating cell, and it is already measured.

This file scores the raw `always attn@0.15` arm -- NOT the gate -- by category, because the gate
mixes in the relational-keyword rule and would confound the test with its own routing.
"""
import json
import random
import statistics as st
from collections import defaultdict

HR = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33_hrbench_transfer.jsonl"
VS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase32_conditional.jsonl"
RAND = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase33b_rand_control.jsonl"


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def circ(vals, inst):
    by = defaultdict(list)
    for v, i in zip(vals, inst):
        by[i].append(v)
    return [min(v) for k, v in sorted(by.items()) if len(v) >= 4]


def main():
    rows = [json.loads(l) for l in open(HR)]
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    inst = [r["instance"] for r in rows]

    print("=" * 78)
    print("THE DISCRIMINATING CELL: HR-Bench, sub-token throughout, split by REGION COUNT")
    print("=" * 78)
    print("  both categories are deeply sub-token (~233 px per merged token at B0=300),")
    print("  so the SIZE account predicts allocation wins in BOTH. Only region count differs.")
    print()
    print(f"  {'category':<9}{'n':>5}{'uniform':>9}{'attn@0.15':>11}{'delta':>8}"
          f"{'95% CI':>18}{'circular delta':>16}")
    for cat in ("single", "cross"):
        g = [r for r in rows if r["category"] == cat]
        gi = [r["instance"] for r in g]
        u = [hit(r, "uniform") for r in g]
        a = [hit(r, "attn@0.15") for r in g]
        d = [x - y for x, y in zip(a, u)]
        lo, hi = boot(d)
        cu, ca = circ(u, gi), circ(a, gi)
        print(f"  {cat:<9}{len(g):>5}{100*st.mean(u):>8.1f}%{100*st.mean(a):>10.1f}%"
              f"{100*st.mean(d):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
              f"{100*(st.mean(ca)-st.mean(cu)):>+15.1f}pp")

    # placement vs cropping, per category -- is the localizer equally good in both?
    try:
        rr = {r["row_id"]: r for r in (json.loads(l) for l in open(RAND))}
        print("\n=== IS THE LOCALIZER ITSELF WORSE ON `cross`? (attn vs random placement) ===")
        print("  if attn > rand in BOTH categories, the localizer works in both and the `cross`")
        print("  loss is about CROPPING DISCARDING CONTEXT, not about failing to find the target.")
        print(f"  {'category':<9}{'n':>5}{'attn':>8}{'rand':>8}{'attn-rand':>12}{'95% CI':>18}")
        for cat in ("single", "cross"):
            g = [r for r in rows if r["category"] == cat and r["row_id"] in rr]
            if not g:
                continue
            a = [hit(r, "attn@0.15") for r in g]
            b = [1.0 * (max(range(4), key=lambda i: rr[r["row_id"]]["probs"]["rand@0.15"][i])
                        == r["label"]) for r in g]
            d = [x - y for x, y in zip(a, b)]
            lo, hi = boot(d)
            print(f"  {cat:<9}{len(g):>5}{100*st.mean(a):>7.1f}%{100*st.mean(b):>7.1f}%"
                  f"{100*st.mean(d):>+11.1f}   [{100*lo:+.1f},{100*hi:+.1f}]")
    except FileNotFoundError:
        print("\n  (rand control not found; skipping placement contrast)")

    # ---- V*Bench: the same split, where BOTH accounts predict a win
    try:
        vs = [json.loads(l) for l in open(VS)]
        print("\n=== V*BENCH for comparison (both accounts predict a win in both categories) ===")
        k = "probs" if "probs" in vs[0] else None
        cats = sorted({r.get("category") for r in vs})
        arms = sorted(vs[0][k]) if k else []
        base = "uniform" if "uniform" in arms else arms[0]
        cand = [a for a in arms if a.startswith("attn")]
        print(f"  arms: {arms}")
        if k and cand:
            for cat in cats:
                g = [r for r in vs if r.get("category") == cat]
                u = [1.0 * (max(range(4), key=lambda i: r[k][base][i]) == r["label"]) for r in g]
                a = [1.0 * (max(range(4), key=lambda i: r[k][cand[0]][i]) == r["label"]) for r in g]
                print(f"  {str(cat):<22}n={len(g):<4} uniform {100*st.mean(u):5.1f}%"
                      f"  {cand[0]} {100*st.mean(a):5.1f}%  delta {100*(st.mean(a)-st.mean(u)):+.1f}pp")
    except FileNotFoundError:
        print("\n  (V*Bench phase32 file not found)")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("""  attn BEATS uniform on `single` and LOSES on `cross`, with both sub-token:
       -> the boundary is REGION COUNT, not target size. §5.5 must be restated, and the size
          predictor demoted to a correlate that holds on single-region benchmarks.
  attn loses on BOTH:
       -> the HR-Bench failure is about image SCALE, not about region count; neither account is
          supported and the boundary stays a cross-benchmark observation.""")


if __name__ == "__main__":
    main()
