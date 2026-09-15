"""
Phase 60 analyzer: does the logit lens falsify §10's "the information is absent" claim?

  uniform_best_layer > uniform_final  -> a READ-OUT failure: recoverable signal discarded late.
                                         §10 would be wrong and a best-layer read-out is a free win.
  uniform flat at chance, oracle rises -> the crop ADDS information. §10 confirmed by direct
                                         measurement rather than inferred from intervention nulls.

Reported for all items and, separately, for the SUB-TOKEN stratum (target < 1 merged token at
B0=300), which is where §10's claim is specifically about.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase60_logit_lens.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def curve(rows, key):
    nL = len(rows[0][key])
    return [st.mean(1.0 * (max(range(4), key=lambda i: r[key][li][i]) == r["label"]) for r in rows)
            for li in range(nL)]


def report(rows, name):
    nL = len(rows[0]["uniform"])
    cu, co = curve(rows, "uniform"), curve(rows, "oracle")
    print(f"\n=== {name}  (n={len(rows)}) ===")
    print(f"  {'layer':<7}{'uniform':>10}{'oracle':>10}{'oracle-uniform':>17}")
    for li in range(nL):
        mark = "  <- final" if li == nL - 1 else ""
        if li % 2 == 0 or li == nL - 1:
            print(f"  L{li:<6}{100*cu[li]:>9.1f}%{100*co[li]:>9.1f}%{100*(co[li]-cu[li]):>+16.1f}{mark}")
    bu = max(range(nL), key=lambda i: cu[i])
    bo = max(range(nL), key=lambda i: co[i])
    print(f"\n  uniform: final {100*cu[-1]:.1f}%   best L{bu} {100*cu[bu]:.1f}%"
          f"   gain from best-layer read-out {100*(cu[bu]-cu[-1]):+.1f}pp")
    print(f"  oracle : final {100*co[-1]:.1f}%   best L{bo} {100*co[bo]:.1f}%")
    # per-item: is the best layer better than the final layer, out of sample?
    d = []
    for r in rows:
        f = 1.0 * (max(range(4), key=lambda i: r["uniform"][-1][i]) == r["label"])
        b = 1.0 * (max(range(4), key=lambda i: r["uniform"][bu][i]) == r["label"])
        d.append(b - f)
    lo, hi = boot(d)
    print(f"  best-layer minus final-layer (uniform): {100*st.mean(d):+.1f}pp "
          f"CI[{100*lo:+.1f},{100*hi:+.1f}]  {'SIG' if (lo>0 or hi<0) else 'n.s.'}")
    print("  (the best layer is CHOSEN on these same items, so this is an OPTIMISTIC upper bound)")
    first = next((li for li in range(nL) if co[li] - cu[li] > 0.10), None)
    print(f"  first layer where oracle exceeds uniform by >10pp: "
          f"{'L'+str(first) if first is not None else 'never'}")
    return cu, co


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 50:
        print(f"only {len(rows)} rows; wait")
        return
    print(f"LOGIT LENS: ABCD decoded from the last token's residual at every layer.")
    print(f"n = {len(rows)}; chance = 25%")
    report(rows, "ALL ITEMS")
    sub = [r for r in rows if r["tokens_on_target"] < 1.0]
    if len(sub) >= 30:
        report(sub, "SUB-TOKEN stratum (target < 1 merged token at B0=300)")
    print("\n" + "=" * 70)
    cu = curve(rows, "uniform")
    bu = max(range(len(cu)), key=lambda i: cu[i])
    if cu[bu] - cu[-1] > 0.02:
        print(f"  => a READ-OUT gap exists: uniform's best layer beats its final layer by "
              f"{100*(cu[bu]-cu[-1]):+.1f}pp.")
        print("     §10's 'information is absent' needs qualifying, and a best-layer read-out is")
        print("     worth testing as a free, one-pass method.")
    else:
        print("  => NO read-out gap: uniform's final layer is already its best. The signal the crop")
        print("     supplies is not present at ANY layer under uniform, so §10's conclusion is")
        print("     confirmed by direct measurement, not merely inferred from intervention nulls.")


if __name__ == "__main__":
    main()
