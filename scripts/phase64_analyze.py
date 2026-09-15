"""
Phase 64 analyzer: which late-layer component destroys the answer, and can ablating it beat
truncation?

The key column is EVIDENCE-DEPENDENCE. §12B showed the late layers help when visual evidence is
strong (oracle + resolvable: final beats L24 by 1.8pp) and hurt when it is weak. Truncation gives up
the helpful part. A component ablation that helps on weak evidence WITHOUT hurting strong evidence
is strictly better than truncation; one that hurts both is not.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase64_component_ablation.jsonl"


def boot(d, n=20000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 50:
        print(f"only {len(rows)} rows; wait")
        return
    hit = lambda r, c, a: 1.0 * (max(range(4), key=lambda i: r["probs"][c][a][i]) == r["label"])
    arms = [a for a in rows[0]["probs"]["uniform"] if a != "baseline"]
    sub = [r for r in rows if r["tokens_on_target"] < 1.0]
    res_ = [r for r in rows if r["tokens_on_target"] >= 1.0]
    print(f"n = {len(rows)} (sub-token {len(sub)}, resolvable {len(res_)})")
    print("All arms are ONE pass at identical tokens -- ablation changes no compute.\n")

    for cond in ("uniform", "oracle"):
        print(f"=== condition: {cond} ===")
        base_all = [hit(r, cond, "baseline") for r in rows]
        print(f"  baseline {100*st.mean(base_all):.1f}%")
        print(f"  {'ablation':<14}{'all Δ':>9}{'95% CI':>17}{'sub-tok Δ':>12}{'resolvable Δ':>15}")
        for a in arms:
            d = [hit(r, cond, a) - hit(r, cond, "baseline") for r in rows]
            lo, hi = boot(d)
            ds = ([hit(r, cond, a) - hit(r, cond, "baseline") for r in sub] if len(sub) > 20
                  else [0])
            dr = ([hit(r, cond, a) - hit(r, cond, "baseline") for r in res_] if len(res_) > 20
                  else [0])
            print(f"  {a:<14}{100*st.mean(d):>+8.1f}   [{100*lo:+.1f},{100*hi:+.1f}]"
                  f"{100*st.mean(ds):>+11.1f}{100*st.mean(dr):>+14.1f}"
                  f"{'  SIG' if (lo>0 or hi<0) else ''}")
        print()

    print("=" * 74)
    print("THE TEST THAT MATTERS: helps weak evidence WITHOUT hurting strong evidence?")
    print("=" * 74)
    print(f"  {'ablation':<14}{'uniform/sub-token':>19}{'oracle/resolvable':>20}{'verdict':>26}")
    best = None
    for a in arms:
        dw = [hit(r, "uniform", a) - hit(r, "uniform", "baseline") for r in sub]
        ds_ = [hit(r, "oracle", a) - hit(r, "oracle", "baseline") for r in res_]
        lw, hw = boot(dw)
        w, s_ = st.mean(dw), (st.mean(ds_) if len(res_) > 20 else 0.0)
        v = ("strictly better than truncation" if (lw > 0 and s_ >= -0.005)
             else "helps weak, costs strong" if lw > 0
             else "no gain on weak evidence")
        if lw > 0 and (best is None or w > best[0]):
            best = (w, a, s_, lw, hw)
        print(f"  {a:<14}{100*w:>+18.1f}{100*s_:>+19.1f}{v:>26}")
    print()
    if best:
        print(f"  => best: {best[1]}  weak-evidence {100*best[0]:+.1f}pp "
              f"CI[{100*best[3]:+.1f},{100*best[4]:+.1f}], strong-evidence {100*best[2]:+.1f}pp")
        if best[2] >= -0.005:
            print("     This ablation is STRICTLY BETTER than truncation: it recovers the harmful")
            print("     part of the late layers while keeping the helpful part.")
        else:
            print("     It still costs accuracy where evidence is strong, so it is not strictly")
            print("     better than truncation -- report the trade-off.")
    else:
        print("  => NO component ablation helps on weak evidence. The damage is distributed across")
        print("     attention and MLP, truncation is the only instrument available, and nothing")
        print("     finer is claimed.")


if __name__ == "__main__":
    main()
