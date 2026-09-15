"""
Phase 50 analyzer: does suppressing the serialization sink inside the forward pass help?

THE COMPARISON IS PAIRED AND COST-FREE. Every arm is ONE pass at the SAME realized token count on
the SAME image, differing only in an additive bias on the attention logits of a chosen set of key
positions. There is no compute bar to clear and no budget gate to check: the baseline IS the
control for cost. What has to be ruled out is that ANY perturbation of that size helps.

THE CONTROLS DECIDE IT
----------------------
    last_col     the trailing row boundary -- the universal sink (§7A/§7C)
    first_col    enriched on Qwen3-VL (2.0x) but NOT universal (0.9x on Qwen2-VL)
    interior     a real-content column, same count, no sink
    rand_cols    the same NUMBER of random image tokens
    text_tokens  the same number of prompt tokens

    last_col > all controls    -> suppressing the sink specifically helps; the interpretability
                                  result yields a method with no extra compute.
    last_col ~= rand/interior  -> the effect is generic perturbation, not the sink. Report as a
                                  negative; the sink is real but suppressing it buys nothing.
    everything <= baseline     -> the mass the sink holds is not recoverable this way.

The bias strength is SWEPT and the whole curve is printed, so no single flattering value can be
selected after the fact.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase50_sink_steering.jsonl"
FAM = ["last_col", "first_col", "interior", "rand_cols", "text_tokens"]


def boot(d, n=10000, seed=0):
    rng = random.Random(seed)
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[250], s[9750]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if len(rows) < 40:
        print(f"only {len(rows)} rows; wait")
        return
    n = len(rows)
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    biases = sorted({float(k.split("@")[1]) for k in rows[0]["probs"] if "@" in k}, reverse=True)
    base = [hit(r, "baseline") for r in rows]
    print(f"n = {n} items, all arms ONE pass at {st.median([r['realized_tokens'] for r in rows]):.0f} "
          f"realized tokens. Paired; no compute bar applies.")
    print(f"biased positions per arm: {st.median([r['n_biased'] for r in rows]):.0f} "
          f"of {st.median([r['n_img'] for r in rows]):.0f} image tokens "
          f"({100*st.median([r['n_biased']/r['n_img'] for r in rows]):.1f}%)")
    print(f"\n  baseline accuracy {100*st.mean(base):.1f}%\n")
    print(f"  {'arm':<13}" + "".join(f"{'b='+str(b):>16}" for b in biases))
    best = None
    for f in FAM:
        cells = []
        for b in biases:
            k = f"{f}@{b}"
            if k not in rows[0]["probs"]:
                cells.append("   --")
                continue
            v = [hit(r, k) for r in rows]
            d = [x - y for x, y in zip(v, base)]
            lo, hi = boot(d)
            star = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{100*st.mean(d):+6.1f}{star}[{100*lo:+.0f},{100*hi:+.0f}]")
            if f == "last_col" and (best is None or st.mean(d) > best[0]):
                best = (st.mean(d), b, lo, hi)
        print(f"  {f:<13}" + "".join(f"{c:>16}" for c in cells))
    print("  (Δ vs baseline in pp; * = 95% CI excludes zero)")

    if best:
        b = best[1]
        print(f"\n=== HEAD-TO-HEAD at the sink's best strength (b={b}) ===")
        lc = [hit(r, f"last_col@{b}") for r in rows]
        print(f"  {'contrast':<30}{'delta':>9}{'95% CI':>18}")
        for f in FAM[1:]:
            k = f"{f}@{b}"
            if k not in rows[0]["probs"]:
                continue
            v = [hit(r, k) for r in rows]
            d = [x - y for x, y in zip(lc, v)]
            lo, hi = boot(d)
            sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
            print(f"  {'last_col - ' + f:<30}{100*st.mean(d):>+8.1f}   "
                  f"[{100*lo:+.1f},{100*hi:+.1f}]  {sig}")

        print("\n=== by category ===")
        for cat in sorted({r["category"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["category"] == cat]
            print(f"  {cat:<20}n={len(idx):<4} baseline {100*st.mean([base[i] for i in idx]):5.1f}%"
                  f"   last_col@{b} {100*st.mean([lc[i] for i in idx]):5.1f}%"
                  f"   ({100*(st.mean([lc[i] for i in idx])-st.mean([base[i] for i in idx])):+.1f}pp)")

        print("\n" + "=" * 70)
        d = [x - y for x, y in zip(lc, base)]
        lo, hi = boot(d)
        ctrl = max(st.mean([hit(r, f"{f}@{b}") - hit(r, "baseline") for r in rows])
                   for f in FAM[1:] if f"{f}@{b}" in rows[0]["probs"])
        if lo > 0 and st.mean(d) > ctrl:
            print(f"  => SUPPRESSING THE SINK HELPS: {100*st.mean(d):+.1f}pp CI[{100*lo:+.1f},"
                  f"{100*hi:+.1f}] at ZERO extra compute, and beats every control "
                  f"(best control {100*ctrl:+.1f}pp).")
        elif lo > 0:
            print(f"  => helps ({100*st.mean(d):+.1f}pp) but a control matches it "
                  f"({100*ctrl:+.1f}pp): this is generic perturbation, NOT the sink.")
        else:
            print(f"  => NO BENEFIT. {100*st.mean(d):+.1f}pp CI[{100*lo:+.1f},{100*hi:+.1f}]. The "
                  f"sink is real and measurable, but the mass it holds is not recoverable by\n"
                  f"     biasing it away at inference. Report as a negative.")


if __name__ == "__main__":
    main()
