"""Phase 28 analysis (RQ-C): is the exchange rate a property of the TASK or of one model?

Verdict rules fixed before results:
  * Budget gate first, PER MODEL: `crop_only@B0`'s realized median must be within 10% of B0, and
    each `uniform@B` within 10% of B. Rows failing it are void, not reported.
  * Comparison across models is by RATIO to each model's own B0. Absolute token counts are not
    comparable across tokenizers (Qwen3-VL merges 16px patches; LLaVA-NeXT packs 336px tiles).
  * `crop_random@B0` is the decider inside each model: beating it is what separates placement from
    "any sub-region".
  * SCALE CONFOUND, stated up front: Qwen3-VL here is 2B, the others 7B. If the 2B shows the larger
    gap, capacity is a live alternative explanation and must be reported as such, not explained away.
"""
import json, math, random, statistics as st, collections

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p, d = k/n, 1+z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)


def paired(a, b, nb=4000, seed=19):
    rng = random.Random(seed); n = len(a); d = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in ix)/n - sum(b[i] for i in ix)/n)
    d.sort(); return d[int(.025*nb)], d[int(.975*nb)]


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/phase28_arch_results.jsonl")]
    by = collections.defaultdict(dict)
    for x in rows:
        by[x["model"]][x["question_id_full"]] = x      # dedupe, keep last
    summary = {}
    for m, d in by.items():
        r = list(d.values())
        # only rows carrying the full post-fix arm set
        sweep = sorted({int(k.split("@")[1]) for k in r[-1]["pred"] if k.startswith("uniform@")})
        r = [x for x in r if all(f"uniform@{B}" in x["pred"] for B in sweep)
             and "crop_only@B0" in x["pred"]]
        if len(r) < 20:
            print(f"\n=== {m}: only {len(r)} complete rows -- skipped\n"); continue
        B0 = r[-1]["B0"]
        hit = lambda k: [1 if x["pred"][k] == x["label"] else 0 for x in r]
        tok = lambda k: int(st.median([x["realized_tokens"][k] for x in r]))
        print("\n" + "=" * 78)
        print(f"{m}   n={len(r)}   B0={B0}   crop_only realized {tok('crop_only@B0')}")
        print("=" * 78)
        off = abs(tok("crop_only@B0") - B0) / B0
        print(f"  GATE crop_only: {100*off:.1f}% off B0" + ("  OK" if off < .10 else "  !! VOID"))
        base, br = hit("crop_only@B0"), hit("crop_random@B0")
        lo, hi = paired(base, br)
        print(f"  crop_only {100*sum(base)/len(r):.1f}%   crop_random {100*sum(br)/len(r):.1f}%"
              f"   placement effect {100*(sum(base)-sum(br))/len(r):+.1f}pp CI [{100*lo:+.1f},{100*hi:+.1f}]")
        print(f"\n  {'req':<7}{'realized':<10}{'x B0':<7}{'acc':<9}vs crop_only@B0")
        crossed = None
        for B in sweep:
            k = f"uniform@{B}"
            g = abs(tok(k) - B) / B
            h = hit(k); lo, hi = paired(h, base)
            d_ = 100*(sum(h)-sum(base))/len(r)
            mark = "  !! budget off" if g > .10 else ""
            if lo > 0 and crossed is None:
                crossed = tok(k); mark += "  <-- CROSSES"
            print(f"  {B:<7}{tok(k):<10}{tok(k)/max(1,tok('crop_only@B0')):<7.1f}"
                  f"{100*sum(h)/len(r):<8.1f}%{d_:+7.1f}pp CI [{100*lo:+6.1f},{100*hi:+6.1f}]{mark}")
        mx = tok(f"uniform@{sweep[-1]}")
        ratio = mx / max(1, tok("crop_only@B0"))
        summary[m] = (ratio, crossed, 100*sum(base)/len(r), 100*sum(h)/len(r))
        if crossed:
            print(f"\n  => CROSSES at {crossed} tokens = {crossed/max(1,tok('crop_only@B0')):.1f}x the placed budget")
        elif len(sweep) < 2 or ratio < 1.5:
            # NOT a bound. LLaVA-NeXT's realizable ladder is a single rung (floor 1416, ceiling
            # 2144 -- dynamic range 1.5x, FINDINGS 4Q), so there is no second budget to sweep to.
            # Printing "no crossing up to 1.0x" here would read as a null result about token
            # scaling when it is the ABSENCE OF AN AXIS. Say so instead.
            print(f"\n  => NO AXIS. Realizable ladder is {len(sweep)} rung(s), max {ratio:.1f}x the")
            print("     placed budget. This architecture cannot express a budget sweep, so it")
            print("     contributes a PLACEMENT contrast only and NO exchange-rate row.")
            summary[m] = (ratio, "n/a", 100*sum(base)/len(r), 100*sum(h)/len(r))
            continue
        else:
            print(f"\n  => NO CROSSING up to {mx} tokens = {ratio:.1f}x the placed budget (a BOUND)")

    print("\n" + "=" * 78)
    print("RQ-C VERDICT: is the exchange rate architecture-general?")
    print("=" * 78)
    # Measured floor->ceiling range per architecture (FINDINGS 4Q). The bare multiple tested is
    # NOT comparable across models: onevision's 3.9x is ~67% of everything it can express, while
    # Qwen's 26x is ~2-21% of theirs. Always show the fraction of range actually covered.
    DYN = {"qwen2vl": 1944.0, "qwen3vl": 124.3, "onevision": 5.8, "llavanext": 1.5}
    print(f"  {'model':<13}{'xB0 tested':<12}{'dyn range':<11}{'% of range':<12}"
          f"{'crossed?':<10}{'crop_only':<11}{'uniform@max'}")
    for m, (ratio, crossed, co, un) in summary.items():
        d = DYN.get(m)
        pct = f"{100*ratio/d:.0f}%" if d else "?"
        print(f"  {m:<13}{ratio:<12.1f}{(f'{d:.1f}x' if d else '?'):<11}{pct:<12}"
              f"{(('at '+str(crossed)) if crossed not in (None,'n/a') else ('n/a' if crossed=='n/a' else 'NO')):<10}"
              f"{co:<11.1f}{un:.1f}")
    print("""
  All models no-crossing  => the axis matters across tokenizers that disagree about everything;
                             the protocol is worth adopting generally.
  Mixed                   => report the spread, name which architectures differ, and DO NOT average.
                             The claim narrows to a survey, honestly.
  NOTE the scale confound: Qwen3-VL is 2B, the others 7B.""")


if __name__ == "__main__":
    main()
