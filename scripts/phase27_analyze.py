"""Phase 27 analysis: the exchange rate between a spatial prior and compute.

Verdict rules fixed before results:
  * The crossing point is only meaningful because `crop_only` is held at ONE budget while `uniform`
    sweeps; both arms are single-image, so the sweep is format-clean at every point (Phase 23's
    confound cannot recur).
  * Realized tokens are reported for every rung. A rung whose realization missed its request is
    labelled by what it REALIZED.
  * If no crossing occurs, the result is a BOUND ("no crossing up to B_max"), never a claim about
    all budgets.
"""
import json, math, random, statistics as st

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def paired(a, b, nb=5000, seed=19):
    rng = random.Random(seed); n = len(a); d = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in ix)/n - sum(b[i] for i in ix)/n)
    d.sort(); return d[int(.025*nb)], d[int(.975*nb)]


def main():
    r = [json.loads(l) for l in open(f"{DATA}/phase27_exchange_results.jsonl")]
    seen, u = set(), []
    for x in r:
        if x["question_id_full"] not in seen:
            seen.add(x["question_id_full"]); u.append(x)
    r = u
    sweep = sorted(int(k.split("@")[1]) for k in r[0]["pred"] if k.startswith("uniform@"))
    hit = lambda k: [1 if x["pred"][k] == x["label"] else 0 for x in r]
    tokmed = lambda k: int(st.median([x["realized_tokens"][k] for x in r]))

    print(f"Qwen3-VL-2B on V*Bench, n={len(r)}")
    print(f"  median target area fraction {st.median([x['bbox_area_frac'] for x in r]):.6f}")

    print("\n" + "="*80)
    print("THE FIXED QUERY-PLACED REFERENCE (single image, budget held at 300)")
    print("="*80)
    ref = {}
    for nm in ["crop_only@300", "crop_random@300", "alloc_query_2img@300"]:
        h = hit(nm); ref[nm] = h
        lo, hi = wilson(sum(h), len(h))
        tag = "   (TWO images -- continuity only, not used for the crossing)" if "2img" in nm else ""
        print(f"  {nm:<22}{100*sum(h)/len(h):6.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]"
              f"  {tokmed(nm)} tok{tag}")

    print("\n" + "="*80)
    print("THE SWEEP: query-INDEPENDENT spending, single image, increasing budget")
    print("="*80)
    base = ref["crop_only@300"]; bt = tokmed("crop_only@300")
    print(f"  {'requested':<11}{'realized':<10}{'accuracy':<12}{'vs crop_only@300 (paired)'}")
    crossed = None
    for B in sweep:
        k = f"uniform@{B}"; h = hit(k); rt = tokmed(k)
        lo, hi = paired(h, base)
        d = 100*(sum(h)-sum(base))/len(h)
        mark = ""
        if lo > 0 and crossed is None:
            crossed = (B, rt); mark = "   <-- CROSSES"
        elif hi < 0:
            mark = "   still behind"
        print(f"  {B:<11}{rt:<10}{100*sum(h)/len(h):>6.1f}%     "
              f"{d:+6.1f}pp CI [{100*lo:+6.1f},{100*hi:+6.1f}]{mark}")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    if crossed:
        print(f"  Query-independent spending CATCHES UP at {crossed[1]} realized tokens.")
        print(f"  Exchange rate: {crossed[1]/bt:.1f}x the tokens to match a {bt}-token placed crop.")
    else:
        mx = tokmed(f"uniform@{sweep[-1]}")
        print(f"  NO CROSSING up to {mx} realized tokens ({mx/bt:.0f}x the placed budget).")
        print(f"  Stated as a BOUND: query-independent budget does not substitute for placement")
        print(f"  in this regime, up to {mx} tokens. NOT a claim about all budgets.")

    print("\n" + "="*80)
    print("SCALING (RQ-B): crossing budget by target size")
    print("="*80)
    qs = sorted(x["bbox_area_frac"] for x in r)
    cuts = [qs[len(qs)//3], qs[2*len(qs)//3]]
    for nm, sel in [("smallest 1/3", lambda x: x["bbox_area_frac"] < cuts[0]),
                    ("middle 1/3", lambda x: cuts[0] <= x["bbox_area_frac"] < cuts[1]),
                    ("largest 1/3", lambda x: x["bbox_area_frac"] >= cuts[1])]:
        ix = [i for i, x in enumerate(r) if sel(x)]
        b = 100*sum(base[i] for i in ix)/len(ix)
        row = f"  {nm:<14} n={len(ix):<4} crop_only@300={b:5.1f}%  "
        cross = None
        for B in sweep:
            h = hit(f"uniform@{B}")
            a = 100*sum(h[i] for i in ix)/len(ix)
            row += f"{tokmed(f'uniform@{B}')}:{a:.0f}%  "
            if cross is None and a >= b:
                cross = tokmed(f"uniform@{B}")
        row += f"|| crosses at {cross if cross else '>'+str(tokmed(f'uniform@{sweep[-1]}'))}"
        print(row)


if __name__ == "__main__":
    main()
