"""
Phase 31 analyzer. Written before the data existed so the selection rule cannot drift.

THE HEADLINE IS THE CROSS-CATEGORY HELD-OUT NUMBER, and nothing else.
    W is chosen on `direct_attributes` and applied to `relative_position`;
    W is chosen on `relative_position`  and applied to `direct_attributes`;
    every item is then scored under a W picked without seeing that item's category, and the
    headline is the pooled accuracy over all 191.
The full W sweep is printed as sensitivity and is explicitly IN-SAMPLE -- its best cell is not a
result and must never be quoted as the method's accuracy.

Bars, fixed in PLAN.md before the experiment was written:
    56.5%  uniform@300     -- beats doing nothing
    66.0%  uniform@600     -- EARNS THE SECOND FORWARD PASS (the honest floor)
    72.9%  44% of oracle   -- beats the published training-free proposer
    93.7%  oracle ceiling
A number between 56.5 and 66.0 is a NEGATIVE: the second pass was better spent on resolution.
"""
import json
import statistics as st
from collections import defaultdict

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase31_proposer_accuracy.jsonl"
UNIFORM_600 = 66.0      # Phase 27, compute-matched control
BAR_44 = 72.9           # 44% oracle capture against Phase 27's uniform@300 / crop_only@300


def boot(a, b, n=10000, seed=0):
    """Paired bootstrap CI on the difference of two 0/1 vectors over the same items."""
    import random
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(.025 * n)], s[int(.975 * n)]


def main():
    rows = [json.loads(l) for l in open(PATH)]
    if not rows:
        print("no rows yet")
        return
    print(f"n = {len(rows)}")
    cats = defaultdict(int)
    for r in rows:
        cats[r["category"]] += 1
    print("  categories:", dict(cats))

    arms = sorted(rows[0]["pred"].keys())
    hit = {a: [1.0 * (r["pred"][a] == r["label"]) for r in rows] for a in arms}
    tokn = {a: [r["realized_tokens"][a] for r in rows] for a in arms}

    print("\n=== BUDGET GATE (all arms must realize ~B0=300) ===")
    voided = set()
    for a in arms:
        med = st.median(tokn[a])
        off = abs(med - 300) / 300
        flag = "OK" if off < .10 else "!! VOID"
        if off >= .10:
            voided.add(a)
        print(f"  {a:14} median {med:6.0f} tok  {100*off:5.1f}% off B0   {flag}")

    print("\n=== raw accuracies (IN-SAMPLE sweep -- sensitivity only, NOT the headline) ===")
    print(f"  {'arm':14} {'acc':>7}  {'vs uniform':>12}")
    ub = hit["uniform"]
    for a in arms:
        d = 100 * (sum(hit[a]) - sum(ub)) / len(ub)
        v = "  VOID" if a in voided else ""
        print(f"  {a:14} {100*st.mean(hit[a]):6.1f}% {d:+11.1f}pp{v}")

    wins = sorted({float(a.split("@")[1]) for a in arms if a.startswith("attn@")})
    print("\n=== CROSS-CATEGORY HELD-OUT (THE HEADLINE) ===")
    cl = sorted(cats)
    if len(cl) < 2:
        # Caught by the analyzer sweep on partial data: V*Bench is ordered by category, so early
        # rows are all `direct_attributes` and the holdout set is EMPTY. The headline is not
        # computable without both categories -- say so rather than crashing or, worse, silently
        # falling back to in-sample selection, which is the exact leak this design exists to avoid.
        print(f"  ONLY ONE CATEGORY PRESENT ({cl}). The cross-category holdout needs both.")
        print("  No headline number. In-sample sweep above is sensitivity only. "
              "Re-run when all 191 rows exist.")
        return
    chosen = {}
    for held in cl:
        other = [r for r in rows if r["category"] != held]
        if not other:
            print(f"  no holdout data for '{held}' -- skipping headline")
            return
        best, bw = -1, None
        for w in wins:
            acc = st.mean([1.0 * (r["pred"][f"attn@{w}"] == r["label"]) for r in other])
            if acc > best:
                best, bw = acc, w
        chosen[held] = bw
        sel = "+".join(c for c in cl if c != held)
        print(f"  W for '{held}' chosen on '{sel}': W={bw}  (selection acc there {100*best:.1f}%)")

    ho, hu, ho_r = [], [], []
    for r in rows:
        w = chosen[r["category"]]
        ho.append(1.0 * (r["pred"][f"attn@{w}"] == r["label"]))
        ho_r.append(1.0 * (r["pred"][f"rand@{w}"] == r["label"]))
        hu.append(1.0 * (r["pred"]["uniform"] == r["label"]))
    acc = 100 * st.mean(ho)
    accr = 100 * st.mean(ho_r)
    accu = 100 * st.mean(hu)
    acco = 100 * st.mean(hit["oracle"])
    lo, hi = boot(ho, hu)
    lo2, hi2 = boot(ho, ho_r)
    print(f"\n  HELD-OUT attn_prop : {acc:.1f}%")
    print(f"  same-window random : {accr:.1f}%   (the placement null)")
    print(f"  uniform@300        : {accu:.1f}%")
    print(f"  oracle@300         : {acco:.1f}%")
    print(f"\n  attn - uniform : {acc-accu:+.1f}pp  CI [{100*lo:+.1f}, {100*hi:+.1f}]")
    print(f"  attn - random  : {acc-accr:+.1f}pp  CI [{100*lo2:+.1f}, {100*hi2:+.1f}]"
          f"   <- is the LOCALIZER doing the work, or just magnification?")

    cap = 100 * (acc - accu) / max(acco - accu, 1e-9)
    print(f"\n  ORACLE CAPTURE : {cap:.1f}%   (published training-free proposer: 31-44%)")
    print("\n=== VERDICT against pre-registered bars ===")
    for name, bar in [("uniform@300 (do nothing)", accu),
                      ("uniform@600 (compute-matched)", UNIFORM_600),
                      ("44% oracle capture (published)", BAR_44),
                      ("oracle ceiling", acco)]:
        print(f"  {'PASS' if acc > bar else 'FAIL'}  {acc:.1f}% vs {bar:.1f}%  -- {name}")
    if acc <= accu:
        print("\n  => NEGATIVE. The proposer does not beat doing nothing.")
    elif acc <= UNIFORM_600:
        print("\n  => NEGATIVE. Beats the baseline but LOSES to spending the same two-pass compute")
        print("     on plain resolution. The second forward pass is not earned.")
    elif acc <= BAR_44:
        print("\n  => PARTIAL. Earns its compute but does not beat the published training-free")
        print("     proposer. Report as a bounded result, not a method win.")
    else:
        print("\n  => CLEARS THE BAR. Training-free, data-free, budget-preserving, and above the")
        print("     published training-free proposer's oracle capture.")
    if lo2 <= 0:
        print("\n  !! placement CI includes zero: the gain is NOT attributable to the localizer.")
        print("     Whatever it achieved, a random window of the same size did too.")

    print("\n=== by category (held-out W) ===")
    for c in cl:
        g = [i for i, r in enumerate(rows) if r["category"] == c]
        print(f"  {c:20} n={len(g):3d}  attn {100*st.mean([ho[i] for i in g]):5.1f}%"
              f"  rand {100*st.mean([ho_r[i] for i in g]):5.1f}%"
              f"  uniform {100*st.mean([hu[i] for i in g]):5.1f}%")

    print("\n=== by GT size (does it work where allocation matters most?) ===")
    idx = sorted(range(len(rows)), key=lambda i: rows[i]["gt_area_frac"])
    for k in range(4):
        g = idx[k*len(idx)//4:(k+1)*len(idx)//4]
        print(f"  gt_area {rows[g[0]]['gt_area_frac']:.5f}-{rows[g[-1]]['gt_area_frac']:.5f} "
              f"n={len(g):3d}  attn {100*st.mean([ho[i] for i in g]):5.1f}%"
              f"  uniform {100*st.mean([hu[i] for i in g]):5.1f}%"
              f"  oracle {100*st.mean([hit['oracle'][i] for i in g]):5.1f}%")


if __name__ == "__main__":
    main()
