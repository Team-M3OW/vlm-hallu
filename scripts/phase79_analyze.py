"""
Phase 79 analyzer: is DCR's gain the RESTORATION OF ANSWER FORMATION?

MECHANISM UNDER TEST, fixed before the data existed
---------------------------------------------------
Phase 66: under uniform encoding the answer is decodable at NO layer; under an oracle crop it
appears ABRUPTLY at ~L22. If DCR works by supplying the information that layer converts, then:

  P1  uniform   -> flat, low at every layer, and "ever argmax" near chance
  P2  oracle    -> a sharp step late in the stack
  P3  head      -> a step too, intermediate between uniform and oracle
  P4  THE CONTROL: split the head arm by whether its own window COVERS.
        covered -> trajectory tracks ORACLE
        missed  -> trajectory tracks UNIFORM
      If P4 does not separate, the gain is NOT restored answer formation and the mechanism is
      wrong however good P1-P3 look. P4 is the whole experiment.

"ever argmax" = the correct option is the argmax at SOME layer. On a 4-way MCQ with 28 layers this
is inflated by chance alone (Phase 66 measured 72.2% for localised-but-wrong items against 68.1% for
missed windows -- i.e. chance), so it is reported WITH its uniform-arm reference, never alone.

LENS INTEGRITY: `lens_drift` is the L1 gap between the naive final-layer path and model.logits. It
must be NON-ZERO -- that gap is the SS12D double-normalisation bug, and a zero would mean the
correct path is not actually being used.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase79_lens_mechanism.jsonl"


def boot(a, b, n=6000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n)]


def main():
    R = [json.loads(l) for l in open(PATH)]
    if len(R) < 40:
        print(f"only {len(R)} rows; wait")
        return
    ARMS = list(R[0]["traj"].keys())
    nL = len(R[0]["traj"][ARMS[0]])
    print(f"n = {len(R)}, {nL} layers")
    drift = st.mean([r["lens_drift"]["uniform@300"] for r in R])
    print(f"lens integrity: naive-vs-true final-layer L1 drift = {drift:.4f} "
          f"{'OK (the SS12D bug is being avoided)' if drift > 1e-3 else '!! ZERO -- correct path NOT in use'}")

    acc = lambda r, a, i: 1.0 * (max(range(4), key=lambda j: r["traj"][a][i][j]) == r["label"])
    print(f"\n=== PER-LAYER ACCURACY ===")
    print("  layer " + "".join(f"{a:>14}" for a in ARMS))
    for i in range(nL):
        if i < nL - 12 and i % 4:
            continue
        row = "".join(f"{100*st.mean([acc(r,a,i) for r in R]):13.1f}%" for a in ARMS)
        print(f"  L{i:<4d}" + row)

    print("\n=== THE STEP: biggest single-layer jump, and where ===")
    for a in ARMS:
        cur = [st.mean([acc(r, a, i) for r in R]) for i in range(nL)]
        d = [(cur[i] - cur[i - 1], i) for i in range(1, nL)]
        j = max(d)
        print(f"  {a:14} final {100*cur[-1]:5.1f}%   biggest jump {100*j[0]:+5.1f}pp at L{j[1]}"
              f"   (max over layers {100*max(cur):5.1f}%)")

    print("\n=== 'EVER ARGMAX' (correct at SOME layer) -- read against the uniform reference ===")
    for a in ARMS:
        e = st.mean([1.0 * any(acc(r, a, i) for i in range(nL)) for r in R])
        print(f"  {a:14}{100*e:6.1f}%")

    print("\n" + "=" * 72)
    print("P4 -- THE CONTROL: split the HEAD arm by whether its own window covers")
    print("=" * 72)
    cov = [r for r in R if r["head_cov"] >= .5]
    mis = [r for r in R if r["head_cov"] < .5]
    print(f"  {'stratum':22}{'head':>9}{'uniform':>10}{'oracle':>10}{'head-uniform':>15}")
    for nm, S in [("head COVERS", cov), ("head MISSES", mis)]:
        if not S:
            continue
        h = [acc(r, "head@0.15", nL - 1) for r in S]
        u = [acc(r, "uniform@300", nL - 1) for r in S]
        o = [acc(r, "oracle@0.15", nL - 1) for r in S]
        lo, hi = boot(h, u)
        print(f"  {nm:22}{100*st.mean(h):8.1f}%{100*st.mean(u):9.1f}%{100*st.mean(o):9.1f}%"
              f"{100*(st.mean(h)-st.mean(u)):+11.1f}pp CI[{lo:+.1f},{hi:+.1f}]")
    if cov and mis:
        hc = st.mean([acc(r, "head@0.15", nL - 1) for r in cov])
        uc = st.mean([acc(r, "uniform@300", nL - 1) for r in cov])
        hm = st.mean([acc(r, "head@0.15", nL - 1) for r in mis])
        um = st.mean([acc(r, "uniform@300", nL - 1) for r in mis])
        print(f"\n  gain where it covers {100*(hc-uc):+.1f}pp   "
              f"where it misses {100*(hm-um):+.1f}pp")
        if (hc - uc) - (hm - um) > 0.10:
            print("\n  => MECHANISM CONFIRMED. DCR's gain is restored answer formation: the step")
            print("     appears only where the window actually supplies the evidence.")
        else:
            print("\n  => MECHANISM NOT SUPPORTED. The gain does not depend on coverage, so it is")
            print("     not the restoration of answer formation. Report that.")

    print("\n=== where in the stack the arms SEPARATE (head vs uniform, per layer) ===")
    for i in range(max(0, nL - 14), nL):
        h = [acc(r, "head@0.15", i) for r in R]
        u = [acc(r, "uniform@300", i) for r in R]
        d = 100 * (st.mean(h) - st.mean(u))
        bar = "#" * max(0, int(d))
        print(f"  L{i:<3d}{d:+6.1f}pp {bar}")


if __name__ == "__main__":
    main()
