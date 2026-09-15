"""
Phase 52 analyzer: does the residual-stream "crop direction" carry the crop's benefit?

The delta geometry (printed by the runner) already shows a LARGE SHARED COMPONENT: |mean d| /
mean|d_i| = 0.78 / 0.75 / 0.68 / 0.58 at L8 / L14 / L20 / L26, with mean cosine 0.53-0.75. So a
common "allocate to the evidence" direction exists in activation space. The question this file
answers is whether that direction carries the ANSWER-relevant information or merely the generic
statistics of a cropped image (sharper, tighter, different scale), which would move the residual a
long way without making the target legible.

    v_item   inject the item's OWN oracle-minus-uniform displacement. A REACHABILITY TEST, not a
             method: if injecting the exact displacement the crop produces does not recover oracle
             accuracy, the crop's benefit is NOT a linear shift of the last token's residual, and
             no steering vector of this form can work.
    v_mean   the out-of-fold mean direction -- the actual method.
    v_rand   a norm-matched random direction -- controls for "a shift of this size".

All steered arms are ONE pass at the same realized tokens as baseline, so this is paired and free.
Layer and alpha are both swept and the whole grid is printed.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase52_residual_steering.jsonl"


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
    keys = [k for k in rows[0]["probs"] if "@" in k]
    Ls = sorted({int(k.split("@L")[1].split("@")[0]) for k in keys})
    As = sorted({float(k.split("@a")[1]) for k in keys})
    base = [hit(r, "baseline") for r in rows]
    orc = [hit(r, "oracle") for r in rows]
    gap = st.mean(orc) - st.mean(base)
    print(f"n = {n}.  baseline {100*st.mean(base):.1f}%   oracle crop {100*st.mean(orc):.1f}%"
          f"   gap {100*gap:+.1f}pp")
    print("All steered arms: ONE pass, same realized tokens as baseline (paired, no compute bar).\n")

    peak = {}
    for fam in ("v_item", "v_mean", "v_rand"):
        print(f"  {fam}")
        print(f"    {'layer':<7}" + "".join(f"{'a='+str(a):>17}" for a in As))
        for L in Ls:
            cells = []
            for a in As:
                k = f"{fam}@L{L}@a{a}"
                if k not in rows[0]["probs"]:
                    cells.append("   --"); continue
                v = [hit(r, k) for r in rows]
                d = [x - y for x, y in zip(v, base)]
                lo, hi = boot(d)
                star = "*" if (lo > 0 or hi < 0) else " "
                cells.append(f"{100*st.mean(d):+6.1f}{star}[{100*lo:+.0f},{100*hi:+.0f}]")
                if fam not in peak or st.mean(d) > peak[fam][0]:
                    peak[fam] = (st.mean(d), L, a)
            print(f"    L{L:<6}" + "".join(f"{c:>17}" for c in cells))
        print()
    print("  (Δ vs baseline in pp; * = 95% CI excludes zero)")

    print("\n=== THE REACHABILITY TEST ===")
    print("  NOTE: injection at a layer ADJACENT TO THE READOUT is near-tautological. The model has")
    print("  28 layers; adding (h_oracle - h_uniform) at L26 sets the residual EQUAL to h_oracle two")
    print("  layers from the head, so reproducing oracle logits there shows only that the readout")
    print("  depends on the final residual. Layers <= 20 are the informative ones and are used for")
    print("  the headline; L26 is printed separately and labelled.")
    NEAR = [L for L in Ls if L >= 24]
    FAR = [L for L in Ls if L < 24]
    for L in NEAR:
        b_ = max((st.mean([hit(r, f"v_item@L{L}@a{a}") - hit(r, "baseline") for r in rows]), a)
                 for a in As if f"v_item@L{L}@a{a}" in rows[0]["probs"])
        print(f"  [TAUTOLOGICAL] v_item at L{L}: {100*b_[0]:+.1f}pp -- not evidence")
    peak["v_item"] = max(
        (st.mean([hit(r, f"v_item@L{L}@a{a}") - hit(r, "baseline") for r in rows]), L, a)
        for L in FAR for a in As if f"v_item@L{L}@a{a}" in rows[0]["probs"])
    if "v_item" in peak:
        d0, L, a = peak["v_item"]
        print(f"  injecting each item's OWN oracle displacement, best at L{L} a={a}: "
              f"{100*d0:+.1f}pp")
        print(f"  the crop that PRODUCED that displacement is worth {100*gap:+.1f}pp")
        print(f"  -> at an INFORMATIVE layer a residual shift recovers {100*max(d0,0)/gap:.1f}% of the crop")
        if d0 < 0.25 * gap:
            print("     => the crop's benefit is NOT a linear shift of the last token's residual.")
            print("        No steering vector of this form can carry it, and v_mean cannot either.")
        else:
            print("     => a substantial part IS reachable by a residual shift; v_mean is worth having.")

    print("\n=== THE METHOD ARM vs its CONTROL ===")
    if "v_mean" in peak and "v_rand" in peak:
        dm, Lm, am = peak["v_mean"]
        vm = [hit(r, f"v_mean@L{Lm}@a{am}") for r in rows]
        dr, Lr, ar = peak["v_rand"]
        vr = [hit(r, f"v_rand@L{Lr}@a{ar}") for r in rows]
        d = [x - y for x, y in zip(vm, vr)]
        lo, hi = boot(d)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  v_mean best {100*dm:+.1f}pp (L{Lm}, a={am});  "
              f"v_rand best {100*dr:+.1f}pp (L{Lr}, a={ar})")
        print(f"  v_mean - v_rand = {100*st.mean(d):+.1f}pp  CI[{100*lo:+.1f},{100*hi:+.1f}]  {sig}")

        print("\n" + "=" * 70)
        if dm > 0 and sig == "SIG":
            print(f"  => RESIDUAL STEERING WORKS: {100*dm:+.1f}pp at zero extra compute, and it beats")
            print("     a norm-matched random direction. A transferable allocation direction exists.")
        else:
            print("  => RESIDUAL STEERING DOES NOT WORK. A shared direction exists in activation space")
            print("     but injecting it does not transfer the crop's benefit: the displacement")
            print("     carries the STATISTICS of a cropped image, not the evidence that makes the")
            print("     target legible. Consistent with §10B -- when the target is sub-token the")
            print("     information is absent from the representation, not merely mis-weighted.")


if __name__ == "__main__":
    main()
