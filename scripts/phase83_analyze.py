"""
Phase 83 analyzer: does SS14L replicate on Qwen2-VL-7B?

WHICH CONTRAST IS THE REPLICATION -- read this before the numbers
-----------------------------------------------------------------
    blockmean vs layerK   <- THE REPLICATION. Both are computed from Qwen2-VL's OWN attention;
                             one reads late layers (L15-26, rescaled), the other reads layer 2.
                             This is SS14L's claim, tested on a second model.

    linear vs layerK      <- NOT the replication. The `linear` arm loads Qwen3-VL's learned weights
                             (phase73_layer_structure.json) and applies them to Qwen2-VL attention.
                             That is a CROSS-MODEL WEIGHT TRANSFER -- an interesting but different
                             question, and it must not be quoted as the replication.

SS14L on Qwen3-VL-2B, at 10% keep:
    none 56.5% | random 38.2% | layer-2 34.6% (BELOW random, -22.0pp vs none) | blockmean 56.0%
    blockmean - layerK = +21.4pp

PRE-REGISTERED PREDICTION: SS14K measured Qwen2-VL's early layers at gt_pct 0.620 against Qwen3's
0.456, so the layer-2 penalty should be LARGER here. A SMALLER penalty refutes the stated mechanism
("the question-conditioned signal does not exist yet at layer 2").

REJECTION RULE: if blockmean does not beat layerK with a CI clear of zero at 10% keep, SS14L is
single-model and is REJECTED under the standard adopted 2026-09-16.
"""
import json
import random
import statistics as st

PATH = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase83_pruning_qwen2vl.jsonl"
Q3 = {"none": .565, "rand10": .382, "layer2_10": .346, "block10": .560, "gain10": .214}


def boot(a, b, n=10000, seed=0):
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    s = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return 100 * s[int(.025 * n)], 100 * s[int(.975 * n)]


def main():
    R = [json.loads(l) for l in open(PATH)]
    if len(R) < 60:
        print(f"only {len(R)} rows; wait"); return
    hit = lambda r, a: 1.0 * (max(range(4), key=lambda i: r["probs"][a][i]) == r["label"])
    KEEP = sorted({float(k.split("@")[1]) for k in R[0]["probs"] if "@" in k})
    print(f"n = {len(R)}   (Qwen3-VL reference in brackets)")
    none = [hit(r, "none") for r in R]
    print(f"  no pruning {100*st.mean(none):.1f}%  [{100*Q3['none']:.1f}%]\n")

    print(f"  {'keep':>6}{'rand':>9}{'layer2':>9}{'blockmn':>10}{'linear*':>10}")
    for kf in KEEP:
        row = "".join(f"{100*st.mean([hit(r, f'{c}@{kf}') for r in R]):8.1f}%"
                      for c in ("rand", "layerK", "blockmean", "linear"))
        print(f"  {100*kf:5.0f}%" + row)
    print("  * linear = Qwen3-VL weights transferred; NOT the replication contrast")

    print("\n=== THE REPLICATION: blockmean vs layerK (both from this model's own attention) ===")
    verdict = None
    for kf in KEEP:
        b = [hit(r, f"blockmean@{kf}") for r in R]
        l = [hit(r, f"layerK@{kf}") for r in R]
        d = 100 * (st.mean(b) - st.mean(l))
        lo, hi = boot(b, l)
        sig = "SIG" if lo > 0 else ("neg" if hi < 0 else "n.s.")
        ref = f"  [Qwen3: +{100*Q3['gain10']:.1f}pp]" if kf == 0.10 else ""
        print(f"  keep {100*kf:3.0f}%:  {d:+6.1f}pp  CI[{lo:+.1f},{hi:+.1f}]  {sig}{ref}")
        if kf == 0.10:
            verdict = (d, lo, hi)

    print("\n=== is layer-2 below random, as on Qwen3? ===")
    for kf in KEEP:
        l = [hit(r, f"layerK@{kf}") for r in R]
        rd = [hit(r, f"rand@{kf}") for r in R]
        print(f"  keep {100*kf:3.0f}%:  layer2 - random {100*(st.mean(l)-st.mean(rd)):+6.1f}pp"
              f"{'  [Qwen3: -3.7pp, BELOW random]' if kf == 0.10 else ''}")

    print("\n=== does a late read-out still prune for free? ===")
    for kf in KEEP:
        b = [hit(r, f"blockmean@{kf}") for r in R]
        lo, hi = boot(b, none)
        print(f"  keep {100*kf:3.0f}%:  blockmean - no-pruning {100*(st.mean(b)-st.mean(none)):+6.1f}pp"
              f" CI[{lo:+.1f},{hi:+.1f}]{'  [Qwen3: -0.5pp]' if kf == 0.10 else ''}")

    d, lo, hi = verdict
    l10 = st.mean([hit(r, "layerK@0.1") for r in R])
    n10 = st.mean(none)
    pen = 100 * (l10 - n10)
    print("\n" + "=" * 72)
    print(f"layer-2 penalty vs no pruning: {pen:+.1f}pp   [Qwen3: {100*(Q3['layer2_10']-Q3['none']):+.1f}pp]")
    print(f"PREDICTION was that this would be LARGER on Qwen2-VL (its early layers are worse).")
    if pen < 100 * (Q3["layer2_10"] - Q3["none"]):
        print("  -> prediction MET (penalty is larger here)")
    else:
        print("  -> prediction NOT MET (penalty is smaller) -- the stated mechanism is wrong or")
        print("     incomplete, whatever the replication verdict below.")
    print()
    if lo > 0:
        print("=> ★ SS14L REPLICATES. A late read-out beats layer-2 on a second architecture.")
        print("   The paper's headline claim holds at 2 models.")
    else:
        print("=> SS14L DOES NOT REPLICATE. Under the standard adopted 2026-09-16 it is REJECTED")
        print("   as a general claim and becomes a Qwen3-VL-2B observation. The paper falls back")
        print("   to the read-out defect (2 models) and DCR (2 models).")


if __name__ == "__main__":
    main()
