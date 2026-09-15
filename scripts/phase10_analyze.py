"""Phase 10 analysis: which factor causes the two-image vs crop-alone gap?

Pre-registered patterns (from the phase10 script docstring, written before results):
  SUPPORTS re-anchoring : crop_alone ~= crop_conn ~= crop_crop >> full_crop_* , other_full BETWEEN
  REFUTES  (dilution)   : other_full_crop ~= full_crop_conn
  REFUTES  (image count): crop_crop ~= full_crop_conn
"""
import json
import math
import random

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
ARMS = ["full_alone", "crop_alone", "crop_conn", "crop_crop", "blank_crop",
        "other_full_crop", "crop_full_conn", "full_crop_noconn", "full_crop_conn"]
DESC = {
    "full_alone":       "full image only                 (baseline)",
    "crop_alone":       "crop only                       (strong condition)",
    "crop_conn":        "crop + connector text           (isolates CONNECTOR)",
    "crop_crop":        "[crop + crop]                   (isolates IMAGE COUNT)",
    "blank_crop":       "[black + crop]                  (count, no scene*)",
    "other_full_crop":  "[OTHER image's scene + crop]    (isolates SCENE IDENTITY)",
    "crop_full_conn":   "[crop + full]  order swapped    (isolates POSITION)",
    "full_crop_noconn": "[full + crop]  no connector     (connector within 2-img)",
    "full_crop_conn":   "[full + crop]                   (Phase 7 replication)",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def paired(a, b, nb=3000, seed=5):
    rng = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(nb):
        ix = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in ix) / n - sum(b[i] for i in ix) / n)
    d.sort()
    return d[int(.025 * nb)], d[int(.975 * nb)], sum(1 for x in d if x > 0) / nb


def main():
    rows = [json.loads(l) for l in open(f"{DATA}/phase10_context_interference_results.jsonl")]
    seen, u = set(), []
    for r in rows:
        if r["uid"] not in seen:
            seen.add(r["uid"]); u.append(r)
    rows = u
    P = [r for r in rows if r["group"] == "positive"]
    N = [r for r in rows if r["group"] == "negative"]
    print(f"positives (confident denials, object PRESENT): {len(P)}")
    print(f"negatives (object ABSENT, false-positive arm): {len(N)}\n")

    def hits(rs, arm):
        return [1 if r["arms"][arm] > 0.5 else 0 for r in rs if arm in r["arms"]]

    print("=" * 94)
    print("RECOVERY (P(yes)>0.5) on confident denials  |  FP rate on absent objects  |  discrimination")
    print("=" * 94)
    res = {}
    for a in ARMS:
        hp, hn = hits(P, a), hits(N, a)
        res[a] = (hp, hn)
        lo, hi = wilson(sum(hp), len(hp))
        fp = 100 * sum(hn) / len(hn) if hn else float("nan")
        print(f"  {DESC[a]:<34} {100*sum(hp)/len(hp):5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]"
              f"   FP {fp:5.1f}%   disc {100*sum(hp)/len(hp)-fp:+6.1f}pp")
    print("  * black is NOT inert for this model (Phase 8: it asserts presence on black images),")
    print("    so crop_crop -- not blank_crop -- is the load-bearing image-count control.")

    print("\n" + "=" * 94)
    print("KEY CONTRASTS (paired, same items)")
    print("=" * 94)
    contrasts = [
        ("crop_alone", "full_crop_conn", "THE GAP: crop alone vs [full+crop]"),
        ("crop_alone", "crop_conn",      "connector text alone            (expect ~0)"),
        ("crop_alone", "crop_crop",      "image count alone               (expect ~0)"),
        ("crop_crop",  "full_crop_conn", "adding the FULL SCENE, count held at 2"),
        ("other_full_crop", "full_crop_conn", "OTHER scene vs OWN scene  (the re-anchoring test)"),
        ("crop_alone", "other_full_crop", "crop alone vs OTHER scene present"),
        ("full_crop_noconn", "full_crop_conn", "connector within two-image      (expect ~0)"),
        ("crop_full_conn", "full_crop_conn", "position of full image (crop first)"),
    ]
    for a, b, lab in contrasts:
        ha, hb = res[a][0], res[b][0]
        n = min(len(ha), len(hb))
        lo, hi, fr = paired(ha[:n], hb[:n])
        d = 100 * (sum(ha[:n]) / n - sum(hb[:n]) / n)
        star = "  <<<" if (lo > 0 or hi < 0) else ""
        print(f"  {lab:<50} {d:+6.1f}pp  CI [{100*lo:+6.1f},{100*hi:+6.1f}] {100*fr:5.1f}%{star}")

    print("\n" + "=" * 94)
    print("VERDICT against the pre-registered patterns")
    print("=" * 94)
    r = {a: 100 * sum(res[a][0]) / len(res[a][0]) for a in ARMS}
    print(f"  crop cluster (alone/conn/crop+crop): "
          f"{[round(r[a],1) for a in ('crop_alone','crop_conn','crop_crop')]}")
    print(f"  full cluster ([full+crop] +/- conn): "
          f"{[round(r[a],1) for a in ('full_crop_conn','full_crop_noconn')]}")
    print(f"  other-scene arm:                     {r['other_full_crop']:.1f}%")

    # Verdicts MUST be read on discrimination (recovery - FP), never raw recovery. The
    # other_full_crop arm carries FP 19.1% against ~0-4% elsewhere, so on raw recovery it looks
    # like re-anchoring evidence purely from a yes-bias shift -- the exact trap in FINDINGS
    # bugs #7/#10 and the Sec 5.1 counterfactual. dd() below is the difference-in-discrimination.
    for a, b, lab, sup, ref in [
        ("crop_crop", "full_crop_conn",
         "scene content causes the gap (count held at 2)",
         "SCENE CONTENT is the cause; image count and format exonerated.",
         "gap is NOT attributable to scene content."),
        ("other_full_crop", "full_crop_conn",
         "re-anchoring: OTHER scene vs the model's OWN scene",
         "own scene suppresses MORE than an arbitrary scene: RE-ANCHORING supported.",
         "any full scene suppresses equally: GENERIC INTERFERENCE, not re-anchoring."),
    ]:
        lo, hi, _ = dd(res[a], res[b])
        print(f"\n  [{lab}]")
        print(f"    diff-in-discrimination CI [{100*lo:+.1f}, {100*hi:+.1f}] pp")
        print(f"    -> {sup if lo > 0 else ref}")


def dd(pair_a, pair_b, nb=4000, seed=9):
    """Bootstrap CI on the difference of DISCRIMINATION (recovery - false-positive rate) between
    two arms. Positives and negatives are different items, so this is a difference-of-differences,
    not a per-item paired test."""
    rng = random.Random(seed)
    pa, na = pair_a
    pb, nb_ = pair_b
    out = []
    for _ in range(nb):
        pi = [rng.randrange(len(pa)) for _ in range(len(pa))]
        ni = [rng.randrange(len(na)) for _ in range(len(na))]
        da = sum(pa[i] for i in pi) / len(pa) - sum(na[i] for i in ni) / len(na)
        db = sum(pb[i] for i in pi) / len(pb) - sum(nb_[i] for i in ni) / len(nb_)
        out.append(da - db)
    out.sort()
    return out[int(.025 * nb)], out[int(.975 * nb)], sum(1 for x in out if x > 0) / nb


if __name__ == "__main__":
    main()
