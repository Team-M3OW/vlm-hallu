"""
Phase 11: RePOPE label-noise audit -- a CORRECTNESS check on every result in this project, not a
new direction.

RePOPE (arXiv 2504.15707, github.com/YanNeu/RePOPE) re-annotated the 500 MSCOCO val images behind
POPE with three-way labels (yes / no / ambiguous, dual independent annotation, ambiguous pruned).
Reported error rates: **9.3% of POPE positives** and 1.7% of negatives are mislabeled.

WHY THIS THREATENS OUR RESULTS
------------------------------
Our entire interpretability programme runs on the `confident_denial` cohort: POPE positives where
Qwen3-VL gives P(yes) < 0.01. That is *selected for the model insisting the object is absent* --
exactly where mislabeled positives (object actually absent) must concentrate. If the error rate
inside that cohort is 30-40% rather than 9.3%, then part of the "dissociation", part of the 40.4%
crop recovery, and *most plausibly the 42% recovered-by-neither residual* are label noise rather
than model failure. The residual is the tell: items no intervention recovers are exactly what you
would expect if the object is not there.

WHAT THIS SCRIPT REPORTS
------------------------
  1. RePOPE error/ambiguity rate inside each cohort we use, vs. POPE positives overall.
     A cohort rate far above the 9.3% base rate is direct evidence of label-noise enrichment.
  2. Every headline number recomputed on RePOPE-CLEAN items only (label still "yes", not pruned).
  3. The recovered-by-neither residual, split by RePOPE verdict -- does "no intervention recovers
     it" mostly mean "the object was never there"?

CPU-only; no GPU, no model. Joins on (split, question_id), which is how POPE, our uids
(`pos_{split}_{question_id}`), and the RePOPE annotation files all key.
"""
import json
import math
import sys

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
REPOPE = f"{DATA}/repope"  # vendored from github.com/YanNeu/RePOPE (arXiv 2504.15707)
SPLITS = ["random", "popular", "adversarial"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def load_repope():
    """(split, qid) -> {'pope': label, 'repope': label or None if pruned as ambiguous}"""
    out = {}
    for sp in SPLITS:
        pope = [json.loads(l) for l in open(f"{REPOPE}/coco_pope_{sp}.json")]
        rep = [json.loads(l) for l in open(f"{REPOPE}/coco_repope_{sp}.json")]
        rmap = {str(x["question_id"]): x["label"] for x in rep}
        for x in pope:
            qid = str(x["question_id"])
            out[(sp, qid)] = {"pope": x["label"], "repope": rmap.get(qid)}
    return out


def verdict(rp, key):
    """CLEAN = RePOPE confirms the POPE positive. FLIPPED = RePOPE says the object is absent.
    AMBIGUOUS = pruned by RePOPE. MISSING = not in the RePOPE files at all."""
    e = rp.get(key)
    if e is None:
        return "MISSING"
    if e["repope"] is None:
        return "AMBIGUOUS"
    return "CLEAN" if e["repope"] == e["pope"] else "FLIPPED"


def key_of(rec):
    return (rec["split"], str(rec["question_id"]))


def report(name, keys, rp):
    from collections import Counter
    c = Counter(verdict(rp, k) for k in keys)
    n = len(keys)
    bad = c["FLIPPED"]
    lo, hi = wilson(bad, n)
    print(f"  {name:<44} n={n:<5} FLIPPED {bad:4d} ({100*bad/n:5.1f}% [{100*lo:.1f},{100*hi:.1f}])"
          f"  AMBIG {c['AMBIGUOUS']:4d} ({100*c['AMBIGUOUS']/n:4.1f}%)  MISSING {c['MISSING']}")
    return c


def main():
    rp = load_repope()
    print(f"RePOPE loaded: {len(rp)} POPE probes across {len(SPLITS)} splits\n")

    recs = [json.loads(l) for l in open(f"{DATA}/phase1_results_qwen_dedup.jsonl")]
    positives = [r for r in recs if r["label"] == "yes"]

    print("=" * 104)
    print("1. LABEL-NOISE ENRICHMENT BY COHORT  (baseline: RePOPE reports 9.3% error on POPE positives)")
    print("=" * 104)
    report("ALL POPE positives we evaluate", [key_of(r) for r in positives], rp)
    denials = [r for r in positives if r["p_yes_real"] < 0.01]
    report("confident_denial cohort (P(yes)<0.01)", [key_of(r) for r in denials], rp)

    p4 = [json.loads(l) for l in open(f"{DATA}/phase4_localize_results.jsonl")]
    p4pos = [r for r in p4 if r["group"] == "confident_denial"]
    report("Phase 4 dissociation cohort", [key_of(r) for r in p4pos], rp)

    p10 = [json.loads(l) for l in open(f"{DATA}/phase10_context_interference_results.jsonl")]
    p10pos = [r for r in p10 if r["group"] == "positive"]
    report("Phase 7/9/10 crop cohort", [key_of(r) for r in p10pos], rp)

    p8 = [json.loads(l) for l in open(f"{DATA}/phase8_grounding_detector_results.jsonl")]
    p8pos = [r for r in p8 if r["group"] == "positive"]
    report("Phase 8 BROAD positive sample (unselected)", [key_of(r) for r in p8pos], rp)

    print("\n" + "=" * 104)
    print("2. HEADLINE NUMBERS RECOMPUTED ON RePOPE-CLEAN ITEMS ONLY")
    print("=" * 104)

    def clean(rows):
        return [r for r in rows if verdict(rp, key_of(r)) == "CLEAN"]

    p4d = {r["uid"]: r for r in p4pos}
    cl = clean(p4pos)
    print(f"\n  Phase 4 dissociation  (all {len(p4pos)} -> clean {len(cl)})")
    for lab, rows in [("ALL", p4pos), ("RePOPE-CLEAN", cl)]:
        e = sum(1 for r in rows if r["n_pred_boxes"] > 0) / len(rows)
        i = sum(1 for r in rows if r["best_iou"] > 0.5) / len(rows)
        print(f"    {lab:<14} emits box {100*e:5.1f}%   IoU>0.5 {100*i:5.1f}%")

    print(f"\n  Phase 10 context interference  (all {len(p10pos)} -> clean {len(clean(p10pos))})")
    arms = ["crop_alone", "crop_crop", "full_crop_conn", "crop_full_conn", "other_full_crop"]
    for lab, rows in [("ALL", p10pos), ("RePOPE-CLEAN", clean(p10pos))]:
        cells = "  ".join(
            f"{a}={100*sum(1 for r in rows if r['arms'].get(a,0)>0.5)/len(rows):5.1f}%" for a in arms)
        print(f"    {lab:<14} {cells}")
    # the gap, on clean items only
    cp = clean(p10pos)
    ca = sum(1 for r in cp if r["arms"]["crop_alone"] > 0.5) / len(cp)
    fc = sum(1 for r in cp if r["arms"]["full_crop_conn"] > 0.5) / len(cp)
    print(f"    -> THE GAP on clean items: {100*ca:.1f}% vs {100*fc:.1f}%  = {100*(ca-fc):+.1f}pp")

    print("\n" + "=" * 104)
    print("3. THE 'RECOVERED BY NEITHER' RESIDUAL -- is it label noise?")
    print("=" * 104)
    p7 = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{DATA}/phase7_vision_zoom_results.jsonl")}
    p10m = {r["uid"]: r for r in p10pos}
    common = [u for u in p10m if u in p4d and u in p7]
    res_keys, rec_keys = [], []
    for u in common:
        recovered = (p10m[u]["arms"]["crop_alone"] > 0.5) or (p4d[u]["best_iou"] > 0.5)
        (rec_keys if recovered else res_keys).append(key_of(p10m[u]))
    print(f"  n={len(common)} confident denials with both a crop arm and a grounding arm")
    report("RECOVERED (crop-alone or grounding IoU>0.5)", rec_keys, rp)
    report("RESIDUAL  (recovered by NEITHER)", res_keys, rp)
    print("\n  If the residual's FLIPPED rate greatly exceeds the recovered group's, the residual is")
    print("  substantially label noise -- i.e. 'no intervention recovers it' partly means")
    print("  'the object was never there'.")


if __name__ == "__main__":
    main()
