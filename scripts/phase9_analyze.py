"""
Phase 9 (E2) analysis: the readout x resolution factorial.

Reading rules baked in, so the table cannot be misread:
  * Common metric across ALL cells = "asserts the object is present" (yes/no: P(yes)>0.5;
    grounding: emits a bbox instead of "There are none.").
  * IoU>0.5 is the stricter secondary, reported ONLY where the grounding channel is used, and
    ALWAYS beside its model-free trivial-box floor -- a box covering the whole input scores
    IoU>0.5 on 17.0% of crops but only 6.4% of full images, so the two IoU cells are NOT on a
    common floor and a raw comparison between them is invalid.
  * Every cell carries its negative arm. Phase 8 established that the grounding channel asserts
    presence on ~everything (100% "box" on P(yes)'s own false positives, and boxes on black
    images), so an emission rate without its false-positive rate is meaningless.
"""
import json
import sys
import math
import random
import statistics as st

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase6_attention_entanglement import load_bbox_lookup
from phase7_vision_zoom import union_bbox, padded_crop_box, PAD_FRAC
from phase4_localize_vs_answer import iou

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k, n, label, extra=""):
    lo, hi = wilson(k, n)
    print(f"    {label:<46} {k:4d}/{n:<4d} {100*k/n:6.1f}%  [{100*lo:5.1f},{100*hi:5.1f}] {extra}")


def paired_sign(a, b, n_boot=3000, seed=5):
    """Bootstrap CI on the difference of two paired binary rates (same items)."""
    rng = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    d.sort()
    return d[int(0.025 * n_boot)], d[int(0.975 * n_boot)], sum(1 for x in d if x > 0) / n_boot


def main():
    p9 = [json.loads(l) for l in open(f"{DATA}/phase9_factorial_results.jsonl")]
    seen, u = set(), []
    for r in p9:
        if r["uid"] not in seen:
            seen.add(r["uid"]); u.append(r)
    p9 = u
    P = [r for r in p9 if r["group"] == "positive"]
    N = [r for r in p9 if r["group"] == "negative"]

    p7 = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{DATA}/phase7_vision_zoom_results.jsonl")}
    p4 = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{DATA}/phase4_localize_results.jsonl")}
    recs = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{DATA}/phase1_results_qwen_dedup.jsonl")}
    bb, isz = load_bbox_lookup()

    print(f"positives (confident denials): {len(P)}   negatives (object absent): {len(N)}")
    if len(N) == 0:
        print("!! NEGATIVE ARM NOT YET RUN -- positive-arm numbers below are UNINTERPRETABLE alone.")

    # ---- model-free trivial-box floors ----
    def floors(rows):
        crop, full = [], []
        for r in rows:
            f = recs[r["uid"]]; key = (f["split"], str(f["question_id"]))
            boxes, sz = bb.get(key), isz.get(key)
            if not boxes or not sz:
                continue
            W, H = sz
            x0, y0, x1, y1 = union_bbox(boxes)
            cx0, cy0, cx1, cy1 = padded_crop_box(x0, y0, x1, y1, W, H, PAD_FRAC)
            cw, ch = cx1 - cx0, cy1 - cy0
            gtc = [[max(0, b[0]-cx0), max(0, b[1]-cy0), min(cw, b[0]+b[2]-cx0), min(ch, b[1]+b[3]-cy0)]
                   for b in boxes]
            gtf = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes]
            crop.append(max(iou([0, 0, cw, ch], g) for g in gtc))
            full.append(max(iou([0, 0, W, H], g) for g in gtf))
        return crop, full

    fl_crop, fl_full = floors(P)

    print("\n" + "=" * 92)
    print("THE FACTORIAL -- common metric: 'asserts the object is present'")
    print("all rows are the SAME items: confident denials (baseline P(yes)<0.01, object present)")
    print("=" * 92)

    uids = [r["uid"] for r in P]
    print("\n  ROW 1: ORIGINAL PIXELS (unmodified full image)")
    base = [1 if p7[u]["baseline_p_yes"] > 0.5 else 0 for u in uids if u in p7]
    rate(sum(base), len(base), "yes/no readout      -> P(yes)>0.5", "(0% by cohort construction)")
    g_full = [1 if p4[u]["n_pred_boxes"] > 0 else 0 for u in uids if u in p4]
    rate(sum(g_full), len(g_full), "grounding readout   -> emits a box")
    iou_full = [1 if p4[u]["best_iou"] > 0.5 else 0 for u in uids if u in p4]
    rate(sum(iou_full), len(iou_full), "grounding, STRICT   -> IoU>0.5",
         f"| trivial-box floor {100*sum(1 for i in fl_full if i>0.5)/len(fl_full):.1f}%")

    print("\n  ROW 2: ZOOMED (oracle crop of the same photo -- GT-located, no new information)")
    z = [1 if p7[u]["oracle_zoom_p_yes"] > 0.5 else 0 for u in uids if u in p7]
    rate(sum(z), len(z), "yes/no, two-image [full+crop]  -> P(yes)>0.5")
    zc = [1 if r["crop_alone_yesno"] > 0.5 else 0 for r in P]
    rate(sum(zc), len(zc), "yes/no, crop alone             -> P(yes)>0.5")
    tg = [1 if r["twoimg_ground"]["n_boxes"] > 0 else 0 for r in P]
    rate(sum(tg), len(tg), "grounding, two-image [full+crop] -> emits a box")
    cg = [1 if r["crop_alone_ground"]["n_boxes"] > 0 else 0 for r in P]
    rate(sum(cg), len(cg), "grounding, crop alone          -> emits a box")
    ci = [1 if r["crop_alone_ground"]["best_iou"] > 0.5 else 0 for r in P]
    rate(sum(ci), len(ci), "grounding, crop alone, STRICT  -> IoU>0.5",
         f"| trivial-box floor {100*sum(1 for i in fl_crop if i>0.5)/len(fl_crop):.1f}%")

    print("\n  CONTENT CONTROL (random crop of the same image, object NOT in it)")
    rr = [1 if p7[u]["random_zoom_p_yes"] > 0.5 else 0 for u in uids if u in p7]
    rate(sum(rr), len(rr), "yes/no, two-image [full+random crop]")
    rc = [1 if r["randcrop_alone_yesno"] > 0.5 else 0 for r in P if "randcrop_alone_yesno" in r]
    rate(sum(rc), len(rc), "yes/no, random crop alone")
    rg = [1 if r["randcrop_alone_ground"]["n_boxes"] > 0 else 0 for r in P if "randcrop_alone_ground" in r]
    rate(sum(rg), len(rg), "grounding, random crop alone -> emits a box")

    if N:
        print("\n  NEGATIVE ARM (object genuinely ABSENT; size-matched random crop)")
        print("  -- these are FALSE-POSITIVE rates. Emission cells above are meaningless without them.")
        nz = [1 if r["crop_alone_yesno"] > 0.5 else 0 for r in N]
        rate(sum(nz), len(nz), "yes/no, crop alone           -> P(yes)>0.5  [FP]")
        ng = [1 if r["crop_alone_ground"]["n_boxes"] > 0 else 0 for r in N]
        rate(sum(ng), len(ng), "grounding, crop alone        -> emits a box [FP]")
        ntg = [1 if r["twoimg_ground"]["n_boxes"] > 0 else 0 for r in N]
        rate(sum(ntg), len(ntg), "grounding, two-image         -> emits a box [FP]")
        nf = [1 if p4[r["uid"]]["n_pred_boxes"] > 0 else 0 for r in N if r["uid"] in p4]
        rate(sum(nf), len(nf), "grounding, full image        -> emits a box [FP]")

        print("\n  DISCRIMINATION (assert-rate on present MINUS assert-rate on absent):")
        for lab, pos, neg in [("yes/no,   crop alone", zc, nz),
                              ("grounding, crop alone", cg, ng),
                              ("grounding, two-image ", tg, ntg),
                              ("grounding, full image", g_full, nf)]:
            d = sum(pos)/len(pos) - sum(neg)/len(neg)
            print(f"    {lab:<26} {100*d:+6.1f} pp")

    print("\n" + "=" * 92)
    print("HEAD-TO-HEAD, matched metric at last: resolution vs readout on identical items")
    print("=" * 92)
    common = [u for u in uids if u in p7 and u in p4]
    zoom_a = [1 if p7[u]["oracle_zoom_p_yes"] > 0.5 else 0 for u in common]
    gnd_i = [1 if p4[u]["best_iou"] > 0.5 else 0 for u in common]
    print(f"  n={len(common)}")
    print(f"    resolution (oracle zoom, asserts)        {100*sum(zoom_a)/len(common):5.1f}%")
    print(f"    readout (grounding, full image, IoU>0.5) {100*sum(gnd_i)/len(common):5.1f}%")
    lo, hi, fr = paired_sign(gnd_i, zoom_a)
    print(f"    paired diff 95% CI [{100*lo:+.1f}, {100*hi:+.1f}] pp, {100*fr:.1f}% of resamples favor readout")
    zs, gs = set(u for u, v in zip(common, zoom_a) if v), set(u for u, v in zip(common, gnd_i) if v)
    print(f"    disjointness: zoom-only {len(zs-gs)}, readout-only {len(gs-zs)}, both {len(zs&gs)}, "
          f"NEITHER {len(common)-len(zs|gs)} ({100*(len(common)-len(zs|gs))/len(common):.1f}%)")


if __name__ == "__main__":
    main()
