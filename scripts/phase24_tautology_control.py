"""
Tautology control for the Phase 24 `obj` probe. No GPU, no forward passes -- the saved feature bank
already contains everything needed.

THE SUSPICION
-------------
Phase 24 reports `obj` peak AUROC **1.000** on the cohort the model answers WRONG (n=195, permutation
null 0.595) and **0.999** on sub-token-and-wrong. An AUROC of exactly 1.000 on the cohort where the
model fails is extraordinary, and §4 reads it as "the target is encoded even when the model errs".

THE CONFOUND, found by reading how the ROI is chosen (phase24_probe_layer_sweep.py:190-196):

    if group == "positive":                      # object IS in the image
        roi = GT bounding box of the real annotated object
    else:                                        # object is ABSENT -- there is no box to use
        roi = a UNIFORMLY RANDOM rectangle, size ~U(5%,30%) of each side

So the `obj` block pools hidden states over **a GT object region for positives** and over **a random
rectangle for negatives**. A probe on those features need not represent the queried object at all --
it only has to answer "are these tokens from an annotated object region or from a random rectangle?"
GT boxes sit on salient, object-shaped, often-centred regions; random rectangles are mostly
background and differ in size, position and content statistics. That distinction is linearly trivial
and, crucially, **has nothing to do with the model's answer**, which would explain why the
wrong-answer cohort scores as high as the correct one.

Note the `rand` control does NOT catch this. For a positive item, `obj`=GT box and `rand`=random box;
for a NEGATIVE item, `obj`=random box and `rand`=random box -- drawn from the same distribution. So
`rand` is a valid control for "is this location special" but not for the positive/negative ROI-type
asymmetry, because the asymmetry only exists in the `obj` block.

THE CONTROLS (all free; features already saved)
----------------------------------------------
 C1  REGION-TYPE PROBE -- the decisive one. Using POSITIVE items ONLY, classify `obj` vectors vs
     `rand` vectors (label = which block, not presence). Both come from the same image and the same
     forward pass, so nothing differs except GT-region vs random-region. If this is near 1.0, region
     type is linearly decodable and the headline AUROC is explained without any object representation.

 C2  RANDOM-ROI PRESENCE -- use the `rand` block for BOTH classes and redo the presence task. Now
     positives and negatives have identically-distributed ROIs, so the ROI-type shortcut is gone.
     Whatever survives here is presence signal that is NOT ROI-type.

 C3  ROI-GEOMETRY-ONLY BASELINE -- predict presence from the ROI's geometry alone (token count,
     coverage flags, grid size) with NO hidden states at all. If geometry alone separates the
     classes, the shortcut is available even before the model runs.

Read together: if C1 is high and C2 collapses toward the permutation null, the Phase 24 `obj` result
is an ROI-selection artifact and §4's "encoded even when wrong" claim does not stand as written.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")

FEATS = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase24_probe_sweep_feats.npz"
META = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data/phase24_probe_sweep_meta.jsonl"
BLOCKS = ["obj", "rand", "last", "last_blank"]
FOLDS = 5


def cv_auroc(X, y, folds=FOLDS, seed=7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    sc = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(np.unique(y[tr])) < 2:
            return float("nan")
        s = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=0.1).fit(s.transform(X[tr]), y[tr])
        sc[te] = m.predict_proba(s.transform(X[te]))[:, 1]
    return roc_auc_score(y, sc)


def perm_null(X, y, reps=20, seed=101):
    rng = np.random.RandomState(seed)
    v = []
    for r in range(reps):
        yp = y.copy()
        rng.shuffle(yp)
        a = cv_auroc(X, yp, seed=200 + r)
        if not np.isnan(a):
            v.append(a)
    return float(np.percentile(v, 95)) if v else float("nan")


def main():
    meta = [json.loads(l) for l in open(META)]
    z = np.load(FEATS)
    F = z["feats"]                      # (n, 237568) float16 -- keep fp16; promoting the whole
    layers = [int(v) for v in z["layers"]]   # bank to fp32 once OOM-killed a run (bug list)
    n = len(meta)
    D = F.shape[1] // (len(layers) * len(BLOCKS))
    print(f"n={n}  feat dim {F.shape[1]}  hidden {D}  layers {len(layers)}")

    def block(bi, li, rows=None):
        off = (li * len(BLOCKS) + bi) * D
        X = F[rows, off:off + D] if rows is not None else F[:, off:off + D]
        return np.asarray(X, dtype=np.float32)   # promote only the slice actually used

    y = np.array([1 if m["group"] == "positive" else 0 for m in meta])
    correct = np.array([m["correct"] for m in meta])
    assert correct.dtype == bool
    pos = y == 1
    wrong = ~correct

    print(f"  positives {pos.sum()}  negatives {(~pos).sum()}  wrong {wrong.sum()}")

    # ---------------------------------------------------------------- C1
    print("\n" + "=" * 74)
    print("C1  REGION-TYPE PROBE: obj vs rand, POSITIVE items only")
    print("    same image, same forward pass; only GT-region vs random-region differs.")
    print("    If this is ~1.0 the headline AUROC needs no object representation at all.")
    print("=" * 74)
    pidx = np.where(pos)[0]
    print(f"  {'layer':<8}{'AUROC(obj vs rand)':>20}")
    peak_c1 = (None, -1)
    for li in [i for i in (0,4,8,12,16,20,24,26,28) if i < len(layers)]:
        Xo, Xr = block(0, li, pidx), block(1, li, pidx)
        X = np.vstack([Xo, Xr])
        yy = np.concatenate([np.ones(len(Xo)), np.zeros(len(Xr))])
        a = cv_auroc(X, yy)
        if a > peak_c1[1]:
            peak_c1 = (li, a)
        print(f"  L{layers[li]:<7}{a:>20.3f}")
    print(f"  peak {peak_c1[1]:.3f} at L{layers[peak_c1[0]]}")

    # ---------------------------------------------------------------- C2
    print("\n" + "=" * 74)
    print("C2  RANDOM-ROI PRESENCE: same presence task, but `rand` block for BOTH classes")
    print("    ROI distribution now identical across classes -> ROI-type shortcut removed.")
    print("=" * 74)
    for name, rows in [("ALL ITEMS", np.ones(n, bool)),
                       ("WRONG (the C2 cohort)", wrong)]:
        if (y[rows] == 1).sum() < 20 or (y[rows] == 0).sum() < 20:
            print(f"  {name}: too few in one class, skipped")
            continue
        yy = y[rows]
        null = perm_null(block(1, min(14, len(layers)-1), rows), yy)
        print(f"\n  {name}  (n={int(rows.sum())}, pos={int(yy.sum())})   perm null {null:.3f}")
        print(f"  {'layer':<8}{'obj (GT vs RANDOM roi)':>24}{'rand (random vs random)':>26}")
        best_o, best_r = -1, -1
        for li in [i for i in (0,8,16,20,24,28) if i < len(layers)]:
            ao = cv_auroc(block(0, li, rows), yy)
            ar = cv_auroc(block(1, li, rows), yy)
            best_o, best_r = max(best_o, ao), max(best_r, ar)
            print(f"  L{layers[li]:<7}{ao:>24.3f}{ar:>26.3f}")
        print(f"  peak: obj {best_o:.3f}   rand {best_r:.3f}   null {null:.3f}")
        print(f"  -> ROI-type-free presence signal = rand peak vs null: "
              f"{best_r:.3f} vs {null:.3f} "
              f"({'SURVIVES' if best_r > null + .02 else 'DOES NOT SURVIVE'})")

    # ---------------------------------------------------------------- C3
    print("\n" + "=" * 74)
    print("C3  GEOMETRY-ONLY BASELINE: predict presence from ROI geometry, NO hidden states")
    print("=" * 74)
    G = np.array([[m["n_obj_tokens"], 1.0 * m["zero_coverage"],
                   m["grid"][0], m["grid"][1], m["grid"][0] * m["grid"][1],
                   m["tokens_on_object"] if m["tokens_on_object"] is not None else -1.0]
                  for m in meta], dtype=np.float32)
    for name, rows in [("ALL ITEMS", np.ones(n, bool)), ("WRONG", wrong)]:
        if (y[rows] == 1).sum() < 20 or (y[rows] == 0).sum() < 20:
            continue
        a = cv_auroc(G[rows], y[rows])
        print(f"  {name:<24} geometry-only AUROC {a:.3f}   (n={int(rows.sum())})")
    print("\n  If geometry alone is well above chance, the ROI-type shortcut is available")
    print("  BEFORE the model runs, and the `obj` probe can exploit it trivially.")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""  C1 high  AND  C2 `rand` at null  ->  the Phase 24 `obj` result is an ROI-SELECTION
       ARTIFACT. §4's "encoded even when the model is wrong" does not stand as written, and the
       1.000 AUROC is region-type decoding. The fix is a presence probe whose ROI does not depend
       on the label -- which POPE cannot supply directly, since absent objects have no box.
  C1 high  BUT  C2 `rand` clearly above null  ->  the shortcut EXISTS but is not the whole story;
       report the ROI-matched number as the real effect size and retract the 1.000.
  C1 near chance  ->  suspicion refuted; the original reading stands.""")


if __name__ == "__main__":
    main()
