"""Phase 24 analysis: locate the depth at which the target stops being available.

Verdict rules, fixed before results:
  * A layer is called INFORMATIVE only if its `obj` AUROC beats BOTH the paired `rand` control
    (same item, same size, different location) and the `last_blank` language-prior floor. A raw
    probe AUROC is not a finding -- Phase 16's 55.6% "recovery" was pure bias shift, and a probe
    has the same failure mode via scene co-occurrence.
  * Cross-validated. An in-sample probe on 2048 dims and ~1400 items separates anything.
  * The load-bearing cohort is the items the model gets WRONG. On items it answers correctly,
    "the information is present" is unsurprising and says nothing about C2.
  * The `last` curve, not the `obj` curve, is what tests irreversibility: `obj` high while `last`
    sits at the blank floor means the signal is never routed to the position that answers.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase14_analyze import auroc, fit_logreg

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BLOCKS = ["obj", "rand", "last", "last_blank"]
FOLDS = 5


def cv_auroc(X, y, folds=FOLDS, seed=7):
    """Out-of-fold AUROC. In-sample would be meaningless at D=2048."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if y[tr].sum() in (0, len(tr)):
            return float("nan")
        scores[te] = fit_logreg(X[tr], y[tr].astype(float))(X[te])
    return auroc(scores[y == 1].tolist(), scores[y == 0].tolist())


def perm_null(X, y, reps=20, seed=101):
    """Empirical chance ceiling. With D=2048 and n as low as ~195 in the failure cohort, a
    cross-validated probe can score well above 0.5 on pure noise; 0.5 is NOT the right threshold.
    Shuffle the labels, rerun the identical CV, and take the 95th percentile of the null AUROCs.
    A layer only counts as informative if it clears this, not if it clears 0.5."""
    rng = np.random.RandomState(seed)
    out = []
    for r in range(reps):
        yp = y.copy()
        rng.shuffle(yp)
        a = cv_auroc(X, yp, seed=200 + r)
        if not np.isnan(a):
            out.append(a)
    if not out:
        return float("nan")
    return float(np.percentile(out, 95))


def main():
    # Keep the feature bank in float16 (0.53 GB) and promote only the block under test.
    # Converting the whole bank to float32 up front costs an extra ~1.1 GB and got this analysis
    # OOM-killed while a 7B model was loading in another process.
    d = np.load(f"{DATA}/phase24_probe_sweep_feats.npz")
    F = d["feats"]
    layers = d["layers"].tolist()
    meta = [json.loads(l) for l in open(f"{DATA}/phase24_probe_sweep_meta.jsonl")]
    y = np.array([1 if m["group"] == "positive" else 0 for m in meta])
    D = F.shape[1] // (len(layers) * len(BLOCKS))
    correct = np.array([m["correct"] for m in meta])
    # MUST be bool. If these were 0/1 ints, `~correct` below yields -1/-2 and numpy would treat it
    # as FANCY INDEXING rather than a mask -- silently selecting elements at index -1 and -2 over
    # and over, with no error. That is the same bug class as the `(~y[rows]).sum()` cohort guard
    # (bug list). Verified bool today; asserted so a change to the meta format fails loudly.
    assert correct.dtype == bool, f"`correct` must be bool, got {correct.dtype}"
    toks = np.array([m["tokens_on_object"] for m in meta])
    print(f"n={len(meta)}  positives={int(y.sum())}  layers={len(layers)}  hidden_dim={D}")
    print(f"model correct on {100*correct.mean():.1f}% of items; "
          f"median tokens_on_object (positives) = {np.median(toks[y==1]):.2f}")

    def block(bi, li, rows=None):
        off = (li * len(BLOCKS) + bi) * D
        X = F[:, off:off + D]
        X = X if rows is None else X[rows]
        return np.ascontiguousarray(X, dtype=np.float32)

    def sweep(rows, title):
        print("\n" + "=" * 78)
        print(title + f"   (n={int(np.sum(rows))}, positives={int(y[rows].sum())})")
        print("=" * 78)
        if (y[rows] == 1).sum() < 20 or (y[rows] == 0).sum() < 20:
            print("  too few items in one class -- skipped")
            return
        yy = y[rows]
        # empirical chance ceiling from a mid-depth layer's `obj` block (label-shuffled)
        mid = len(layers) // 2
        null95 = perm_null(block(0, mid, rows), yy)
        print(f"  permutation null (95th pct of label-shuffled CV AUROC, L{layers[mid]} obj): "
              f"{null95:.3f}  <-- the real chance level, not 0.500")
        print(f"  {'layer':<7}" + "".join(f"{b:>13}" for b in BLOCKS) + "   verdict")
        best = (None, -1)
        for li, L in enumerate(layers):
            a = {}
            for bi, b in enumerate(BLOCKS):
                a[b] = cv_auroc(block(bi, li, rows), yy)
            # a layer must beat the paired control, the blank floor, AND the permutation null
            informative = (a["obj"] > a["rand"] + .02 and a["obj"] > a["last_blank"] + .02
                           and a["obj"] > null95)
            routed = (a["last"] > a["last_blank"] + .02 and a["last"] > null95)
            v = ("obj>controls" if informative else "-") + ("  routed" if routed else "")
            if a["obj"] > best[1]:
                best = (L, a["obj"])
            print(f"  L{L:<6}" + "".join(f"{a[b]:>13.3f}" for b in BLOCKS) + f"   {v}")
        print(f"  peak `obj` AUROC {best[1]:.3f} at layer {best[0]}"
              + ("   (BELOW the permutation null -- no signal)" if best[1] <= null95 else ""))

    allrows = np.ones(len(y), bool)
    sweep(allrows, "ALL ITEMS  (context; not the load-bearing cohort)")
    sweep(~correct, "ITEMS THE MODEL ANSWERS WRONG  <-- THE C2 COHORT")
    sweep(correct, "ITEMS THE MODEL ANSWERS CORRECTLY  (comparison)")

    # the sub-token regime is C3's domain, and is where C2 is supposed to bind
    sub = (y == 0) | (toks < 2.0)
    sweep(sub & ~correct, "SUB-TOKEN TARGETS (<2 tok) THE MODEL GETS WRONG  <-- C2 x C3")

    print("\n" + "=" * 78)
    print("POWER NOTE (read before believing any cell above)")
    print("=" * 78)
    print(f"""  The whole RePOPE-clean pool contains only 153 wrong positives and 42 wrong negatives,
  so the failure cohort cannot exceed ~195 items no matter how long this runs. That is the
  ceiling POPE imposes, not a sampling choice. Every failure-cohort row is therefore reported
  against a permutation null rather than 0.500, and a null at or above ~0.6 means the cohort is
  too small to support ANY claim at that layer -- say so rather than reading the point estimate.""")

    print("\n" + "=" * 78)
    print("READING THE RESULT (pre-registered interpretations)")
    print("=" * 78)
    print("""  obj high across depth, last flat at last_blank
      -> the target is encoded at its own visual tokens and NEVER ROUTED to the answer position.
         That is a mechanism for C2, not a restatement of it.
  obj high early then collapsing at some layer band
      -> the signal is OVERWRITTEN; name the band, and that band is the paper's figure.
  obj and last BOTH above their controls while the answer is wrong
      -> the readout HAS the information and does not use it. C2's current framing
         ("irreversible") would then be wrong and must be rewritten.
  ⚠️ WHAT THE `obj` CURVE DOES *NOT* SHOW -- read before quoting it.
      For positives `obj` pools the GT bbox tokens; for negatives it pools a RANDOM region (the
      object is absent, so there is no box). So an `obj` probe partly separates "patch embeddings
      of a real object" from "patch embeddings of a random patch of scene" -- a generic
      object-vs-background distinction available in the raw visual embeddings, NOT evidence that
      the model encodes whether the QUERIED category is present. The L0 (embedding-layer) value
      being ~0.96 is a symptom of exactly this: no transformer block has run yet.
      The query-conditional question is Phase 15's (queried vs size-matched other category), and
      the answer-relevant question here is the `last` vs `last_blank` curve, not `obj`.

  obj never beats rand on the wrong-answer cohort
      -> Phase 14's "it is encoded" does not hold where it matters, and the capacity-floor
         reading returns. This would contradict Phase 14 and needs reconciling, not burying.""")


if __name__ == "__main__":
    main()
