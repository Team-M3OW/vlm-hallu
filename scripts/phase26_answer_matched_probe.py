"""
Phase 26: the ANSWER-MATCHED probe -- Phase 24's circularity fix, run on Phase 24's own features.

WHY
---
Phase 24's `last` probe was circular (see FINDINGS 4L's retraction). `last` is the position the
yes/no logits are read from, so the model's OWN ANSWER is linearly decodable there by construction.
Phase 24's cohorts were selected BY correctness, which makes the true label an exact function of the
answer (wrong-only: label == NOT answer; correct-only: label == answer). A probe could therefore
score ~1.0 by decoding the answer and flipping the sign, without representing visual evidence at
all -- and `last_blank` reaching 0.75-0.82 with NO IMAGE confirmed it.

THE FIX: condition on the answer, probe the truth inside it.
----------------------------------------------------------
    ANSWERED-NO cohort   n=612   153 objects actually PRESENT (false negatives)
                                 459 objects actually ABSENT  (true negatives)
The model said "no" to every one of them. Its answer is CONSTANT across the probe's two classes, so
decoding the answer yields exactly nothing. Any remaining AUROC is genuine information about the
input. This is the question C2 has always been asking, finally asked without the confound:

    **When the model denies an object that is present, does its internal state still distinguish
    that case from a genuine absence?**

  YES at `last`  -> the readout HAS the evidence and does not act on it. "Irreversible" is the wrong
                    word for C2 and the claim must be rewritten as a USE failure, not a routing one.
  NO at `last`   -> the evidence never reaches the answer position on failures; C2's routing story
                    survives, now with a mechanism instead of four nulls.

THE CONTROL THAT DECIDES WHETHER THE PROBE ADDS ANYTHING
--------------------------------------------------------
Within this cohort `p_yes` still VARIES (0.001..0.5). If false negatives simply carry higher p_yes
than true negatives, a scalar already separates the classes and a 2048-dim probe that ties it has
discovered nothing. So `p_yes` alone is scored as a baseline row, and the probe is only interesting
where it BEATS that scalar.

Also reported: the permutation null (0.5 is not the chance level at these n), and `last_blank`
(same position, blank image) as the language-prior floor.

NOTE the `obj` block is still NOT interpretable here, for the reason in FINDINGS 4L: for positives
it pools the GT bbox, for negatives a random region, so it partly measures object-vs-background in
the raw embeddings. It is printed for completeness and must not be cited.

Uses Phase 24's saved features -- NO new forward passes.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
from phase14_analyze import auroc, fit_logreg
from phase24_analyze import cv_auroc, perm_null

DATA = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
BLOCKS = ["obj", "rand", "last", "last_blank"]


def main():
    d = np.load(f"{DATA}/phase24_probe_sweep_feats.npz")
    F = d["feats"]
    layers = d["layers"].tolist()
    meta = [json.loads(l) for l in open(f"{DATA}/phase24_probe_sweep_meta.jsonl")]
    D = F.shape[1] // (len(layers) * len(BLOCKS))
    said_yes = np.array([m["p_yes_fp16"] > 0.5 for m in meta])
    present = np.array([1 if m["group"] == "positive" else 0 for m in meta])
    p_yes = np.array([m["p_yes_fp16"] for m in meta])
    toks = np.array([m["tokens_on_object"] if m["tokens_on_object"] is not None else -1.0
                     for m in meta])

    def block(bi, li, rows):
        off = (li * len(BLOCKS) + bi) * D
        return np.ascontiguousarray(F[rows, off:off + D], dtype=np.float32)

    def sweep(rows, title):
        y = present[rows]
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        print("\n" + "=" * 80)
        print(f"{title}   n={len(y)}  present={n1}  absent={n0}")
        print("=" * 80)
        if n1 < 20 or n0 < 20:
            print("  too few in one class -- skipped")
            return
        # scalar baseline: does the model's own confidence already separate the classes?
        pb = auroc(p_yes[rows][y == 1].tolist(), p_yes[rows][y == 0].tolist())
        null95 = perm_null(block(2, len(layers) // 2, rows), y)
        print(f"  BASELINE  p_yes alone (a single scalar): AUROC {pb:.3f}")
        print(f"  permutation null (label-shuffled, L{layers[len(layers)//2]} last): {null95:.3f}")
        print(f"  a probe is only INTERESTING where `last` beats BOTH {max(pb, null95):.3f} "
              f"and its own blank-image floor\n")
        print(f"  {'layer':<7}" + "".join(f"{b:>13}" for b in BLOCKS) + "   verdict")
        for li, L in enumerate(layers):
            if L % 2 and L not in (len(layers) - 1,):
                continue                      # every other layer: the sweep is CPU-expensive
            a = {b: cv_auroc(block(bi, li, rows), y) for bi, b in enumerate(BLOCKS)}
            good = (a["last"] > max(pb, null95) + .02 and a["last"] > a["last_blank"] + .02)
            print(f"  L{L:<6}" + "".join(f"{a[b]:>13.3f}" for b in BLOCKS)
                  + ("   last>baseline+blank" if good else "   -"))

    no_rows = ~said_yes
    print("PHASE 26 -- answer-matched probe, built from Phase 24 features (no new forward passes)")
    sweep(no_rows, "ANSWERED-NO: model said 'no' to EVERY item; probe asks if the object is present")
    sub = no_rows & ((present == 0) | (toks < 2.0))
    sweep(sub, "ANSWERED-NO, SUB-TOKEN targets (<2 tok) -- C2 x C3")
    sweep(said_yes, "ANSWERED-YES (mirror; absent class is only the 42 false positives)")

    print("\n" + "=" * 80)
    print("HOW TO READ THIS")
    print("=" * 80)
    print("""  `last` clearly above the p_yes baseline AND above last_blank
      -> the answer position carries evidence the answer does not use. C2 becomes a USE failure:
         the information is there at the readout and is not acted on. Rewrite C2.
  `last` at or below the p_yes baseline
      -> the probe adds nothing over the model's own confidence. No mechanism claim is licensed;
         say that plainly rather than reporting the raw AUROC.
  `last` near the permutation null
      -> on failures the evidence does not reach the answer position. C2's routing story survives
         and now has a mechanism.""")


if __name__ == "__main__":
    main()
