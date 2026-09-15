# Phase 24 — Layer sweep of presence probes ⚠ PARTIALLY RETRACTED

## 1. Research question
Across layers, how well can presence be decoded from the object's own tokens, from a random region,
and from the final position?

## 2. Finding and contribution (plain English)
Two of the three curves are fine; **two claims were retracted**, and the retractions are the useful
content.

The `last`-position curve is **circular** — it decodes the answer the model is about to give, not
the evidence. Retracted the same day it was written.

The `obj` curve reached AUROC **1.000** on the cohort the model answers wrong, which would have been
extraordinary. Phase 24's tautology control (below) showed it is an **ROI-selection artefact**: for
positives the ROI is a GT object box, for negatives a random rectangle, so the probe only has to
tell "annotated object region" from "random rectangle" — which is trivial and has nothing to do with
the queried object.

## 3. Numbers that changed
- Retracted: `obj` peak AUROC **1.000** → ROI artefact.
- Control C1 (obj vs rand on positives only) is **flat at 0.964–0.971 including 0.967 at L0** — a
  layer-0 result proves it cannot be about learned object representation.
- ROI-matched residual: **0.742 vs null 0.600**; on all items **0.599 vs 0.532**.
- What survives: "presence is weakly decodable from arbitrary tokens", not local encoding.

## 4. Keep in paper: 4/10
Keep only as a **negative-results / methodology** item. It is one of the clearest examples in the
project of a spectacular number that was an artefact, and the C1-at-L0 diagnostic is a reusable
trick.

## 5. Experiment, step by step
1. Extract features for four blocks — `obj`, `rand`, `last`, `last_blank` — at many layers.
2. Train cross-validated probes for presence.
3. **The control that decided it (C1):** using *positive items only*, classify `obj` vectors against
   `rand` vectors from the same image and same forward pass. Label is which block, not presence.
   If this is near 1.0, region type alone explains the headline.
4. **C2:** redo presence using the `rand` block for *both* classes, so the ROI distribution is
   identical and the shortcut is gone.
5. **C3:** predict presence from ROI **geometry only**, with no hidden states, to show the shortcut
   exists before the model even runs.
