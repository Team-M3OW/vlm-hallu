# Phase 2 — Is the size effect just centre-crop preprocessing?

## 1. Research question
Objects near the image edge might be lost to LLaVA's centre-cropping. Does position predict accuracy
independently of size — and does it survive in a model that never centre-crops?

## 2. Finding and contribution (plain English)
Position matters on its own, and it matters in **Qwen3-VL, which does no destructive cropping**.
So "it's just LLaVA's preprocessing" is ruled out as the whole story. This mattered because the
project's standing constraint is that preprocessing bugs are not an architectural finding.

## 3. Numbers that changed
- Area-weighted centroid distance from image centre predicts accuracy with bootstrap 95% CI
  **[−0.227, −0.092]**, surviving category fixed effects and an aspect-ratio covariate.
- Effect present in Qwen3-VL (no centre crop).

## 4. Keep in paper: 3/10
Mostly a ruled-out alternative. One line in Limitations or a footnote; it is superseded by later
work that never depends on position.

## 5. Experiment, step by step
1. Compute each object's area-weighted centroid distance from the image centre.
2. Regress accuracy on that distance with category fixed effects and aspect ratio as a covariate.
3. Bootstrap the coefficient.
4. Repeat in Qwen3-VL, whose processor preserves the full frame, to separate "edge objects are
   cropped away" from "edge objects are processed worse".
