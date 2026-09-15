# Phase 37 — Coverage: the continuous, category-free form of the boundary

## 1. Research question
Two candidate boundaries have each failed their own decisive test. Is there one quantity that
subsumes both — and can it be measured per item without any category label?

## 2. Finding and contribution (plain English)
Yes: **the fraction of the question's evidence set that the reallocated window actually covers.**
Pooling all items and discarding the category labels, the accuracy change is a clean dose–response in
coverage, crossing zero at about 25%.

And the practically important number falls out of it: **half of all items have zero coverage** — the
window misses the target entirely — and on those, allocation *costs* 15.6pp. The headline +12pp is a
large win on covered items netted against a real loss on missed ones.

## 3. Numbers that changed
| coverage of GT by the deployed window | n | uniform | attn@0.15 | Δ |
|---|---|---|---|---|
| **0% (missed)** | 96 | 52.1% | 36.5% | **−15.6** [−26.0,−5.2] |
| 0–25% | 11 | 63.6% | 36.4% | −27.3 |
| 25–75% | 12 | 75.0% | 100.0% | +25.0 |
| **100%** | 62 | 58.1% | 95.2% | **+37.1** [+24.2,+50.0] |

Coverage by category: direct_attributes 52.6%, relative_position 22.4% — which tracks the sign flip.

## 4. Keep in paper: 9/10
The mechanism section's core table. The "50% of windows miss" line is the single most useful thing
the paper says to a practitioner.

## 5. Experiment, step by step
1. Compute coverage on the window that was **actually evaluated**, using the logged `peak_frac` —
   not a recomputed one.
2. Replicate the deployed crop's edge-clamping **exactly**, or the coverage is measured on a window
   that never existed. (A later invariant test enforces this.)
3. Bin by coverage, pooling all categories, and report `uniform`'s own level per bin so ceiling
   effects are visible.
4. Check whether category adds anything **within** a coverage bin — if coverage is the mediator, the
   categories should agree there.
5. State clearly that coverage uses the GT box and is therefore an **explanatory variable only**:
   no method reads it, no hyper-parameter is selected on it.
