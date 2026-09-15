# Phase 40 — Multi-window allocation: the intervention the causal argument was missing

## 1. Research question
Coverage makes no reference to **contiguity**. So two windows covering the evidence should work as
well as one. Does moving the second window's *location*, with everything else fixed, change accuracy?

## 2. Finding and contribution (plain English)
The control is what makes this decisive: `two_same` duplicates the *first* window, so against it
`two_diff` holds image count, total budget, per-image resolution and prompt format constant. **Only
the second window's location varies.**

Pooled, the effect is **zero**. But the pre-registered subgroup analysis — written before the run —
is exactly where the account says the effect must live: **+38.9pp** where the second window added
coverage, **+50.0pp** where it rescued an item from zero coverage, and **−4.0pp** where it added
nothing.

Both readings are kept because both were pre-registered: the pooled null bounds the **method**, the
dose–response upgrades the **mechanism**.

## 3. Numbers that changed
| subgroup | n | two_diff − two_same |
|---|---|---|
| 2nd window **added coverage** | 18 (9%) | **+38.9pp** [+11.1,+66.7] |
| added nothing | 173 (91%) | −4.0pp [−8.7,+0.0] |
| **rescued from zero coverage** | 14 (7%) | **+50.0pp** [+14.3,+78.6] |

Pooled: **+0.0pp** [−5.2,+5.2]. Mean coverage rises only 40.5% → 45.4%.

## 4. Keep in paper: 7/10
The interventional evidence for coverage, with the honest pooled null attached. The method
implication is the same one the paper keeps reaching: the second window must be **gated**, because
ungated it pays +38.9pp on 9% of items and −4.0pp on the rest.

## 5. Experiment, step by step
1. Find the top-2 attention peaks separated by ≥0.25 in normalised coordinates.
2. Build `two_diff` (peak1 + peak2) and **`two_same` (peak1 duplicated)** — the control that isolates
   location from resolution and format.
3. Add `two_rand` (peak1 + a random window) as the placement control.
4. Budget-gate all arms to the same **total** realized tokens.
5. Pre-register the subgroup analysis **in the analyzer, before the run**: subgroup membership is
   fixed by (peak1, peak2, box) before the model answers, so it is stratification on the
   intervention's dose, not post-outcome selection.
