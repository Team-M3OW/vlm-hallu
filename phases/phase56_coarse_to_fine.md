# Phase 56 — Coarse-to-fine localisation at 4K

## 1. Research question
At 4K the localiser's grid cell spans ~233px, so the peak is imprecise. Localise **twice** — once on
the full image, then again inside a mid-level crop at 2.9× finer resolution. Does precision fix it?

## 2. Finding and contribution (plain English)
**No, and the null is unusually clean.** The refinement genuinely works as engineering: cell size
drops 237px → 83px and the peak relocates a median of 489px. The accuracy effect is **exactly zero**.

The isolating contrast is the point: same window size, same final budget, differing only in whether
the centre came from one localisation or two. **+0.0pp, CI [−3.0,+2.9].** The fine peak is no more
on-target than the coarse one.

This is the fourth failed attempt to win at 4K, and it is what justified claiming a boundary rather
than trying a fifth.

## 3. Numbers that changed
| arm | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| mid@0.35 | 2 | 586 | 44.2% | 59.9% | −15.7 |
| coarse_fine@0.15 | 2 | 587 | 42.6% | 59.9% | −17.3 |
| c2f_fine@0.15 | 3 | 880 | 42.6% | 62.5% | −19.8 |
| gated c2f_fine | 1.4 | 529 | 57.0% | 58.9% | −1.9 |

**c2f_fine − coarse_fine = +0.0pp [−3.0,+2.9].**

## 4. Keep in paper: 6/10
The decisive member of the four-attempts table, because it isolates proposal precision and rules it
out as the binding constraint.

## 5. Experiment, step by step
1. Localise on the full image (coarse peak, cell ≈ 237px).
2. Crop W=0.35 around it, refit, and localise **again** (fine peak, cell ≈ 83px).
3. Map the fine peak back into original image coordinates.
4. Crop the **same** W=0.15 window around the fine peak and answer.
5. The contrast that matters is c2f_fine vs coarse_fine — identical window and budget, one
   localisation vs two.
6. Join the bar arms from a previous run on identical items, with an asserted consistency check.
