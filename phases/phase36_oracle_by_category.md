# Phase 36 — Does a PERFECT crop also fail on relational questions?

## 1. Research question
Phase 35 claimed multi-region questions are un-croppable. The decisive test, and the one a reviewer
asks first: does the **oracle** crop — a perfect single-region proposal — also fail there?

## 2. Finding and contribution (plain English)
**No, it wins.** The oracle crop gains +19.7pp on relational questions. So a perfect crop of one
region does not fail, and the region-count explanation is dead.

What it revealed instead: the GT box for a relational question is the **union** of the objects
involved — 7.9× larger in area than for single-object questions. The oracle crop keeps the whole
evidence set; a crop centred on one attention peak does not. That observation becomes the coverage
mechanism.

## 3. Numbers that changed
- Oracle vs uniform: direct_attributes **+47.8pp** [+39.1,+56.5]; relative_position **+19.7pp**
  [+5.3,+34.2].
- Window sweep: relative_position stays negative at every W (−9.2 / −7.9 / −2.6 / −13.2) — a bigger
  window does not fix it.
- Median GT box area: relative_position **0.0048** vs direct_attributes **0.0006** (7.9×).

## 4. Keep in paper: 7/10
This is the test that killed our own hypothesis, and the union-box observation it produced is what
the whole mechanism section rests on. Keep both.

## 5. Experiment, step by step
1. Score the already-logged `oracle` arm **by category** — the data existed; only the analysis was
   missing.
2. Pre-register the reading: oracle also loses ⇒ region count confirmed; oracle wins ⇒ it is
   proposal/window quality and the hypothesis collapses.
3. Sweep the window size per category, to separate "needs two regions" from "needs a bigger window".
4. Measure GT box area and aspect ratio per category to test the union hypothesis.
5. Report main effects separately from the 2×2 interaction, which is underpowered.
