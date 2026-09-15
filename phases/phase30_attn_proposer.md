# Phase 30 (30/30b/30c/30d) — Can the model's own attention replace the oracle box?

## 1. Research question
Every result so far uses a GT box. For a method we need a proposer. Is the attention map good enough
to localise the target — and if not, why not?

## 2. Finding and contribution (plain English)
The localiser **is** in the attention map, but the **argmax is not the target**, and that gap is the
whole problem. The GT cell ranks in the top ~3.4% of cells against an exact chance of 50% — yet the
top-1 cell is almost never the target (0/191 at every layer initially).

The reason is attention sinks: positionally fixed cells absorbing 41–45% of the selected mass. This
is what sets up Phase 34's discovery that those sinks are a serialisation artefact.

Phase 30d's design decision worth keeping: the background is estimated **leave-one-out by
construction**, because a background fitted on all items and evaluated on them would use the test
set to build its own normaliser.

## 3. Numbers that changed
- `gt_pct` (rank of GT cell as a fraction of n_img; **chance = 0.500 exactly by construction**):
  best **0.034**.
- argmax-in-GT **17.3%** (157× chance) with LOO background; **15.7%** with ring-masking alone.
- Layer curve is **non-monotone**: L0–L15 give 0.153–0.398, L16–L26 give 0.027–0.119, L27 collapses
  back to 0.408. A *regime*, not "later is better".
- Sinks absorb **41–45%** of selected mass at 5 positions.

## 4. Keep in paper: 8/10
The `gt_pct` metric with exact chance is the right primary measure and should be kept. The
non-monotone layer curve matters because it is what makes the relative-layer-block transfer
non-trivial (validated later in Phase 44).

## 5. Experiment, step by step
1. **Pre-register the decision rule before running** (30a did: "median area_frac > 0.8 → DEAD").
2. Dump last-token attention over image tokens at **all 28 layers**, not a sample — the 5-layer
   sample had already found L20 far better than FastV's L2.
3. Define the primary metric as `gt_pct`, whose chance value is exactly 0.500 by construction, and
   the secondary as P(argmax inside GT).
4. Compare read-outs: raw, LOO-background-normalised, z-scored, and outer-ring-masked.
5. Estimate the positional background **leave-one-out**, on a coarse relative grid so it models
   position rather than content.
