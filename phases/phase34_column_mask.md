# Phase 34 — The attention sink is columnar, not cornered

## 1. Research question
The sink was described as sitting at image corners. Stratified by grid shape, is that actually true —
and does it imply a better mask?

## 2. Finding and contribution (plain English)
"Corners" was **wrong**, and wrong in a way that matters. The effect is **columnar**: the last column
carries 5.1× its fair share of attention, the first column 2.9×, and the top and bottom rows carry
**no excess at all** (0.9× and 0.8×). "Corners" was a strong column effect intersected with a null
row effect.

Two tests fix the mechanism. A **transpose test** (14×21 vs 21×14, identical token count, geometry
transposed) shows identical positions in (row, col) space but only 4/8 shared absolute indices — so
these are not register tokens. And replaying modal-grid sink indices where they land in the interior
gives **1.1% against 3.3% chance**, i.e. *below* chance.

The reading: column `gw−1` precedes a raster row-wrap and column 0 follows one, while top and bottom
rows are ordinary raster positions. The "spatial" sink is a **1-D sequence artefact wearing a 2-D
costume**.

## 3. Numbers that changed
| region | mass | area | enrichment |
|---|---|---|---|
| last column | 25.3% | 5.0% | **5.1×** |
| first column | 14.2% | 5.0% | 2.9× |
| top row | 5.5% | 6.3% | **0.9×** |
| bottom row | 5.3% | 6.3% | **0.8×** |

Mask comparison against **area-matched random twins**: columns give **88% of the ring mask's effect
at 44% of the cost** (0.99 vs 0.50 pp per % of cells deleted). But column-only does **not** beat the
ring mask outright (gt_pct Δ +0.023 in the ring's favour).

## 4. Keep in paper: 8/10
The mechanism is a genuine correction to a widely-repeated description. The honest split must be
kept: **mechanism confirmed, method improvement not**.

## 5. Experiment, step by step
1. Stratify the attention dumps by grid shape — the modal shape (14×21) dominates the pooled counts
   and makes absolute and relative position look identical.
2. Compute mass share and area share per region; report **enrichment** so regions of different size
   are comparable.
3. **Transpose test:** compare 14×21 against 21×14 (same n, transposed geometry). Register tokens
   predict identical absolute indices; positional effects predict identical (row, col).
4. **Interior replay:** take the modal grid's sink indices, replay them on other shapes, and keep
   only those landing in the interior. A register token should re-fire; these fire below chance.
5. **Score every mask against an area-matched random twin**, because `gt_pct` is a rank and deleting
   cells improves it mechanically regardless of what was deleted.
