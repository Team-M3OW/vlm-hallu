# Phase 43 — Score the LLaVA models on the same five regions as the Qwen models

## 1. Research question
Phase 41 scored grid models on four disjoint content regions but newline models on a single `OTHER`
bucket holding 95% of positions. That is a weaker claim. Does the full five-region split agree?

## 2. Finding and contribution (plain English)
Partly — and it forced a retraction. The **separator** result survives, but **"rows carry no excess
mass" does not**: the bottom row is enriched 2.1–3.0× in both LLaVA models and 0.6–0.8× in both Qwen
models. A genuine cross-family divergence.

This phase also caught a bug in our own bucketing. Naively treating every inter-separator run as a
row swallows LLaVA's **separator-free base image** as "row 0", putting **59.8% of all positions** in
a bucket that can hold at most ~5%. That manufactured a spurious bottom-row result before the fix.

## 3. Numbers that changed
| region | OneVision | LLaVA-NeXT | Qwen3-VL | Qwen2-VL |
|---|---|---|---|---|
| newline separator | **2.0×** | **2.3×** | — | — |
| last col (pre-sep) | 1.3× | 1.2× | **3.4×** | **4.5×** |
| top row | **0.7×** | **0.7×** | 0.8× | 1.3× |
| **bottom row** | **3.0×** | **2.1×** | **0.6×** | **0.8×** |

## 4. Keep in paper: 6/10
Required for §4.1 to be honest. The retraction ("rows carry no excess mass" is not universal) must
travel with the universal claim.

## 5. Experiment, step by step
1. Segment rows using the separator positions.
2. **Keep only modal-length segments as rows**, excluding the base image explicitly — this is the fix
   for the 59.8%-in-one-bucket bug.
3. Bucket non-separator tokens exactly as in the grid models: first col, last col, top row, bottom
   row, interior.
4. Report enrichment with bootstrap CIs per region.
5. Sanity-check that a single row's area share is ≈ 1/n_rows; a later invariant test enforces it.
