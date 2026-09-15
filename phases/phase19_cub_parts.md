# Phase 19 — CUB part-level geometry

## 1. Research question
If whole birds are too large, are the *diagnostic parts* (head, beak) small enough to be sub-token?

## 2. Finding and contribution (plain English)
No. Even at part granularity CUB targets are far above the sub-token regime — the head is a median
20.5 tokens at B₀=300. This closed CUB as a venue permanently.

## 3. Numbers that changed
- CUB bbox: median **90.6 tokens** at B₀=300, **0.0%** sub-token.
- CUB head: median **20.5 tokens**, **0.0%** sub-token.
- For contrast, V\*Bench: median **0.32 tokens**, **71.2%** sub-token.

## 4. Keep in paper: 3/10
One line in the benchmark-selection paragraph, as justification for why V\*Bench and HR-Bench are
the venues and CUB is not.

## 5. Experiment, step by step
1. Load CUB part annotations; construct head and beak boxes.
2. Convert each to merged tokens at B₀=300 using the processor's own grid.
3. Compute the fraction of targets below one merged token.
4. Compare against V\*Bench and RePOPE on the same axis.
