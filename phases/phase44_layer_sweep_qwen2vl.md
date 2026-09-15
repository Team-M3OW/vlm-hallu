# Phase 44 — Is the transferred layer block validated, or just assumed?

## 1. Research question
The relative block [0.55L, 0.95L] was reverse-engineered from Qwen3-VL's L16–26 — a **non-monotone**
regime, which is exactly the kind of thing that should not be rescaled to another architecture on
faith. Is it right for Qwen2-VL?

## 2. Finding and contribution (plain English)
Validated, and convincingly: **8 of Qwen2-VL's 8 best layers fall inside the transferred block**, and
the same regime shape appears (collapse around L14, rise again at L27).

So the cross-architecture numbers stand **as measured**, not as a lower bound under a borrowed
hyper-parameter.

## 3. Numbers that changed
- Best 8 layers by `gt_pct`: 16, 18, 19, 20, 21, 22, 23, 26 — **8/8 inside** L15–L26.
- Median `gt_pct` **0.041 inside** the block vs **0.390 outside** (9.5×).
- Best single layer L19: `gt_pct` **0.010**, argmax-in-GT **16.7%**.

## 4. Keep in paper: 6/10
One paragraph, but it closes a real objection — otherwise every cross-architecture number is
"measured at a depth chosen on a different model".

## 5. Experiment, step by step
1. Sweep **every** layer of Qwen2-VL on `gt_pct` (chance 0.500 exactly), with the deployed ring mask.
2. Use no accuracy arm, so nothing can be tuned to flatter the method.
3. Count how many of the best layers fall inside the transferred block.
4. Compare median `gt_pct` inside vs outside.
5. Pre-register the reading: if the best regime is elsewhere, the method's numbers become a lower
   bound and must be labelled as such.
