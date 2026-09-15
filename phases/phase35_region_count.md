# Phase 35 — Is the boundary about target size or about how many regions the question needs?

## 1. Research question
The size predictor got HR-Bench backwards. Splitting HR-Bench by its own category (both categories
are deeply sub-token, so size is held fixed), does **region count** predict the sign?

## 2. Finding and contribution (plain English)
It looked decisive: the sign flips by region count on two benchmarks, and the localiser works in both
categories, so `cross` fails because the crop **destroys the second region**, not because it misses.

**This explanation was superseded by Phase 36.** The oracle arm — a perfect crop of one region —
*wins* on relational questions, so "two regions are un-croppable" is false. The measurements here
stand and are reused; the causal reading is replaced by evidence-set coverage.

## 3. Numbers that changed
| benchmark | single-region | multi-region |
|---|---|---|
| V\*Bench | direct_attributes **+13.9pp** | relative_position **−9.2pp** |
| HR-Bench 4k | single −1.8pp (n.s.) | cross **−15.8pp** [−21.5,−10.0] |

Localiser works in both HR-Bench categories: attn − rand = +20.5pp (single), +6.8pp (cross).

## 4. Keep in paper: 5/10
Keep the measurements and the sign flip; **do not keep the region-count explanation**. It is a good
example of a hypothesis that survived one decisive test and died on the next.

## 5. Experiment, step by step
1. Score the **raw** `always attn@0.15` arm by category — not the gate, which mixes in the keyword
   rule and would confound the test with its own routing.
2. Use HR-Bench, where both categories are deeply sub-token, so target size is held fixed.
3. Add the placement control per category to separate "the localiser misses" from "the crop discards".
4. Check the same split on V\*Bench for replication.
5. Decouple size from category by splitting V\*Bench at the median target size and re-testing the
   sign flip within each size stratum.
