# Phase 21 — Where does allocation stop paying?

## 1. Research question
Every allocation result so far was measured where allocation should win. Stratifying RePOPE by
target size over a 400× range, where does the benefit cross zero — if it does?

## 2. Finding and contribution (plain English)
The positive half reproduces cleanly and monotonically over a 400× size range, on a second benchmark
with a different task format (yes/no, not MCQ) and a different annotation source. Crucially the
false-positive rate is **unchanged**, so the gain is not yes-bias.

The negative half is **not testable here**, and we say so rather than claiming support: uniform
saturates at 97.5–99.2% on every resolvable stratum, and no crossing can be observed without headroom.

This data sat unanalysed for weeks before being picked up — the analysis, not the run, was the
missing piece.

## 3. Numbers that changed
| tokens on target | n | uniform | allocation | Δ | vs random |
|---|---|---|---|---|---|
| <0.5 | 120 | 57.5% | 74.2% | **+16.7** [+7.5,+25.8] | +15.0 |
| 0.5–2 | 120 | 89.2% | 95.8% | +6.7 [+0.8,+12.5] | +13.3 |
| 2–8 | 120 | 98.3% | 100.0% | +1.7 | ceiling |
| 8–32 | 120 | 97.5% | 98.3% | +0.8 | ceiling |
| >32 | 120 | 99.2% | 98.3% | −0.8 | ceiling |

FP unchanged: **7.5% → 7.5%**. Budget gate spread 1.7%.

## 4. Keep in paper: 7/10
The second-benchmark replication of the exchange rate, with a different chance level (50%, not 25%)
that must be flagged whenever it appears next to MCQ numbers.

## 5. Experiment, step by step
1. **Stratify by design** on tokens-on-object — five strata — rather than sampling broadly and
   binning after, because the tails are where the crossing would live.
2. Use **all confidence levels**, not the confident-denial cohort: a negative delta is unobservable
   on items uniform already gets wrong.
3. Carry `alloc_random` through every stratum so the decider is present along the whole curve.
4. Add a third arm varying the 25/75 vs 50/50 split, to separate "allocation fails" from "the split
   is wrong".
5. Report `uniform`'s own level per stratum, so ceiling effects are visible rather than inferred.
