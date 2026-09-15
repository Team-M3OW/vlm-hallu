# Phase 38 — Is coverage causal, or does attention just look at easy items?

## 1. Research question
The attention peak is chosen by the model, so coverage might predict the outcome without causing it.
Three instruments to test that.

## 2. Finding and contribution (plain English)
One instrument works, one is null, one is too weak — and reporting all three honestly is the point.

**Difficulty held fixed** is the strong one: stratifying on whether uniform got the item right, the
gap between covered and missed windows is **70.8pp within a single difficulty stratum**. On items
uniform got wrong, a covering window scores 83.8% and a missing one scores **13.0% — below the 25%
chance level**.

Window size (which we set, not the model) was reported as showing an interior optimum. **That was
later withdrawn** (Phase 39): bootstrapped, every CI crosses zero.

Box size as an instrument was **inconclusive** and we counted it as neither support nor refutation.

## 3. Numbers that changed
| uniform was | window | n | attn@0.15 accuracy |
|---|---|---|---|
| WRONG | missed | 46 | **13.0%** [4.3,23.9] |
| WRONG | covered | 37 | **83.8%** [70.3,94.6] |
| CORRECT | missed | 50 | 58.0% (destroys 42% of what worked) |
| CORRECT | covered | 58 | 87.9% [79.3,94.8] |

## 4. Keep in paper: 7/10
The difficulty-stratified table is strong and belongs in the mechanism section. The withdrawn
interior optimum should be kept visible as a retraction, not deleted.

## 5. Experiment, step by step
1. **Instrument (a):** stratify on `uniform` correctness — the purest available difficulty proxy —
   and compare covered vs missed windows *within* each stratum.
2. **Instrument (b):** sweep window size W, which we control. A causal coverage account predicts an
   interior optimum, since larger W raises coverage but lowers resolution.
3. **Instrument (c):** use box size as an instrument for coverage under random placement.
4. Report instrument strength, not just the result: (c) fails because a W=0.15 window is 2.25% of the
   image while even the largest box band is 1.2%, so it almost never fires.
5. Bootstrap everything — this is where (b) later collapsed.
