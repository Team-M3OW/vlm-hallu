# Phase 9 — How much does resolution alone recover?

## 1. Research question
If the two-image format is suppressing the benefit, what does the crop recover **on its own**?

## 2. Finding and contribution (plain English)
Far more: **40.4%** versus Phase 7's 12.8%, with a false-positive rate of only 3.5% on genuinely
absent objects. So under-resolution is a substantially larger cause than Phase 7 reported, and
Phase 7 had accidentally measured "resolution minus a distractor".

## 3. Numbers that changed
- Crop alone: **40.4%** recovery, FP **3.5%**.
- vs `[full + connector + crop]`: **12.8%**.
- Paired difference **+27.7pp**, CI **[+20.6, +34.8]**, 100% of resamples.

## 4. Keep in paper: 6/10
This is the number to quote for the resolution effect, but it is best presented *inside* Phase 10's
factorial rather than alone.

## 5. Experiment, step by step
1. Re-run the Phase 7 cohort with the crop presented as the **only** image.
2. Keep the identical question and read-out.
3. Score recovery **and** the false-positive rate on absent objects — a recovery number without an
   FP number can be pure yes-bias.
4. Compute the paired difference against the two-image arm and bootstrap.
