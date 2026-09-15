# Phase 45 — Predict coverage from pass 1, for free

## 1. Research question
The gate detects coverage by cropping first, which is what costs the second pass. Can coverage be
predicted from pass 1 alone, before paying for the crop?

## 2. Finding and contribution (plain English)
Yes. The single best signal is **`peak`** — the max of the ring-masked attention map, a number the
localiser already computes on its way to the argmax. It predicts coverage at AUROC 0.788 at zero
cost, and it **beats a fitted multi-feature model**, which overfits at n=191.

That makes the policy adaptive: pay the second pass only when `peak` clears a threshold, so the
compute-matched bar moves with the firing rate instead of sitting at `uniform@600`.

**The honest caveat that must travel with it:** random routing at the same cost already earns +3.3pp
from the concavity of the uniform curve, so `peak`'s own contribution is **+1.9pp**, not +5.2pp.

## 3. Numbers that changed
| predictor of coverage | cost | AUROC |
|---|---|---|
| `conf(attn crop)` (deployed) | **a second pass** | 0.835 |
| **`peak` alone, unfitted** | **free** | **0.788** |
| all pass-1 features, 5-fold CV | free | 0.766 |
| conf(uniform) alone | free | 0.675 |

| policy | cost | acc | bar | margin |
|---|---|---|---|---|
| always-2-pass gate | 2.00× | 67.0% | 66.0% | +1.0pp |
| **adaptive, peak-gated** | **1.41×** | 66.5% | 61.3% | **+5.2pp** |

## 4. Keep in paper: 8/10
This is what makes the method affordable, and the random-routing decomposition is what keeps it
honest.

## 5. Experiment, step by step
1. Test whether pass-1 features predict coverage, with 5-fold CV and single-feature ablations.
2. **Fix the objective:** an earlier version maximised "accuracy per pass", which is maximised by
   never firing (0.565/1.00 beats 0.670/2.00) and drove the threshold to fire on 2% of items.
   The question is accuracy **at** a cost, so τ is chosen to hit a target firing rate.
3. Report the **whole cost–accuracy curve**, not one operating point.
4. Score each point against the **measured** uniform sweep at 300·c tokens.
5. Run random routing at the same cost as the control that separates the signal from curve concavity.
