# Phase 58 — Multi-crop in ONE pass: the method result

## 1. Research question
The GT cell ranks in the top 3.4%, so the attention ranking contains much better windows than the
top-1 rule uses — but selecting among candidates costs passes the budget axis outruns. Can we use
several candidates **without** paying for selection?

## 2. Finding and contribution (plain English)
Yes. The processor accepts **several images in one forward pass**, so k windows at B₀/k tokens each
costs one pass at the same total budget. No selection is needed: the model sees all k and answers
from whichever contains the evidence.

The decisive contrast is significant: **multi4 − top1 = +7.9pp** at identical passes and tokens.
And in the regime the mechanism predicts — single-region questions — multi-crop **beats the budget
axis** by +9.5pp, and by +10.6pp at twice the budget.

It fails on multi-region questions (−2.7pp), exactly where coverage says four windows still cannot
cover a dispersed evidence set — but the failure is *milder* than single-window (−9.2pp), which is
what more coverage predicts. The mechanism predicted both the win and the failure.

## 3. Numbers that changed
Coverage rises **40.4% → 55.7%** by handing over four windows instead of one.

| arm | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| top1@0.15 | 2 | 590 | 60.7% | 63.7% | −3.0 |
| **multi4** | **2** | **601** | **68.6%** | 63.9% | +4.7 |

**multi4 − top1 = +7.9pp [+1.0,+14.7] SIG.**

Single-region (n=115): multi4 **+9.5pp [+0.8,+17.4] SIG**; multi4_600 **+10.6pp [+2.8,+18.5] SIG**.

## 4. Keep in paper: 9/10
The method result. Scope that must travel with it: the pooled V\*Bench margin (+4.7pp) is **not**
significant; the significant budget-axis win is in the pre-specified single-region subgroup (n=115,
lower bound +0.8pp); the **gated** variant is not significant; and `multi3_scene` is negative, so the
gain comes from more candidates, not from retaining context.

## 5. Experiment, step by step
1. Take the top-k attention peaks separated by ≥0.20 in normalised coordinates.
2. Crop each at W=0.15 and fit each to **B₀/k** tokens.
3. Pass all k as **several images in one forward pass** with connector sentences.
4. Charge the localiser pass in full; verify the total matches the single-crop arm (601 vs 590).
5. Sweep k ∈ {2,3,4}, add a 2× budget variant and a scene+3-crops variant.
6. Score against the in-run uniform sweep, and separately on the pre-specified single-region subgroup.
