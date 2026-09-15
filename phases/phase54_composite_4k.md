# Phase 54 — Scene + crop composite: never discard the scene

## 1. Research question
Crop-only fails at 4K because it throws away 97.75% of the image. Does passing the **scene and the
crop together**, as two images in one pass at the same total budget, fix it?

## 2. Finding and contribution (plain English)
It **halves the damage but does not win**: from −17.3pp to about −3.5pp. Still negative.

The instructive detail is that `comp_150_150` (48.4% at 589 tokens) is worse than `uniform@300`
(52.8% at 293 tokens) — splitting the budget across two images costs something on its own. The
V\*Bench precedent that motivated this (85.9% at 292 tokens) used the **oracle box**, so its crop was
always on target; with a real proposer the composite inherits the proposal's errors.

## 3. Numbers that changed
| arm | tokens | acc | bar | margin |
|---|---|---|---|---|
| comp_150_150 | 589 | 48.4% | 52.0% | −3.5 |
| comp_300_300 | 886 | 48.0% | 54.8% | −6.8 |
| comp_300_300 W=0.35 | 886 | 47.7% | 54.8% | −7.2 |
| gated variants | 411–530 | — | — | −3.8 to −6.4 |

## 4. Keep in paper: 4/10
One row in the "four attempts at 4K" table. Its value is showing that the failure is not simply
"we discard the scene".

## 5. Experiment, step by step
1. Pass the scene and the crop as two images in one forward pass with a connector sentence.
2. Vary the budget split (150/150, 300/300, 150/450) and the window size.
3. Charge the localiser pass in full.
4. Apply the peak gate offline at the transferred firing rate.
5. Compare against uniform arms measured in the same run.
