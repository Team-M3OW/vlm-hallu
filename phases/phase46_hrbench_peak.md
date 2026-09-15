# Phase 46 — Does the adaptive policy rescue the 4K result?

## 1. Research question
On HR-Bench the always-2-pass gate **loses** to its own compute-matched control. Since that is a cost
problem, does gating fix it?

## 2. Finding and contribution (plain English)
Partly — it removes the failure without turning it into a win. With the V\*Bench-**transferred**
firing rate the margin goes from −3.1pp to **−0.5pp**, i.e. break-even.

The +1.9pp figure available at a different firing rate requires choosing that rate **on HR-Bench
itself**, so it is a bound, not a transfer result. I had to correct my own analyzer here: its verdict
line picked the best rate across the sweep, which is selection on the test set.

## 3. Numbers that changed
| policy | cost | margin vs its own bar |
|---|---|---|
| always-2-pass (deployed) | 2.00× | **−3.1pp** |
| adaptive @ V\*Bench-transferred 40% | 1.40× | **−0.5pp** |
| adaptive @ HR-Bench-best 20% | 1.20× | +1.9pp ← *tuned on this benchmark* |

At the 20% rate, `peak` routing beats random routing by **+2.8pp**, so the signal does work at 4K
even where the policy only breaks even.

## 4. Keep in paper: 6/10
Keep the break-even framing and the explicit separation of transferred vs tuned. This is one of
several places where the paper is stronger for reporting the weaker number.

## 5. Experiment, step by step
1. Log only the missing ingredient — `peak` per instance — and reuse Phase 33's existing answer arms.
2. One forward pass per **instance** (200), not per row (800), since the proposal does not depend on
   the option permutation.
3. Assemble the policy offline at several firing rates.
4. Make the **transferred** rate the headline; show the tuned rate separately and label it.
5. Compute the bar from HR-Bench's own measured uniform sweep.
