# Phase 31 — Does the attention proposer earn its second forward pass? (pre-registered negative)

## 1. Research question
Turn the localiser into a method: pass 1 at B₀=300 to get the attention peak, crop, pass 2 to answer.
Does it clear the compute-matched bar of simply spending two passes' worth of tokens uniformly?

## 2. Finding and contribution (plain English)
**No** — a clean pre-registered negative. The fixed policy scores 60.2% with a confidence interval
that includes zero, fails the 66.0% compute-matched bar, and captures only 10% of the oracle gain.

This is the failure that motivates everything downstream: if a fixed always-crop policy cannot pay
for itself, the method has to be about **when** to crop, not how.

## 3. Numbers that changed
- Fixed policy **60.2%**, CI **[−4.2, +11.5]** (includes zero).
- Compute-matched bar (uniform@600): **66.0%** — failed.
- Oracle capture: **10.0%**.

## 4. Keep in paper: 7/10
Keep as the negative that motivates the gate. Reporting it is also what makes the later positive
credible.

## 5. Experiment, step by step
1. Pass 1: uniform@B₀ with attention; average the ring-masked block mean over L16–L26.
2. Mask the outer ring — that is where the sinks live — and take the argmax.
3. Crop a square window of side W around the peak; refit to the **same** budget.
4. Pass 2: answer on the crop.
5. Score against `uniform@600`, the honest bar for a two-pass method, and compute oracle capture.
