# Phase 1 — Does accuracy collapse with object size, and is it just miscalibration?

## 1. Research question
Forced-choice `P(yes)` falls as the queried object gets smaller. Is that a genuine
evidence-integration failure, or could a better decision threshold fix it?

## 2. Finding and contribution (plain English)
Accuracy really does fall with object size in **both** architectures, so it is not one model's
quirk. And it cannot be recalibrated away: even a *perfect* per-size-bin threshold — which no real
system could have — buys almost nothing over a single global threshold. The model is not
mis-reporting a good internal decision; it does not have one.

This is the phase that makes the project about perception rather than calibration.

## 3. Numbers that changed
- Oracle per-bin threshold gain over a global threshold: **LLaVA +0.0162** (patch-token fraction),
  **LLaVA +0.0123** (pixel area fraction), **Qwen +0.0098** balanced accuracy.
- Direction replicates across LLaVA-1.5-7B and Qwen3-VL-2B.

## 4. Keep in paper: 6/10
Good early framing, and the oracle-ceiling argument is the rigorous part worth keeping. Compressible
to a short paragraph now that §13B gives the sharp threshold version of the same phenomenon.

## 5. Experiment, step by step
1. Bin POPE positives by object size (patch-token fraction and pixel-area fraction).
2. For each bin, compute balanced accuracy under one global `P(yes)` threshold.
3. Then compute balanced accuracy under the **best possible threshold chosen per bin** (the oracle).
4. Report the difference. A large difference would mean calibration; a tiny one means the evidence
   is not there to threshold.
5. Repeat on a second architecture to rule out a single-model artefact.
