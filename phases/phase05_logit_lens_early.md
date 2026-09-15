# Phase 5 — Is the object "detected then suppressed" in late layers?

## 1. Research question
A natural story for the dissociation: the model detects the object mid-network and late layers
suppress it. Does an intermediate layer decode presence better than the final layer?

## 2. Finding and contribution (plain English)
No. The **final layer is the best layer**. The apparent mid-layer advantage we first saw was a
calibration coincidence, and once scored on an unbiased sample with per-layer oracle thresholds, the
layers are statistically tied. There is no hidden correct answer being thrown away late.

Worth noting: this conclusion was **re-derived and confirmed 60 phases later** (§12D), after a
normalisation bug briefly made us believe the opposite.

## 3. Numbers that changed
- Final layer AUROC **0.956**, beating every intermediate layer on a balanced 500-pos/500-neg sample.
- Per-layer oracle ceilings statistically tied at **~0.895–0.897**.

## 4. Keep in paper: 5/10
Valuable as an early elimination and as support for §12D. Compress to a sentence and cite the later,
better-instrumented version.

## 5. Experiment, step by step
1. Read hidden states at layers 15/18/20/22/25/28.
2. Project each through the model's unembedding and score presence.
3. **Balance the sample** (500 positive / 500 negative) — the first version was imbalanced and
   produced a spurious mid-layer peak.
4. Compute a per-layer oracle threshold ceiling, so a layer cannot win merely by being better
   calibrated.
5. Compare against the final layer.
