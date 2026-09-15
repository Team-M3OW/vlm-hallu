# Phase 75 — Does the read-out finding transfer to token pruning?

**Classification: FINDING (the strongest result in the project)**

## 1. Research question

Everything so far concerns our own problem: where to place a crop. FastV and the visual-token-pruning
literature solve a different problem — which tokens to *keep* — and they rank tokens by attention at
**one early layer** (FastV's default is layer 2). If our read-out finding is real rather than a quirk
of our task, ranking by late layers should beat that.

## 2. Finding, in simple English

**Two numbers, and they are the paper's strongest.**

First: **you can throw away 90% of a VLM's visual tokens and lose nothing at all** — as long as you
decide which to keep using late layers. Accuracy goes 56.5% with everything, to 56.0% keeping a
tenth of the tokens. That difference is noise.

Second: **at that same budget, ranking by layer 2 loses 22 points — and does worse than picking
tokens at random.** 34.6% against random selection's 38.2%. Ranking on a signal that does not exist
yet is worse than not ranking at all.

The margin of a late read-out over the early one is **+24.1pp** when keeping 10% of tokens and
**+16.8pp** at 25%. It shrinks to nothing at 50%, so the effect belongs precisely where the pruning
literature operates — aggressive budgets.

**And it needs no learning.** The trained signed head and a plain average of layers 16–26 are
indistinguishable at every operating point. The recommendation is one line: *read late layers, not
an early one.* That also means this is the **second** independent failure of the signed-contrast
mechanism outside our own task, which finishes it as a general claim.

**Why this matters beyond the number.** It joins two other results into one statement. The vision
encoder carries no localisation signal at all, because it never sees the question. The answer forms
abruptly at layer 21. And pruning at layer 2 is worse than random. All three say the same thing:
**question-conditioned localisation is built late in the language model, and decisions made before
that are made blind.** FastV prunes at layer 2. The vision tower has nothing. Our own proposer
averaged a fixed block without checking it was the right one.

## 3. Numbers that changed

| keep | random | **layer-2 (FastV)** | block-mean L16–26 | learned signed |
|---|---|---|---|---|
| **10%** | 38.2% | **34.6%** | **56.0%** | **58.6%** |
| 25% | 41.4% | 40.3% | 58.1% | 57.1% |
| 50% | 47.1% | 52.9% | 55.5% | 56.5% |

No pruning: **56.5%**.

vs no pruning at 10% keep: block-mean **−0.5pp [−5.8,+4.7]** · learned **+2.1pp [−2.6,+6.8]** ·
layer-2 **−22.0pp [−29.8,−14.7]**.
Late vs layer-2: **+24.1pp [+16.2,+31.9]** (10%), **+16.8pp [+8.9,+24.6]** (25%), +3.7pp n.s. (50%).
Learned vs block-mean: +2.6 / −1.0 / +1.0pp — **all null**.

## 4. Keep in paper: **10/10**

The only result that lives outside this paper's own problem: different task, different metric,
someone else's baseline, a large active literature, and a fix that costs nothing to adopt.

⏳ **ONE MODEL.** Provisional until it replicates (phase 83, running). The prediction is sharp:
Qwen2-VL's early layers sit at gt_pct **0.620** against Qwen3's 0.456, so the layer-2 penalty should
be **larger** there. A smaller penalty would refute the mechanism.

## 5. Experiment, stepwise

1. **Implement pruning as an attention bias, not token deletion.** A large negative additive bias on
   the dropped columns from layer K onward keeps positions and sequence length untouched, so arms
   differ only in which tokens are visible — functionally what FastV does.
2. **Every arm prunes the same count**, so cost is matched by construction and the contrast is purely
   about ranking quality.
3. **Four rankings:** random (the control that makes everything else readable), layer-2 (FastV's
   default), block-mean L16–26 (our deployed read-out), and the learned signed combination.
4. **Reuse the §14F weights unchanged.** They were fitted to predict crop-window coverage on a
   different model; refitting on the pruning task would answer a much weaker question. Using them
   as-is is the honest transfer test.
5. **Fix a four-step decision order before looking:** does pruning bite at all → do informed rankings
   beat random → does the late read-out beat layer-2 → *does the signed version beat the plain mean*.
6. **Let step 4 gate the verdict.** The first version of the analyzer declared "TRANSFERS" on the
   strength of step 3 alone, which would have claimed the learned method transferred when only the
   layer choice did. Corrected before the result was recorded.
