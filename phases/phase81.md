# Phase 81 — Rank-only allocation: show the model several crops instead of guessing a size

**Classification: NEGATIVE (provisional — a prompting confound is under test)**

## 1. Research question

Four separate attempts to predict a continuous geometric quantity from internals have failed: token
budget from target size, budget from any free signal, window size from target size, and
attention-space reallocation. Meanwhile ranking cells works on two architectures.

So build a method that **only ranks**: take the top-4 separated cells, split a fixed budget evenly
across them, show all four in one pass. No size prediction, no budget prediction, no per-item
geometry at all.

Priced offline first: union coverage of the top-4, *and* above the encoding cliff at 75 tokens each,
is 63.4% against single-crop's 52.9% — **+10.5pp of answerable items at identical total budget**.
At the measured coverage exchange rate that predicts **+5.5pp**.

## 2. Finding, in simple English

**It does not work. Multi-crop is 3.1 points *worse* than a single crop.**

The prediction was +5.5pp and the measurement was −3.1pp, so the model that produced the prediction
is wrong, not merely imprecise. The likely reason: the coverage exchange rate was measured on
*single* crops, where coverage decides whether the evidence is visible at all. With four crops,
three of them are distractors, and the model must now both find the evidence *and* ignore three
irrelevant views. Nothing in the coverage model charges for that.

Ranking still matters inside the format — our top-4 beats four random crops by **+27.7pp** — so the
head is doing its job. The format itself costs more than the extra coverage is worth.

⚠ **A confound I found after the fact, and am testing before recording this as final.** An earlier
phase measured multi-crop at **+7.9pp**, the opposite sign. The clearest difference is the text
between images: that phase used *"Here is another zoomed-in crop from the same image"*; this one used
a bare newline, handing the model four pictures with no indication of what they were. Phase 81b
re-runs with the descriptive connector. If the gain returns, this negative is an artifact of my
prompt.

## 3. Numbers that changed

| arm | acc | tokens |
|---|---|---|
| uniform@300 | 56.5% | 294 |
| uniform@600 (bar) | 63.9% | 600 |
| **dcr_single@300** | **68.6%** | 294 |
| dcr_multi_k4 | 65.4% | 280 |
| argmax_multi_k4 | 66.5% | 280 |
| rand_multi_k4 | 37.7% | 280 |
| oracle@300 | 92.1% | 300 |

multi − single **−3.1pp [−9.4,+3.1]** · multi − rand_multi **+27.7pp [+17.8,+37.2]** ·
multi − bar **+1.6pp [−6.3,+9.4]** · oracle − multi **+26.7pp**.

Offline pricing that predicted the opposite: coverage-and-above-cliff 52.9% (k=1) → 63.4% (k=4),
turning over at k=5 (62.8%).

## 4. Keep in paper: **6/10**

Worth keeping as the honest closure of the allocation direction — and as a case where a quantitative
prediction derived from an earlier phase was falsified, which is more informative than a vague null.
Rating rises if 81b confirms it and falls to a footnote if 81b overturns it.

## 5. Experiment, stepwise

1. **Price the ceiling offline before spending GPU.** Union coverage of the top-k separated cells,
   intersected with the cliff condition at B₀/k tokens each. This is what produced the +5.5pp
   prediction and identified k=4 as the interior optimum.
2. **Enforce separation between crops** (min 0.20 of image width) so the k crops are not k views of
   the same spot.
3. **Freeze out-of-fold head scores to disk first**; the GPU script does no fitting.
4. **Three multi-crop arms at the same k:** ours, the deployed ranking, and random. Without the last
   two, "multi-crop helps" and "our ranking helps" are indistinguishable.
5. **Measure the multi-crop budget rather than assuming it.** The four crops came to 280 tokens
   against single-crop's 294 — slightly *fewer*, so the comparison is not flattered by overspending.
6. **Cache the fit scale per item.** All twelve crops share pixel dimensions, so one scale search
   serves all of them; without this the run takes 2.5 hours instead of 20 minutes.
7. **When the result contradicts an earlier in-project result, find the difference before writing
   the negative.** That is what turned up the connector text.
