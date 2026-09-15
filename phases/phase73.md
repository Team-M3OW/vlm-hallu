# Phase 73 — Why does the re-ranking work, and can it be one line?

**Classification: FINDING (mechanism) + METHOD (the distilled variant)**

## 1. Research question

Two things were open. Is the head just picking better layers than the deployed block? And can a
65-feature tree be collapsed into something you could actually ship?

## 2. Finding, in simple English

**It is not layer selection. It is layer contrast, and that is a sharper claim.**

If the problem were simply "the wrong layers," then the best single layer should be much better than
the average. It is barely better: L17 alone gets 41.9% against the 11-layer average's 39.3%. So no
single layer rescues you.

What the learned version does is give layers **negative** weights. L17, L19, L24, L16 come out
positive; L27, L26, L15, L11 come out negative. And the final layer is not merely useless — its
ranking of the target is **0.529 where pure chance is 0.500**, so it is slightly *anti*-correlated.
The deployed read-out adds it in at weight +1 along with everything else. The learned read-out
subtracts it.

**A mean can only add. That is the entire defect in one sentence.** And it gets worse the more you
average: three layers 45.5%, eleven layers 39.3%, all twenty-eight **36.6%**.

**The shippable version works, but only halfway.** Replace `mean(L16..L26)` with a weighted sum
`Σ wᵢ·Lᵢ` — 28 numbers, a one-line change in any VLM's read-out — and you recover 46% of the gain
(+6.3 of +13.6pp). The other half needs the head's non-linearity and its neighbourhood features. So
there is a simple variant and a strong variant, and the gap between them can be reported honestly
rather than papered over.

**It also resolves a contradiction we had.** Phase 70b found that telling the head where the
serialization sink is bought exactly nothing, which looked like it refuted the sink story. It does
not. Across layers, sink contamination predicts layer quality (ρ = +0.522), and the learned weights
track layer quality (ρ = −0.686). So the sink matters — it decides which **layers** are
trustworthy, not which **cells** are. That is why a per-cell sink indicator was redundant. Both
results stand and now agree.

**One thing that is null:** how much the layers disagree does not tell you whether to trust the
proposal (AUROC 0.572, no better than the peak height we already had). It is not a free confidence
signal.

## 3. Numbers that changed

| proposer | top-1 coverage |
|---|---|
| deployed block-16-26 mean | 39.3% |
| best single layer (L17) | 41.9% |
| **learned linear reweighting (OOF)** | **45.5%** |
| full head, 65 features | 52.9% |

Averaging the k individually-best layers: k=1 41.9% · k=3 **45.5%** · k=7 41.4% · k=11 39.3% ·
k=28 **36.6%**.

Cross-layer correlations: weight vs gt_pct **−0.686** · sink vs gt_pct **+0.522** · weight vs sink
−0.306. Sink enrichment by depth: early **10.49×**, mid 6.10×, late 5.75×. Per-layer gt_pct: early
0.456, mid 0.413, late 0.409 — **no single layer is a good localiser**.

## 4. Keep in paper: **10/10**

This is the mechanism section. It converts "a learned head helps" into a statement about why the
standard read-out is wrong, which is the part that generalises beyond this task.

## 5. Experiment, stepwise

1. **No GPU.** Everything is computed from attention maps already extracted for all 191 items.
2. **Screen every single layer as a proposer.** 28 arms, same ring mask, same coverage definition.
   This is what rules out "the head just found a better layer."
3. **Fit a linear reweighting out-of-fold**, grouped by item, so the 28 weights are never evaluated
   on an item that trained them. Ridge, 5 folds, weights averaged across folds for reporting.
4. **Sweep the averaging width.** Rank layers by their individual performance, average the top k,
   and vary k. Using the *individually best* layers is deliberately the best case for a mean — if it
   still degrades, the problem is averaging and not layer choice.
5. **Characterise each layer independently:** how well it ranks the target (gt_pct, where 0.500 is
   chance by construction), and how much mass it puts on the trailing column relative to a uniform
   map.
6. **Correlate the learned weights against both** across the 28 layers. This is what separates "the
   sink is irrelevant" from "the sink acts through layer quality."
7. **Test disagreement as a confidence signal** — spread of per-layer argmax positions vs whether the
   proposal covers — and report the null.
8. **Note the discrepancy rather than bury it:** the peak-height signal reproduces at AUROC 0.576
   here against 0.788 quoted in an earlier section, under a different target definition. Flagged for
   reconciliation before either number is cited.
