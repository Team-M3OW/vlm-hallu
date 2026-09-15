# Phase 14/15 — Is anything about the object decodable inside the model?

## 1. Research question
The gate for every internal intervention: if we cannot decode the queried object from its own visual
tokens, no activation-level fix can work. Can we?

## 2. Finding and contribution (plain English)
**Phase 14:** yes — a linear probe on the object's own tokens reaches 0.88 AUROC even in the
sub-token stratum, vs 0.59 for a random region. So there is no capacity floor, and the earlier
prediction from Phase 13 ("nothing encoded there") was wrong.

**Phase 15 is the control that makes 14 interpretable**, and it cuts the result down. Phase 14
compared true-box against random-region, so it could be reading **objectness**, not the queried
category. Probing *within* positives — queried-category region vs a size-matched other-category
region in the same image and same forward pass — the signal drops to **~0.71 at L16**.

So category-specific encoding is real but **modest**. Internal steering has a target, but a faint one.

## 3. Numbers that changed
- Phase 14: **0.88 AUROC** in the sub-token stratum vs **0.59** for a paired random region.
- Phase 15, well size-matched pairs (|log2 ratio| < 0.5, n=275): **0.709 at L16**, CI [0.637, 0.778].
- Derivation layer for steering set to **L16–L22**, not the last layer.

## 4. Keep in paper: 6/10
Phase 15 is the keeper — it is a textbook example of a control that halves your own result. Report
14 and 15 together or not at all; 14 alone is misleading.

## 5. Experiment, step by step
1. **Phase 14:** extract hidden states at the object's own visual token positions; train a held-out
   linear probe to predict presence; compare against a paired random region in the same image.
2. Stratify by tokens-on-object to isolate the sub-token regime.
3. **Phase 15:** within positives only, pair the queried-category region against a
   **size-matched other-category** region *inside the same image and the same forward pass*.
4. Sweep layers L10/L16/L22/L27.
5. Restrict to well size-matched pairs to check whether size mismatch was creating or diluting the
   signal — it was **diluting** it.
