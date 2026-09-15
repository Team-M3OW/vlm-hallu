# Phase 69 — Can any free pass-1 signal predict how many tokens an item needs?

**Classification: NEGATIVE** (closes a method family)

## 1. Research question

Phase 66/§14A established that a per-item *budget controller* — no crop, no pixel preprocessing,
just setting `image_grid_thw` per item — has **3.3× of real headroom** (oracle: 84.8% at 2418 mean
tokens vs the plateau's 84.8% at 7990), and that **target size does not unlock it**: with a perfect
ground-truth area fraction the rule `b = t*/area_frac` loses to flat uniform in 11 of 12 cells.

So the idea survived and the *signal* died. This phase asks the only remaining question: does
**anything** computable for free at pass 1 predict an item's required budget?

## 2. Finding, in simple English

No. And the way it fails is the interesting part.

Several free signals are genuinely *correlated* with how many tokens an item needs — the model's own
answer entropy correlates at ρ = 0.34, better than the target's actual size (0.22). But correlation
of that strength buys nothing. The trained predictor gets the exact rung right **9.4%** of the time
against a **35.6%** baseline you'd get by always guessing the most common rung — it is *worse than
a constant*. Its average error is 2 rungs, and a rung is a doubling, so it is off by about 4× in
budget on a typical item.

The reason is the shape of the problem. Items split into two piles: **68 of 191 need only 150
tokens** (they were never hard) and **37 need more than the whole ladder offers** (they are never
solvable by spending). The middle is thin. All of the oracle's 3.3× saving comes from knowing which
pile an item is in — and that is exactly the call no free signal makes.

This matters beyond the negative: it separates two claims the project had been running together.
`tokens_on_target` predicts whether an item is **encodable**. It does not predict the budget at
which it becomes **answerable**. The cliff is necessary, not sufficient.

## 3. Numbers that changed

| quantity | value |
|---|---|
| best univariate ρ with required budget (answer entropy) | 0.340 |
| area fraction, for reference | 0.223 |
| out-of-fold ρ, all 17 features | **0.254** |
| OOF exact-rung accuracy | **9.4%** vs **35.6%** majority baseline |
| OOF mean absolute error | **1.98 rungs** (≈4× in budget) |
| best controller margin vs the ladder at matched tokens | **+1.0pp** |
| same, size-only rule (already refuted) | **+1.8pp** |
| same, **shuffled-label control** | **+0.2pp** |
| threshold swept *on the test items* (cheating upper bound) | **+1.0pp** |

The internals never beat the refuted size rule. That is the tell.

## 4. Keep in paper: **7/10**

Not a headline, but load-bearing. It is the second of two measurements that close the
adaptive-budget family — §14A at the oracle-size level, this at the achievable level — and together
they let the paper say the internal method space is exhausted **on measured evidence** rather than
by assumption. It also forces a correction to §13B's synthesis sentence, which is worth more than
the negative itself.

## 5. Experiment, stepwise

1. **Join three existing artifacts, no GPU.** `phase30c_attn_maps_all.jsonl` (28 layers × 300 cells
   of image attention per item), `phase60_logit_lens.jsonl` (28 layers × 4 answer probabilities),
   `phase27_exchange_results.jsonl` (the 7-rung uniform ladder). The three files spell question ids
   differently (`direct_attributes/0` vs `direct_attributes/sa_4690.jpg::0`), so a positional join
   would fail silently — the join is asserted by requiring `gt_area_frac` to agree to 1e-9.
2. **Build 17 pass-1-free features.** Attention geometry (peak, top-5/top-20 mass, entropy, gini,
   spatial spread, peak/mean, 1-vs-5 ratio); answer state from the logit lens (confidence, margin,
   entropy, cross-layer argmax agreement, late-layer agreement, settling depth); free metadata
   (log pixels, aspect ratio, category). No feature uses the GT box.
3. **Build the label.** The smallest rung from which the item is correct *at that rung and every
   rung above*. Monotone stability is required because 22.5% of items flip correct→wrong going up
   the ladder; regressing on "cheapest rung that happened to be right" fits that noise and inflates
   everything downstream.
4. **Predict out-of-fold.** Gradient boosting, stratified 5-fold × 20 repeats. No item is ever
   scored by a model that saw it.
5. **Turn predictions into a controller and sweep it.** Apply a global offset to the predicted rung,
   −2 through +4, tracing the whole cost–accuracy curve rather than reporting one point.
6. **Score each point against the matched bar.** The uniform ladder interpolated at the controller's
   **own** mean spend. A controller emits a *distribution* of budgets, so the >10% budget gate does
   not apply — and this is exactly the mismatch that once turned +3.9pp [−0.3,+8.2] into a spurious
   "+5.2pp [+1.1,+9.4] significant", so it is enforced rather than remembered.
7. **Run three controls.** Shuffled labels (must land on the ladder — verifies the harness is not
   leaking); size-only (§14A's refuted rule, re-run inside the same harness for an apples-to-apples
   comparison); and a constant rung, which *is* the uniform ladder by construction.
8. **Read the verdict against the rule fixed before the run:** clears the ladder by >2pp at matched
   tokens with the shuffled control flat ⇒ signal exists. It does not.
