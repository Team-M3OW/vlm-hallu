# Phase 71 — Does a better proposal become a better answer?

**Classification: METHOD** (the end-task result)

## 1. Research question

Phase 70 raised top-1 evidence coverage from 39.3% to 52.9%, but coverage is a proposal-quality
metric. A better aim only matters if it changes answers. Does it?

## 2. Finding, in simple English

Yes, and by the amount predicted before the run.

We wrote the prediction into the script first. An earlier phase had measured what coverage is worth:
a window that misses costs 15.6 points, one that fully covers gains 37.1 — a 52.7-point swing. Move
13.6% of items across that line and you should get about **+7.2pp**. We measured **+8.4pp**.

On the benchmark: 60.2% → **68.6%** against the old proposer, and **56.5% → 68.6%** against the
plain model, which is **+12.0pp**.

The important split is by question type. On questions about a single object, the method reaches
75.7% and beats *even doubling the image budget* by 13 points — the first thing in this project to
beat simply spending more tokens. On relational questions ("what is left of the sign") it loses.
That is not a mystery: relational evidence sits in two places at once and one crop cannot cover both.

Three things make this believable rather than lucky. On the 79 items where the new and old proposers
happened to pick the **same** cell, the two arms score **identically** — they are the same
computation, so they must, and they do. All of the gain sits in the items where coverage actually
flipped: where the new proposal covers and the old missed, accuracy jumps **+51.6pp**; where both
cover, or neither does, the arms are indistinguishable. And random placement scores 40.3%, far
below, so this is not "any crop helps."

## 3. Numbers that changed

| arm | acc |
|---|---|
| uniform@300 (vanilla) | 56.5% |
| uniform@600 (compute-matched bar) | 63.9% |
| argmax@0.15 (old proposer) | 60.2% |
| **head@0.15 (ours)** | **68.6%** |
| rand@0.15 (control) | 40.3% |
| oracle@0.15 (ceiling) | 90.1% |

| contrast | Δ | CI |
|---|---|---|
| head − argmax | **+8.4pp** | [+2.6,+14.7] ✔ |
| head − vanilla | **+12.0pp** | [+4.2,+19.9] ✔ |
| head − bar (pooled) | +4.7pp | [−3.7,+13.1] ✗ |
| **single-region: head − bar** | **+13.0pp** | [+2.6,+23.5] ✔ |
| relational: head − bar | −7.9pp | [−21.1,+5.3] ✗ |

Coverage strata: flipped-to-covered **+51.6pp** (n=31); both cover −1.4; neither +1.2.
Internal control: identical proposal **+0.0pp [+0.0,+0.0]** (n=79); different proposal +14.3pp.

## 4. Keep in paper: **10/10**

This is the result the method rests on. It is pre-registered, it has an exact internal control, its
mediator was measured in an earlier phase and predicted this one quantitatively, and its boundary is
stated rather than discovered in review.

## 5. Experiment, stepwise

1. **Freeze the proposals first (phase 71a), on CPU.** Train the head out-of-fold with GroupKFold so
   an item's proposal is never produced by a model that saw it, and write `(cx, cy)` per item to
   disk. The GPU script then does no fitting at all and cannot leak.
2. **Six arms per item, one code path.** `uniform@300`, `uniform@600`, `argmax@0.15`, `head@0.15`,
   `rand@0.15`, `oracle@0.15`. Running the baselines in-line rather than quoting them from an
   earlier phase is what makes the budgets comparable.
3. **Measure tokens, never compute them.** Every arm read from `image_grid_thw`; all landed within
   2.0% of target, so no contrast is voided.
4. **Load in bfloat16 and assert finite logits.** A bf16 checkpoint in fp16 once produced all-NaN
   logits that argmax'd to index 0 and read as a clean negative.
5. **Set the bar at `uniform@600`, not `uniform@300`.** The method pays a localisation pass plus a
   crop pass, so the honest floor is what two passes of plain budget buy.
6. **Write the prediction into the file before running it** — +7.2pp, derived from §6D's coverage
   strata, a phase that never saw this arm.
7. **Check the internal control before reading the headline.** Items where both proposers chose the
   same cell must split 0.0. Compare the *cells*, not their coverage values: most cells have
   coverage exactly 0.0, so comparing coverage would lump every both-missed item into "identical"
   and the control would pass by construction.
8. **Split by category last.** V\*Bench is ordered by category, so any partial read is a biased
   subset — a lesson learned the hard way in phase 72.
