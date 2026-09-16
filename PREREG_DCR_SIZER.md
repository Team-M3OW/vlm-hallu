# Pre-registration — DCR + free pass-1 window sizer

**Written 2026-09-17, before any Qwen2-VL data for this configuration exists.** Phase 97 (the W
transfer) is mid-run at n=41/191 and carries no attention features; phase 100 (Qwen2-VL feature
extraction) has not been written. Committed now so the git timestamp precedes the result.

## Why this exists

On Qwen3-VL two configurations found today clear the equal-compute bar that DCR has never cleared.
Both are marginal, both were found after trying variants, and neither is replicated. Without a
pre-registration the next step is indistinguishable from shopping until something clears.

| configuration | acc | vs uniform@600 |
|---|---|---|
| DCR always-crop, fixed W=0.25 (the incumbent) | 71.7% | +7.9pp [−0.5,+15.7] ✗ |
| **+ free per-item window sizer (OOF)** | **72.8%** | **+8.9pp [+0.5,+17.3]** ✔ |
| + `top1_frac` routing gate (OOF threshold) | 72.8% | +9.6pp [+2.0,+17.7] ✔ |

## The one configuration being carried forward

**PRIMARY — the sizer, not the gate.** Chosen because it is the phase-78 headroom lever, it uses all
six features rather than the best of four, and its first formulation failed for a label-construction
reason (ties collapsing to W=0.15) rather than a failed hypothesis.

    features   peak · peak_over_median · top1_frac · top5_frac · entropy · entropy_norm
               all from the pass-1 ring-masked attention map, all free, NONE derived from the label
               or the GT box
    model      one logistic correctness model per W (C=0.5, max_iter=2000), standardised features
    selection  per item, pick the W with the highest predicted P(correct)
    grid       W in {0.15, 0.25, 0.35, 0.50, 0.70}
    validation 5-fold out-of-fold, folds assigned by shuffled item order, fit inside training folds
               only
    PRIMARY    sizer - uniform@600, matched compute, tokens MEASURED (>10% drift voids it)
    SECONDARY  sizer - fixed W=0.25  (on Qwen3-VL this is +1.0pp [-2.1,+4.2], i.e. NOT significant --
               the sizer's own contribution is small; what it does is push the bar contrast over)

## Decision rule

- **CI clear of zero on Qwen2-VL** → "DCR beats the equal-compute baseline" becomes a **2-model
  claim** and the ledger's REJECTED entry is corrected to SURVIVED.
- **CI spans zero** → the claim stays rejected. It is then reported as *one model only*, and the
  Qwen3-VL result is reported as **exploratory**, with the number of configurations tried stated in
  the paper.
- The `top1_frac` gate is **exploratory** and may NOT be substituted for the sizer if the sizer
  fails. Swapping it in after seeing the sizer fail is the exact move this file exists to prevent.

## What must not happen

1. No re-tuning of the feature set, the W grid, the fold count or the model on Qwen2-VL.
2. No reporting of the sizer against `uniform@300`. The bar is matched compute.
3. If phase 97 shows fixed W=0.25 already transfers on Qwen2-VL, that is a **separate** result and
   does not substitute for this one.
4. The Qwen3-VL numbers above are frozen. They are what they are, including the non-significant
   +1.0pp.

## Status

- Phase 97 — Qwen2-VL W ∈ {0.15,0.25,0.35} end-task arms — **running**, n=41/191.
- Phase 100 — Qwen2-VL pass-1 attention features — **to be written and run after 97**.
- Phase 98 (gate) and 99 (sizer) on Qwen3-VL — done, numbers above.
