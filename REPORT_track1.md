# Track 1 — raising the method's accuracy (phases 118, 120, 121, 122)

Baselines: learned head OOF coverage **63.4% (Q3) / 54.5% (Q2)** at W=0.25; deployed argmax 46.1 / 39.3.
Adoption rule: a change is adopted only if it clears a bootstrap CI on BOTH models and exceeds the ~1pp
noise floor (§14Z). Everything below is OOF (GroupKFold by item, 3 seeds) unless stated.

## Phase 120 — is the miss recoverable from the head's own ranking? (on disk, both models)

| k | Q3 head top-k | Q3 argmax top-k | Q3 misses recovered | Q2 head top-k | Q2 argmax top-k | Q2 misses recovered |
|---|---|---|---|---|---|---|
| 1 | 65.6% | 47.8% | — | 58.6% | 40.3% | — |
| 2 | 70.4% | 57.0% | 14.1% | 64.5% | 54.3% | 14.3% |
| 3 | 76.3% | 65.6% | 31.2% | 65.6% | 60.8% | 16.9% |
| 5 | 79.0% | 72.0% | 39.1% | 69.9% | 69.4% | 27.3% |
| 10 | 84.4% | 79.0% | 54.7% | 78.0% | 79.0% | 46.8% |

**Head's top-1 and top-2 are within 1.5 cells of each other on 67.7% / 66.7% of items.**

### Verification design — argued from the numbers, NOT run
A top-1-vs-top-2 verification costs a third 300-token pass, so the bar moves to uniform@900. Top-2
buys **+4.8 / +5.9pp coverage**; at the project's measured conversion (~0.62pp accuracy per pp
coverage) that is **≈+3–4pp accuracy**. The bar's move 600→900 is 0.58 of a doubling on a
+7pp/doubling axis (§115) ≈ **+4pp**. Predicted net ≈ 0, and two-thirds of the time the second
candidate is the same neighbourhood as the first, so it adds no new coverage at all. **Not worth a
GPU run; recorded as a predicted null.** A top-k verification only becomes viable at k≥5 (+13/+11pp
coverage), which is 6 passes → bar uniform@1800, and the argmax top-k is nearly as good as the head
top-k there — i.e. the ranking advantage of the head is concentrated at k=1–3 and evaporates by k=10
on Qwen2 (78.0 vs 79.0). Multi-crop (4 crops, one pass) already failed −3.1pp (§14M).

## Phase 118 — up-weight the items where placement matters (encfail): NEGATIVE, both models

Sample-weight w on items with uniform@300 wrong AND oracle@0.25 right (72 / 79 of 191). Primary w=3.

| weight | Q3 cov all | Q3 cov encfail | vs w=1 (all) | Q2 cov all | Q2 cov encfail | vs w=1 (all) |
|---|---|---|---|---|---|---|
| 1 (incumbent) | 63.9% | 61.1% | — | 57.1% | 50.6% | — |
| **3** | 62.3% | 59.7% | −1.6 [−5.2,+2.1] | 54.5% | 46.8% | −2.6 [−6.3,+0.5] |
| 10 | 60.7% | 56.9% | −3.1 [−7.9,+1.6] | 50.3% | 41.8% | **−6.8 [−11.5,−2.6]** |

Monotonically worse with weight, on both models, and worse on the very items being up-weighted
(encfail −1.4 / −3.8 at w=3). Reading: the encfail subset is ~40% of items, so up-weighting it
mostly reduces effective sample size at n=191 — the constraint phase 102 already identified.

## Phase 122 — alternative training targets: NEGATIVE, both models

| target | Q3 | vs T0 | Q2 | vs T0 |
|---|---|---|---|---|
| T0 regress coverage@0.25 (incumbent) | 63.9% | — | 57.1% | — |
| T1 classify covers@0.25, balanced | 59.7% | **−4.2 [−7.9,−1.0]** | 56.5% | −0.5 [−3.7,+2.6] |
| T2 regress mean coverage@{0.15,0.25,0.35} | 60.7% | −3.1 [−6.8,+0.5] | 56.5% | −0.5 [−4.2,+3.1] |
| T3 regress coverage@0.15 | 63.4% | −0.5 [−5.2,+3.7] | 54.5% | −2.6 [−6.8,+1.6] |
| T4 regress coverage@0.25² | 60.7% | −3.1 [−6.8,+0.0] | 55.5% | −1.6 [−5.2,+1.6] |

Nothing beats the incumbent target on either model; the graded regression target is the right one
(classification throws away the partial-coverage signal and loses 4.2pp on Qwen3).

**Calibration note:** the incumbent's OOF coverage reads 63.9 / 57.1 here vs 63.4 / 54.5 in the
parent's runs (same data, different fold seeds / extraction). The run-to-run floor on this metric is
closer to **1–2.5pp** than the 1pp stated in §14Z; every contrast above is judged against its own
same-seed incumbent, so this does not affect the verdicts, but it should be stated in the paper.

## Phase 121 — the localisation prompt (GPU, both models, 5 variants × 191 items)

Prior art checked first: ViCrop (2502.17422) localises with a "locate the relevant region first"
prefix (tested as V1); LookWise uses extracted nouns as attention queries (phase 48: noun-token
attention is at chance here, not repeated); ZoomEye prompts per tree node (different mechanism).

| localisation prompt | Q3 argmax | Q3 head OOF | head vs V0 | Q2 argmax | Q2 head OOF | head vs V0 |
|---|---|---|---|---|---|---|
| **V0 question + options + answer-instruction (current)** | 46.1% | 61.8% | — | 39.8% | 56.0% | — |
| V1 ViCrop locate-first prefix + V0 | 43.5% | 61.8% | +0.0 [−4.2,+4.2] | 42.4% | 57.6% | +1.6 [−3.7,+6.8] |
| V2 question + options, **no instruction** | **3.7%** | 45.5% | −16.2 [−23.6,−8.9] | **0.5%** | 42.9% | −13.1 [−19.4,−7.3] |
| V3 question + options + "which region…?" | 4.2% | 40.8% | −20.9 [−28.8,−13.1] | 0.5% | 36.6% | −19.4 [−26.7,−12.0] |
| V4 bare question (72b's bug) | 9.4% | 44.5% | −17.3 [−23.6,−11.0] | 4.2% | 40.8% | −15.2 [−21.5,−9.4] |

### No variant beats V0 — V0 stands. But the sweep re-attributes the 72b effect.
The parent attributed the +19.3pp 4K swing to *restoring the options*. It is not the options: **V2 has
the options and collapses harder than the bare question** (argmax 3.7% / 0.5% — below the ~2% chance
rate of hitting the target cell). What matters is the **answer-instruction line**: the read-out is
the final prompt token's attention, and it localises only when that token is the *answer-emission
point*. Remove the instruction and the last token is the tail of option (D), whose attention is about
that text. Adding a "which region?" question (V3) does not restore it — the model is then answering a
different question.

> **Read attention at the answer position.** This generalises phase 48 (final token best of seven)
> and is a scope condition for every attention-guided localiser: a prompt that does not end at an
> answer-emission point yields an attention map that is near-useless for placement (−36 to −42pp
> on the argmax, both models). It plausibly explains published null results for attention-based
> grounding whose prompts end elsewhere (e.g. ACL Findings'25's "random beats attention" on RefCOCO).

Not run, worth one more variant: question + instruction **without** options, to confirm options
contribute nothing on their own. HR-Bench prompts (phases 72c/116) already end with the instruction.

## Verdict — single best deployable change: **none among those tested**

| lever | outcome |
|---|---|
| encfail up-weighting (118) | worse, both models; significant at w=10 on Q2 |
| alternative training targets (122) | none beat coverage@0.25 regression; classification −4.2 on Q3 |
| top-k verification (120, argued) | predicted ≈0 net: +3–4pp accuracy vs a bar that moves ≈+4pp; top-2 is adjacent to top-1 on 67% of items |
| localisation prompt (121) | V0 already optimal among 5; ViCrop's prefix null on both |

The head at n=191 is at a data-limited optimum on every axis tried today (architecture: phase 102;
features: phase 112; weighting: 118; target: 122; prompt: 121). **The one robust positive from this
track is a scope/mechanism statement, not a gain: the attention read-out requires the prompt to end
at the answer-emission point — worth a paragraph in the read-out section and an audit of every
localising phase for its instruction line, not just its options.**

Files: scripts/phase118_encfail_weighting.py (CI fix), phase120_topk_recovery.py,
phase121_prompt_sweep.py, phase121_analyze.py, phase122_training_target.py; data/phase121_prompts_{qwen3,qwen2}.jsonl;
logs/phase118.log, phase120.log, phase121_*.log, phase122.log. Nothing committed.
