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

(phase 121 prompt sweep pending)
