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

(remaining sections filled as phases 118 / 121 / 122 land)
