# Phase 12 — Rebuild the cohort in fp16

## 1. Research question
The cached `P(yes)` values came from a 4-bit model. Does the confident-denial cohort survive in fp16?

## 2. Finding and contribution (plain English)
A correctness rebuild, not a finding. Cached 4-bit confidences were not valid for selecting a
"confident" cohort, so the cohort was re-identified from scratch in fp16 across all RePOPE-clean
positives. This is bug #6 in the project's own log.

## 3. Numbers that changed
Cohort re-derived in fp16; all downstream phases use `phase12_clean_pyes.jsonl`.

## 4. Keep in paper: 1/10
Methods footnote at most. Worth one line only because quantisation-dependent cohort selection is a
trap others may hit.

## 5. Experiment, step by step
1. Re-run every RePOPE-clean positive in fp16.
2. Store `p_yes_fp16` per item.
3. Re-select the confident-denial cohort on the fp16 values.
4. Verify the overlap with the 4-bit cohort is poor enough to justify the rebuild.
