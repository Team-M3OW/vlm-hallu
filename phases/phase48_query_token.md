# Phase 48 — Which query token's attention should the localiser read?

## 1. Research question
Every phase reads attention from the **last** prompt token, which for MCQ sits after the option text.
Attention from the question's content words is in the same forward pass, so it is free. Is it better?

## 2. Finding and contribution (plain English)
No — the deployed choice is already the best of seven, and this retroactively justifies what had been
an unexamined default.

The interesting part is *which* alternative is worst: attention from the **noun naming the target** is
near chance (`gt_pct` 0.367 against 0.500). **Localisation is not lexical lookup.** It emerges only
after the whole query has been integrated.

## 3. Numbers that changed
| query read-out | gt_pct | peak-in-GT | cov@0.15 |
|---|---|---|---|
| **last (deployed)** | **0.037** | **16.2%** | **40.4%** |
| content words | 0.102 | 13.6% | 34.8% |
| max-pool over question | 0.119 | 12.6% | 33.7% |
| **noun naming the target** | **0.367** | 8.4% | 17.4% |

## 4. Keep in paper: 5/10
A short but genuinely useful negative: it closes a plausible free improvement and supplies a nice
interpretability line about lexical lookup.

## 5. Experiment, step by step
1. Capture the full attention tensor once, then read **different query rows** from it — all seven
   read-outs cost the same single forward pass.
2. Locate the question span and its content words by decoding token ids and removing stopwords.
3. Score every read-out on `gt_pct` (chance 0.500 exactly), peak-in-GT, and coverage.
4. Compare each alternative to the deployed `last` read-out with paired bootstrap.
