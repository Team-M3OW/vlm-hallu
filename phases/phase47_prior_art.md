# Phase 47 — Prior art at a matched realized budget (V\*Bench)

## 1. Research question
Every crop/zoom method reports gains against a baseline that spends **fewer** tokens than it does.
Charged for every token across every pass, do any of them beat simply spending that budget uniformly?

## 2. Finding and contribution (plain English)
**No.** Zoom Eye has the highest raw accuracy in the table and still loses to a single uniform image
of its own budget. The grounding family loses by 6.2pp. Ours is the only policy with a positive
margin, and it is the cheapest arm.

The grounding failure is predicted by our own mechanism: its boxes parse on 98% of items but have
**median IoU 0.029**, because it must ground on the same downscaled view that cannot resolve the
target.

## 3. Numbers that changed
| policy | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| grounding (Chain-of-Spot · Visual CoT · DualFocus) | 2.00 | 595 | 57.6% | 63.8% | **−6.2pp** |
| Zoom Eye (tree search) | **9.00** | **2664** | 75.9% | 77.8% | **−1.9pp** |
| ours, fixed | 2.00 | 591 | 61.3% | 63.8% | −2.5pp |
| **ours, adaptive** | **1.40** | **414** | 62.8% | 60.0% | **+2.8pp** |

Cost-normalised: ours **1.52** acc/1k tokens vs Zoom Eye **0.28**.

## 4. Keep in paper: 9/10
The comparison nobody runs, and the paper's most quotable claim. **No margin is individually
significant at n=191** — that qualification must travel with it until Phase 53 supplies significance.

## 5. Experiment, step by step
1. Reimplement each **policy** on one common backbone, so the comparison isolates the policy from the
   base model. State plainly that this cannot reproduce gains coming from fine-tuning.
2. Charge every arm for every image token across every pass; count a generate call's prefill as one
   image pass.
3. Build the bar from uniform arms measured **in the same run on the same items** — using another
   run's sweep confounds the comparison with item subset.
4. Report both passes and tokens, since tree search is cheap per pass and expensive in total.
5. Diagnose failures mechanistically (e.g. measure the grounding arm's IoU) rather than just scoring
   them.
