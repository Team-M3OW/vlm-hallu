# Phase 18 — Does the effect transfer to fine-grained recognition (CUB)?

## 1. Research question
POPE conflates "small object" with "fine detail". CUB birds fill much of the frame, so they separate
the two. Does the phenomenon appear where the object is **large** but the distinction is fine?

## 2. Finding and contribution (plain English)
No — and that is the point. CUB birds occupy a median 53.8 tokens (vs ~0.16 for POPE confident
denials). The tail does not transfer. This **bounds** the claim: the project is about
*under-resolved* targets, not about fine-grained discrimination in general.

A negative that does real work by preventing an overclaim.

## 3. Numbers that changed
- CUB median bbox area fraction **0.306**, **53.8 tokens on object**.
- POPE confident denials: ~0.0015 area, ~0.16 tokens.
- Effect base in CUB: ~2.2% — does not enlarge the phenomenon.

## 4. Keep in paper: 6/10
Keep as the scope-bounding negative. Later reinforced by the sub-token screen, which found **0.0%**
of CUB targets are sub-token — so CUB does not even qualify as a test case under the final predictor.

## 5. Experiment, step by step
1. Use the CUB **test split** with published labels.
2. Ask three questions per image: true species, **same-genus confusable** species, random
   different-genus species.
3. Keep the identical prompt and logit read-out used everywhere else.
4. Measure geometry first (area fraction, tokens on object) to confirm CUB separates size from detail.
5. Report the failure to transfer as a bound on the claim.
