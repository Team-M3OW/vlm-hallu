# Phase 63 — DoLa and DeCo as baselines ⛔ FIRST VERSION RETRACTED, corrected below

## 1. Research question
A literature survey found our logit-lens territory occupied. How do DoLa and DeCo actually perform in
our setting?

## 2. Finding and contribution (plain English)
The first run scored both against the **corrupted** final layer and reported that our L24 read-out
beat DoLa and matched DeCo. Retracted.

**Corrected, with the model's own logits as the final layer: no layer-decoding rule beats it.**
DoLa dynamic +0.0pp, DeCo best-of-8 −0.7pp, read-at-L24 +0.0pp — and DoLa/DeCo had their
hyper-parameters swept in their favour while ours was fixed.

This is a stronger result than the original error suggested, because it means the decoding channel
yields **nothing** here, not ~5pp. That converges with the three intervention nulls and with §13.

The survey context that makes it interpretable: ICLA reports DoLa *collapsing* on Qwen2.5-VL and DeCo
degrading, so layer methods tuned on LLaVA may simply not transfer to Qwen VLMs — our backbone family.

## 3. Numbers that changed
Corrected, sub-token, n=136:

| method | acc | vs true final |
|---|---|---|
| final layer (TRUE baseline) | 52.9% | — |
| DoLa dynamic (best bucket) | 52.9% | **+0.0** |
| DeCo (best band + α) | 52.2% | −0.7 |
| read at L24 (our retracted claim) | 52.9% | **+0.0** |

## 4. Keep in paper: 7/10
Keep the **corrected** comparison. "No published layer-decoding rule beats the final layer here"
directly supports the paper's causal argument, and citing DoLa/DeCo as baselines is required anyway
given the prior art.

## 5. Experiment, step by step
1. Reimplement DoLa's scoring rule: `log p_final − log p_premature`, premature layer chosen per item
   by max Jensen–Shannon divergence within a bucket, with the adaptive plausibility constraint.
2. Reimplement DeCo: `logits = final + α · max_prob(anchor) · anchor`, anchor from a late band.
3. **Sweep their hyper-parameters and report their best** — this favours them over our fixed rule.
4. Take the mature/baseline layer from the model's own logits, never from a lens.
5. State the scope: these are the published *scoring rules* applied to our MCQ read-out, not the full
   generation-time methods.
