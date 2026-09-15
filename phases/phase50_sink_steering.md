# Phase 50 — Suppress the serialisation sink inside the forward pass

## 1. Research question
The sink absorbs 25% of the last token's attention over image tokens on a position that carries no
image content. Ring-masking removes it from our **read-out**, but the model still spends it.
Suppress it in the forward pass — does accuracy improve?

## 2. Finding and contribution (plain English)
**No.** Suppressing the sink is statistically indistinguishable from suppressing random or interior
image tokens. And it is a genuine negative, not a dead hook: suppressing the same number of **prompt**
tokens costs up to 17.3pp, so the intervention machinery bites hard.

The secondary result is the more interesting one: **biasing *any* 4.8% of image tokens moves accuracy
by ≤2pp, while the same number of text tokens costs 17pp.** The answer is nearly insensitive to which
image tokens receive attention — a sharp statement of how little each image token contributes.

Every arm is one pass at identical tokens, so there is no compute bar to clear.

## 3. Numbers that changed
Δ vs baseline (56.5%), biasing 14 of 294 image tokens:

| arm | b=−1 | b=−2 | b=−4 | b=−8 |
|---|---|---|---|---|
| **last_col (the sink)** | −0.5 | −1.6 | −1.6 | −1.6 |
| interior (content control) | −0.5 | +0.0 | −1.6 | −1.6 |
| rand_cols | +0.5 | +0.0 | −1.0 | −0.5 |
| **text_tokens** | **−6.8\*** | **−14.7\*** | **−16.8\*** | **−17.3\*** |

## 4. Keep in paper: 7/10
Part of the three-intervention causal backbone. The image-vs-text asymmetry is quotable on its own.

## 5. Experiment, step by step
1. Patch `eager_attention_forward` to add a per-module additive bias over key positions before
   softmax.
2. **Self-test the patch:** a zero bias must be a provable no-op, and a −8 bias must provably move the
   output. Both asserted before the run.
3. Build controls at matched count: first column (a non-universal sink), an interior content column,
   random image tokens, and prompt tokens.
4. Sweep the bias strength and report the whole curve, so no single flattering value can be chosen.
5. Keep every arm at one pass and identical realized tokens — this makes it a pure paired comparison.
