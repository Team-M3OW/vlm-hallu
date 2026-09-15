# Phase 60 — Logit lens: is the answer absent, or present but not read out?

## 1. Research question
Three interventions had failed, and we concluded the information is **absent**. But all three
*change behaviour*; none *reads* the representation. The logit lens reads it directly — and could
falsify the conclusion.

## 2. Finding and contribution (plain English)
Two things, one of which turned out to be a bug in our own code.

**What survives:** the answer forms **abruptly at L22**. Both curves sit flat at ~33.8% through L20,
then jump. The oracle advantage is ≈0 before L21 and +38.7pp at L22 — visual evidence becomes
decodable only at answer-formation time, not gradually.

**What did not:** the lens applied the model's final norm to a hidden state HuggingFace had
**already normalised**, corrupting only the last layer. That manufactured an apparent "+5.1pp free
read-out gain" and a whole story about late-layer bias, all retracted in Phase 64. Intermediate
layers were correct throughout, which is exactly why the bug hid so well.

## 3. Numbers that changed
Sub-token stratum, corrected (final layer from the model's own logits):

| layer | uniform | oracle | oracle − uniform |
|---|---|---|---|
| L0–L20 | ~33.8% flat | ~33.8% flat | ≈0 |
| **L22** | 47.1% | **88.2%** | **+41.2** |
| L24 | 52.9% | 92.6% | +39.7 |
| final | 52.9% | 92.6% | +39.7 |

**No read-out gap:** the final layer equals the best intermediate layer exactly (+0.0pp), even with
the best layer chosen in-sample.

## 4. Keep in paper: 7/10
The L22 formation curve is a good figure and is unaffected by the bug. The corrected "no read-out
gap" result **strengthens** the causal argument, and the bug itself belongs in the methods discipline
section.

## 5. Experiment, step by step
1. Run each item twice at the same budget (uniform, oracle crop), capturing the last token's residual
   at **every** layer.
2. Project each through the model's final norm and unembedding; restrict to the A/B/C/D token ids.
3. **Take the final layer from `model.logits`, never from `norm(hidden_states[-1])`** — HF has
   already normalised that one. (This is the fix; the original run did not.)
4. Compare the uniform and oracle curves layer by layer.
5. Test whether any intermediate layer beats the final one, with the layer chosen **out of fold**.
