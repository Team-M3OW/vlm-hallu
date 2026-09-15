# Phase 64 — Component ablation, and the bug that invalidated three sections

## 1. Research question
Truncation is blunt — it discards all of L25–27 including whatever helps. Which **component**
(attention or MLP, per layer) does the damage?

## 2. Finding and contribution (plain English)
Two results, and the second one matters far more than the first.

**The ablation itself is null.** Zeroing attention or MLP in L25/26/27, singly or together, moves
accuracy by ≤2.4pp and never significantly. The damage is not localised to a component.

**And it exposed the bug.** Because this phase computed the baseline a *different way* — from the
model's own logits rather than the lens — its `trunc_L24` arm came out **+0.0pp** where Phase 60 had
+5.1pp. The two implementations agreed on L24 for **100%** of items and on the final layer for only
**82.2%**. Tracing that disagreement found that HuggingFace returns `hidden_states[-1]` **already
normalised**, so our lens had normalised it twice, corrupting only the final layer.

The internal consistency check that made it findable: `both@late` (zeroing all L25–27 updates) is
mathematically identical to reading at L24, and the two agreed with each other while disagreeing with
Phase 60.

## 3. Numbers that changed
- Logit error: `lm(h_last)` vs true logits **0.06**; `lm(norm(h_last))` vs true logits **23.47**.
- Cross-implementation agreement: **100%** on L24, **82.2%** on the final layer.
- Retracted as a result of this phase: §12A (+5.1pp read-out gain), §12B (late-layer evidence-
  dependent bias), §12C (our rule beating DoLa).
- Corrected: the final layer **equals** the best intermediate layer, +0.0pp.

## 4. Keep in paper: 8/10
The ablation null is one line. The bug and how it was caught is worth a full paragraph in the methods
discipline section — it is the clearest example in the project of why a derived quantity computed by
only one code path is not verified.

## 5. Experiment, step by step
1. Register forward hooks on every layer's `self_attn` and `mlp`, returning zeros for selected layers
   to remove exactly that residual update.
2. Ablate each component in L25/26/27 singly, then all together.
3. Run on **both** the uniform image and the oracle crop, so evidence-dependence can be read directly.
4. Include `trunc_L24` computed from the model's own logits — this is the arm that exposed the bug.
5. Use `both@late ≡ trunc_L24` as an internal consistency check.
