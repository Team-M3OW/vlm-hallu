# The method: Depth-Contrast Re-ranking (DCR)

**+16.2pp over the vanilla VLM on V\*Bench, CI [+10.5,+22.5], at 1.61 forward passes.**
No fine-tuning of the VLM. No extra pixels beyond one crop. No preprocessing. Reads only the
model's own attention.

---

## 1. The problem: VLMs spend visual tokens where they are not needed

A VLM encodes an image into ~300 merged tokens. When the question is about something small, the
evidence occupies a fraction of one token and **nothing is encoded to read**: below ~0.15 merged
tokens of target extent, accuracy is 19.0% — *below the 25% chance level* — while the identical
question asked of the resolved region is answered 96–100% correctly (§13B, two architectures).

The field's response is to crop and look again. Every such method decides *where* to crop from the
model's attention, and **that decision is where they lose**: the standard read-out lands on the
evidence only **39.3%** of the time, while on **88.5%** of items some cell in the very same map
would have worked.

## 2. The finding: the read-out destroys its own signal by averaging

Everyone collapses a block of layers into one map and takes the argmax. That is the defect.

| proposer | top-1 evidence coverage |
|---|---|
| deployed block-16-26 mean | 39.3% |
| best single layer (L17) | 41.9% |
| **learned signed combination** | **52.9%** |
| *(true ceiling: any cell covers)* | *88.5%* |

The best single layer beats the 11-layer mean, so this is not "the wrong layers." The learned
weights take **both signs**: L17/L19/L24/L16 positive, **L27/L26/L15/L11 negative**. The final
layer's gt_pct is **0.529 — worse than the 0.500 chance level**, i.e. anti-correlated with the
target, and the deployed read-out adds it in at weight +1 like everything else.

> **A mean can only add. The signal lives in the disagreement between layers, and averaging is
> precisely the operation that destroys disagreement.**

Averaging more layers keeps making it worse: 3 layers 45.5% → 11 layers 39.3% → all 28 **36.6%**.

## 3. The method

Three steps. Steps 1 and 3 are what the model already does.

1. **Glance.** One forward pass at B₀ = 300 tokens. Keep the per-layer attention from the last
   token to the image tokens — already computed, never stored by anyone.
2. **Re-rank.** Score each cell with a small learned head over its 28-layer attention profile, its
   3×3 neighbourhood, and its position. Take the argmax. *Cost: a tree ensemble over 300 cells —
   microseconds, no GPU, no extra pass.*
3. **Answer** from a W = 0.15 window at that cell, re-fit to the same 300 tokens.

**Gate:** on relational questions ("left of", "above", …) skip steps 2–3 and answer from the glance.
Relational evidence spans multiple regions and one window cannot cover it — the §3 coverage account,
confirmed on two benchmarks.

## 4. Results

**V\*Bench, n=191, Qwen3-VL-2B.** Tokens measured from `image_grid_thw`, every arm within 2% of
target. Proposals produced **out-of-fold**, grouped by item.

| | acc | passes | vs vanilla |
|---|---|---|---|
| vanilla uniform@300 | 56.5% | 1 | — |
| uniform@600 (spend the budget instead) | 63.9% | 2 | +7.4 |
| DCR ungated | 68.6% | 2 | **+12.0** [+4.2,+19.9] |
| **DCR gated** | **72.8%** | **1.61** | **+16.2** [+10.5,+22.5] |

**Gated DCR also beats the compute-matched budget baseline by +8.9pp [+2.1,+16.2] while using fewer
passes.** In its operating regime (single-region questions) it is **+27.0pp [+17.4,+36.5]** over
vanilla and **+13.0pp [+2.6,+23.5]** over `uniform@600`.

**Zero-shot transfer — HR-Bench 4k, n=800, nothing refitted.** The re-ranker improves proposals on a
benchmark it never saw, at 4032px: **+4.9pp [+1.8,+8.1]** per-row and **+6.5pp** on CircularEval
(26.5% → 33.0%), rising to **+8.5pp [+4.0,+13.0]** on single-region questions.

## 5. Why it works, causally

- **The evidence region is load-bearing.** Masking it shifts logits **3.32×** more than masking a
  random region and costs **10.5pp** (§14H).
- **But it cannot be exploited internally.** Seven internal interventions are null — sink
  suppression, attention amplification, residual steering, layer read-out, DoLa/DeCo, component
  ablation, and contrastive decoding *steered to the ground-truth region* (ceiling +2.6pp
  [−0.5,+5.8]). The model looks in the right place; the right place does not contain enough.
- **So allocation must add pixels**, and the only remaining lever is deciding where — which is
  exactly what DCR fixes.
- **The gain concentrates where the deficit is:** +13.8pp below the cliff, +16.0pp in the cliff
  zone, **+3.7pp n.s. above it** (§14D(b)). The method is a fix for sub-token evidence and §13B
  predicts its own operating regime.
- **Localisation is built by the LM, not the image.** Every vision-tower arm is at or below the
  2.3% chance rate (§14G), so the signal DCR reads cannot be obtained before the language model runs.

## 6. Limitations, stated plainly

- **The gate is not validated as transferring.** Its keyword rule agrees with V\*Bench's own
  `category` annotation on **99.0%** of items, so on this benchmark it may be the annotation in
  disguise; on HR-Bench the same rule is at chance concordance. The **ungated +12.0pp** is the
  defensible pooled claim; the gated +16.2pp is conditional on the regime being detectable.
- **Relational questions get worse** (−10.5pp vs vanilla ungated). The gate exists because of this.
- **At 4K the allocator still loses to the budget axis** (−12.4pp vs uniform@600), even though the
  re-ranker itself transfers. Better proposals narrow that gap; they do not close it.
- The head is trained on **V\*Bench GT boxes** (out-of-fold). Cross-benchmark transfer is
  demonstrated; cross-benchmark *training* is not.
- One backbone for the end-task numbers; the read-out defect is verified on one model, with
  cross-architecture replication pending.
- **21.5pp** of headroom remains to oracle placement at the same window — DCR captures 27.6% of the
  proposal ceiling, not most of it.
