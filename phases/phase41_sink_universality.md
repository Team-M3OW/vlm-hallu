# Phase 41 — Is the row-boundary sink universal across architectures?

## 1. Research question
Phase 34's serialisation account is a claim about how VLMs flatten images into sequences — common to
all of them. Does it generalise, or is it a Qwen3-VL quirk?

## 2. Finding and contribution (plain English)
It generalises, and the strongest evidence is a prediction only this account could make.

Models with an **implicit** row boundary (Qwen2/3-VL) show the last column enriched 3.4–4.5×. Models
that splice in a **learned `image_newline`** at each row end (LLaVA-OneVision, LLaVA-NeXT) put the
sink **on the separator itself**, 2.0–2.3×.

A newline embedding is not a corner, not a pixel, and not image content. A 2-D "corner" account
predicts nothing special there; the serialisation account predicts exactly it.

## 3. Numbers that changed
| model | boundary position | enrichment |
|---|---|---|
| Qwen3-VL-2B | last column | **3.4×** [3.0,3.8] |
| Qwen2-VL-7B | last column | **4.5×** [4.1,4.8] |
| LLaVA-OneVision-7B | **`image_newline`** | **2.0×** [1.8,2.2] |
| LLaVA-NeXT-7B | **`image_newline`** | **2.3×** [1.9,2.9] |

Onset: corner/column **mass is already 39.9% at L0** — present in the first layer, not built up.

## 4. Keep in paper: 9/10
The best mechanistic-interpretability result in the project: four architectures, two families, two
tokenisation schemes, and a discriminating prediction. Publishable on its own.

## 5. Experiment, step by step
1. Use a **relative** layer block [0.55L, 0.95L] so the measurement is architecture-agnostic.
2. Report **enrichment** (mass share ÷ area share), because raw mass is not comparable across models
   with different grids.
3. For newline models, locate separator positions by **matching merged input embeddings against the
   model's own `image_newline` parameter** — no unpad logic reimplemented, nothing depending on our
   arithmetic.
4. Load every checkpoint in **bfloat16** and assert `torch.isfinite` on the attention — an fp16 load
   of a bf16 checkpoint silently produced all-NaN logits in a sibling phase.
5. Keep per-image values so the enrichment can carry bootstrap CIs.
