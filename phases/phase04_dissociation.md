# Phase 4 — The dissociation: does the model localise what it verbally denies?

## 1. Research question
Take items where the model **confidently** says an object is absent, but it is present. Can the same
model, in the same forward-pass budget, draw a correct box around that object?

## 2. Finding and contribution (plain English)
Yes, often. On the exact items where its verbal judgment is confidently wrong, the model still puts
a correct box on the object **43.2%** of the time. Localisation and verbalised judgment come apart
inside one model on one input.

This is the observation the entire project is built on, and §13 (much later) finally explains it.

## 3. Numbers that changed
| group | n | emits a box | mean best IoU | IoU > 0.5 |
|---|---|---|---|---|
| confident denial (object present) | 243 | **95.1%** | 0.368 | **43.2%** |
| negative control (object absent) | 243 | 45.3% | 0.000 | 0.000 |

## 4. Keep in paper: 8/10
This is the motivating figure. Keep it — but the honest framing now comes from §13: localisation is
**empty**, not merely dissociated. Present Phase 4 as the observation and §13 as its resolution.

## 5. Experiment, step by step
1. Select items where Qwen3-VL gives `P(yes) < 0.01` for an object that IS present.
2. Build a matched control of items where the object is genuinely absent and also confidently denied.
3. Ask the same model to output a bounding box for the queried object.
4. Score box emission rate and IoU against the COCO box.
5. **Caveat kept in the file:** IoU is circular for the absent arm (no GT box exists, so IoU ≡ 0).
   The non-circular signal is **box emission**, 95.1% vs 45.3%.
