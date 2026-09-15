# Phase 10 — Why does the full scene destroy the benefit of a good crop?

## 1. Research question
Holding the crop fixed, what exactly suppresses it: the number of images, the prompt format, the
connector sentence, or the **scene content** itself?

## 2. Finding and contribution (plain English)
The scene content. Image count is exonerated (`[crop+crop]` 38.3% ≈ crop alone 40.4%), prompt format
is exonerated, and the connector sentence actually *helps*. Adding the full scene costs ~25pp.

A high-resolution crop that lets the model answer 40% of the time becomes nearly useless (13%) when
the full scene sits in the same prompt — same pixels, same crop, same model.

A pre-registered hypothesis ("the model re-anchors on its own failed percept") was **refuted**: the
other-scene arm looked better on raw recovery but carried FP 19.1%, and on **discrimination** the CI
spans zero. Any full scene interferes about equally.

## 3. Numbers that changed
| arm | recovery | FP | discrimination |
|---|---|---|---|
| crop only | **40.4%** | 3.5% | +36.9 |
| crop + connector | 48.9% | 4.3% | +44.7 |
| [crop + crop] | 38.3% | 3.5% | +34.8 |
| [other scene + crop] | 29.8% | **19.1%** | +10.6 |
| **[full + crop]** | **12.8%** | 0.0% | **+12.8** |
| [crop + full] (order swapped) | 4.3% | 0.0% | +4.3 |

Recency: full image last is **−8.5pp** worse, CI [−12.8, −4.3].

## 4. Keep in paper: 5/10
Interesting and well-controlled, but it is a **prompt-format** result, and this project's standing
constraint is that such findings stay footnotes. Keep as a methods caution: never evaluate a crop
intervention alongside the full scene without measuring this.

## 5. Experiment, step by step
1. Fix the crop box per item so every arm sees the identical crop.
2. Build arms varying one factor at a time: image count, connector text, ordering, scene content,
   and a black-image control.
3. Score **recovery and false-positive rate** for every arm; define discrimination = recovery − FP.
4. Render verdicts **only on discrimination** — raw recovery is confounded by yes-bias, which is
   exactly what caught the re-anchoring hypothesis.
5. Bootstrap all paired differences.
