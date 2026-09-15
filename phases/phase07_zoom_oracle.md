# Phase 7 — Is the evidence present in the pixels but under-resolved?

## 1. Research question
Give the model a second image: a crop of the **same photograph** around the object, which the
processor upscales so the region gets many more patches. Zero new information. Does the answer change?

## 2. Finding and contribution (plain English)
Yes. Same pixels, more patches, and 12.8% of confident denials flip to correct — while a
size-matched **random** crop of the same image flips none. This locates the failure upstream of the
decoder: the evidence is in the original pixels but under-resolved at the patch grid the model
actually processes.

Two things had to be said honestly: this is an **oracle** demonstration (the crop comes from GT), and
the clean comparison is oracle-vs-random, not oracle-vs-baseline.

## 3. Numbers that changed
| condition | median P(yes) | flips to "yes" |
|---|---|---|
| baseline | 0.000185 | 0% |
| **oracle zoom** | 0.0078 | **12.8%** (18/141) |
| random zoom (size-matched) | 0.00047 | 0% |

oracle > random on **105/141**, p=2.5e-9; flip-rate difference CI **[+0.078, +0.184]**.
Dose–response: flipped items are smaller (median area 0.00145 vs 0.00401).

**Later amended:** the 12.8% **understates the effect ~3×**. Handing the crop *alone* recovers
**40.4%** — Phase 7's two-image format was itself suppressing the benefit (Phase 10).

## 4. Keep in paper: 7/10
The mechanism claim is right and the amendment makes it stronger. But present the corrected number
(40.4%) and cite Phase 7 as where the effect was first isolated.

## 5. Experiment, step by step
1. For each confident denial, crop the COCO box + 25% padding from the **same** photograph.
2. Present `[full image, connector sentence, crop]` and re-ask the identical question.
3. Build the decisive control: a **size-matched random region** of the same image, same two-image
   format, same connector.
4. Score flips to "yes" and paired differences; bootstrap.
5. Check dose–response: flipped items should be smaller than unflipped ones.
