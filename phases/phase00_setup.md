# Phase 0 (0/0b/0c) — Harness, data joins, and the measurement instrument

## 1. Research question
Before any claim: can we measure a VLM's answer and its visual token budget *reliably enough* that a
later contrast means something? Specifically — can we join POPE items to COCO ground truth (area,
bbox, centrality), read a forced-choice answer without generation noise, and count the visual tokens
the model **actually** processed?

## 2. Finding and contribution (plain English)
This phase produced no scientific claim. It produced the instrument every later phase depends on:
a yes/no read-out taken from logits rather than generated text, a join from POPE questions to COCO
boxes, and — most importantly — the habit of reading token counts out of the processor's own
`image_grid_thw` instead of computing them by hand. That last decision is what makes the entire
"matched budget" framing possible later.

## 3. Numbers that changed
No result numbers. Established: POPE 5553 items joined to COCO `val2014`; two model families
(LLaVA-1.5-7B, Qwen3-VL-2B) behind one interface.

## 4. Keep in paper: 2/10
Not a result. One or two sentences in Methods about measured-vs-computed token counts, which is
worth stating because a later bug (#18) came from violating it.

## 5. Experiment, step by step
1. Load POPE (adversarial / popular / random splits) and COCO `val2014` instance annotations.
2. Join each POPE question to its COCO image and the annotation for the queried category; compute
   area fraction, bbox, and centroid distance from image centre.
3. Wrap both models so a question returns `P(yes)` as `softmax` over the `yes`/`no` token ids at the
   final position — no sampling, no generation.
4. Record, per item, the processor's `image_grid_thw`, and derive merged visual tokens from it.
5. Verify the derived count matches what the model actually consumed, on a sample.
