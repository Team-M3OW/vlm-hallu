# Phase 11 — Is the benchmark itself wrong? (RePOPE audit)

## 1. Research question
Our whole cohort is "the model confidently denies an object that is present". How many of those are
actually **mislabelled benchmark items** rather than model failures?

## 2. Finding and contribution (plain English)
Some are — and confident model disagreement is a **7.2× enriched** detector of label error. But
after removing every mislabelled item (the "RePOPE-clean" pool), the findings get **stronger**, not
weaker. So the phenomenon is not label noise.

This is the phase that makes every later number trustworthy. Every subsequent cohort is RePOPE-clean.

## 3. Numbers that changed
- Confident disagreement enriches label errors **7.2×** over base rate.
- On correctly-labelled items only, the core effects are **larger**.
- The clean confident-denial cohort shrinks to **n=54**, which becomes a standing power problem.

## 4. Keep in paper: 7/10
Essential methodological hygiene and a reviewer will ask. One paragraph, plus the 7.2× number which
is independently interesting.

## 5. Experiment, step by step
1. Take every POPE item where the model confidently disagrees with the label.
2. Re-adjudicate against COCO annotations and the image.
3. Measure enrichment of label errors in that cohort vs the base rate.
4. Rebuild the analysis pool excluding mislabelled items ("RePOPE-clean").
5. Re-run the headline contrasts on the clean pool and confirm they strengthen.
