# Phase 27 — How many uniform tokens would it take to match a well-placed crop?

## 1. Research question
Put a number on it: sweep the uniform budget from 150 to 8000 tokens. At what point does uniform
allocation catch a query-placed crop at 300 tokens?

## 2. Finding and contribution (plain English)
**It never does.** No uniform rung reaches the crop arm, all the way to the model's ceiling. So the
exchange rate is a **lower bound set by the architecture**, not an estimate: ≥26×.

This is the number the paper's §2 is built on, and this phase also supplies the measured uniform
sweep that later becomes the "budget axis" bar for scoring every prior-art method.

## 3. Numbers that changed
Qwen3-VL, V\*Bench, measured sweep:
`uniform@150` 46.6% · `@294` 56.5% · `@600` 66.0% · `@1176` 70.2% · `@2400` 75.4% · `@4760` 81.2% ·
`@7957` 84.8%.
`crop_only@300` **93.7%** · `crop_random@300` 38.2% · `alloc_query_2img@292` **85.9%**.
⇒ exchange rate **≥26×**.

## 4. Keep in paper: 9/10
Central. The "no rung catches it" framing is stronger than any point estimate, and the sweep itself
is reused as the compute-matched bar throughout.

## 5. Experiment, step by step
1. Sweep uniform budgets across the model's full dynamic range, measuring realized tokens each time.
2. Run the query-placed crop at the smallest budget (300).
3. Find the lowest uniform rung that matches or beats it.
4. If none does, report the ratio to the **top** rung as a lower bound, and say explicitly that it is
   bounded by the ceiling rather than estimated.
5. Keep `crop_random` alongside so placement is separable from cropping.
