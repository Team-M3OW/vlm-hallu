# Phase 49 — Cheap candidate re-ranking (ABANDONED mid-run)

## 1. Research question
The GT cell ranks in the top 3.4% while the argmax is usually a distractor. Can we score k candidates
cheaply — at B₀/k tokens each — and pick the best, without paying k full passes?

## 2. Finding and contribution (plain English)
**Abandoned before completion**, at the user's direction, because it is a prompt/pipeline-level
method rather than an internal one. The design is recorded because its core insight was correct and
resurfaced in a better form: Phase 58 keeps the "use more than one candidate" idea but removes the
selection step entirely by handing the model all k candidates in one pass.

Worth recording as an idea that was right and implemented wrong.

## 3. Numbers that changed
None — run stopped at ~40/191 items. Offline precursor established the headroom it was chasing:
oracle selection among the top-5 candidates reaches **61.5%** coverage vs the argmax's **40.5%**.

## 4. Keep in paper: 1/10
Do not report. At most one clause noting that candidate **selection** was tried and superseded by
candidate **presentation**.

## 5. Experiment, step by step
1. Pass 1: uniform@B₀ with attention → top-k spatially separated candidates.
2. Pass 2: crop each candidate at B₀/k tokens; pick the most confident.
3. Pass 3: re-run the winner at full B₀ and answer.
4. Controls designed but not run: `select_random` (same cost, random winner) and `select_oracle`
   (winner by true coverage) — which would have measured whether low-resolution scoring preserves the
   confidence signal at all.
