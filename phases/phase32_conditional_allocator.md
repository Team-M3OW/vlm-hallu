# Phase 32 — The conditional allocator: earn the second pass by deciding when to spend it

## 1. Research question
If always-cropping fails, can a gate decide per item whether to use the crop — and does that clear
the compute-matched bar?

## 2. Finding and contribution (plain English)
Yes. Answering from whichever pass is more confident, plus a rule that skips reallocation on
relational questions, reaches 68.6% and clears the 66.0% bar.

Equally important: the **learned** gates lost. A confidence threshold scored 59.7% and a linear probe
on attention features 57.1%. Rules beat learned gates here — and much later (§6E) we found out why:
the two-pass confidence comparison is an unsupervised **coverage detector**, while a threshold or a
probe is a worse one.

This phase also fixed a design flaw in Phase 31: it stores **full probability vectors**, not just the
argmax, which is what made every later gate analysis possible.

## 3. Numbers that changed
- Conditional allocator **68.6%**, **+12.0pp** CI **[+6.8, +17.3]**, oracle capture **32.9%** —
  clears the 66.0% bar.
- `conf_thresh` **59.7%**; `learned_LR` **57.1%** — both fail.

## 4. Keep in paper: 8/10
The method's first real positive, and the learned-gate negatives are worth keeping because they
become explainable later rather than staying mysterious.

## 5. Experiment, step by step
1. Store **full probability vectors** for every arm, so confidence gates can be tested offline.
2. Split candidate gates into **UNFITTED** (evaluable on all 191 with no leakage) and **FITTED**
   (5-fold CV).
3. Enforce forbidden features by never reading them: no GT box, no `gt_area_frac`, no V\*Bench
   `category` label.
4. Score against the compute-matched bar, not against uniform@300.
5. Report oracle capture, defined as (method − uniform@300) / (oracle − uniform@300).
