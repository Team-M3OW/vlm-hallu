# Phase 16 — Activation steering at L16

## 1. Research question
Phase 15 put a faint category signal at L16. Push the representation along that direction: does the
answer change?

## 2. Finding and contribution (plain English)
No — a clean, well-controlled null. The steering direction was derived on items **disjoint** from the
evaluation set, so it is not leaking. This is the first of what became a family of failed internal
interventions, and in hindsight §13 explains all of them at once: there is nothing inside to steer
toward when the target is sub-token.

A trap this phase exposed: a `rand_last` control arm raised `P(yes)` by shifting bias rather than
improving discrimination. After this, every arm in the project reports a false-positive rate.

## 3. Numbers that changed
Null effect on discrimination. Direction v = mean(queried-category tokens) − mean(size-matched
other-category tokens) at L16, injected as `α·‖h‖·v̂`.

## 4. Keep in paper: 5/10
Keep as one row in the eliminations table. Its real value is retrospective: it is the first data
point in the §10/§13 argument that internal interventions cannot work.

## 5. Experiment, step by step
1. Derive the steering direction at L16 on a **disjoint** item set.
2. Inject via a forward hook on `layers[15]`, scaled to the local hidden-state norm.
3. Sweep α.
4. Include a random-direction control **and** a bias-shift control.
5. Score on discrimination (recovery − FP), never raw recovery.
