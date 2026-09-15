# Phase 23 — Does native AnyRes tiling already do what our allocator does?

## 1. Research question
LLaVA-style AnyRes already spends more tokens on high-resolution inputs. At a matched realized
budget, is query-placed allocation still better, or is AnyRes enough?

## 2. Finding and contribution (plain English)
The headline looked good and **the confound was worse**. This phase is mainly a cautionary record:
the first version compared arms whose realized token counts differed, so the "win" partly measured
extra tokens. What survives cleanly is narrower than first claimed.

## 3. Numbers that changed
Headline withdrawn pending the budget gate; the clean statement moved to Phase 25.

## 4. Keep in paper: 2/10
Do not cite the headline. Its value is as the origin of the budget-gate discipline that every later
phase inherits.

## 5. Experiment, step by step
1. Run V\*Bench under native AnyRes tiling and under query-placed allocation.
2. **Measure** realized tokens for both (this is where the confound was found).
3. Recognise that the arms were not budget-matched; void the contrast.
4. Institute the ≥10% budget-gate rule for all later phases.
