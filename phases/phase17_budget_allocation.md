# Phase 17 — At a FIXED token budget, does WHERE you spend beat spending uniformly?

## 1. Research question
"More tokens helps" is not a finding — AnyRes and dynamic resolution already assert it. The real
question: at a **matched realized token budget**, does allocating by query beat allocating uniformly?

## 2. Finding and contribution (plain English)
Yes, decisively, and this is where the project's actual thesis is born. Query-placed allocation at
150 tokens beats uniform allocation at 589 tokens. Roughly 4× fewer visual tokens for ~3× the
recovery.

The decider is not "non-uniform layouts help" — a **size-matched random region** at the identical
budget was pre-registered as the control, and allocation beats it at every budget.

This converts the project's earlier nulls (pointing 0%, steering, attention patching) from failures
into a coherent story: **you cannot redirect attention to capacity that was never allocated, but you
can allocate it differently at no extra cost.**

## 3. Numbers that changed
| budget | uniform | **alloc_query** | alloc_random | crop_only |
|---|---|---|---|---|
| B=150 | 5.6% | **50.0%** | 18.5% | 77.8% |
| B=300 | 11.1% | **38.9%** | 14.8% | 83.3% |
| B=600 | 16.7% | **50.0%** | 14.8% | 66.7% |

alloc_query − alloc_random: **+31.5pp** [+18.5,+44.4] · **+24.1pp** [+13.0,+35.2] ·
**+35.2pp** [+22.2,+48.1]. FP 0.0–4.4% everywhere (no bias shift).

## 4. Keep in paper: 9/10
This is the origin of the paper's central contrast. Caveats that must travel with it: the region is
**oracle** (COCO GT), n=54, and `crop_only` discards the scene so it is not a viable allocator.

## 5. Experiment, step by step
1. Fix a total realized token budget B and verify every arm hits it (spread 0.7–5.6%).
2. `uniform`: whole image at B.
3. `alloc_query`: split B between the scene and the GT object region.
4. **`alloc_random`**: identical split, but a **size-matched random region** — the pre-registered
   decider that separates "allocation" from "non-uniform layouts".
5. `crop_only`: the whole budget on the object region (an upper bound that discards the scene).
6. Repeat at B = 150 / 300 / 600 and bootstrap all paired differences.
7. Report FP on negatives for every arm.
