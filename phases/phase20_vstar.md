# Phase 20 — The allocation result where sub-token targets are the DESIGN, not a tail

## 1. Research question
POPE's phenomenon is a 1.6% tail and CUB failed to enlarge it. On V\*Bench, sub-token targets are the
benchmark's design. Does budget-matched query allocation hold there, on ~100% of items?

## 2. Finding and contribution (plain English)
Yes, and larger. Query-placed allocation at 292 tokens beats uniform allocation at 1176 tokens.
On items uniform allocation gets *wrong*, query allocation recovers 86–93% while random placement
recovers 16–19%. That contrast is the thesis in one line: it is not *that* you allocate
non-uniformly, it is **where**.

One practically important surprise: at adequate budget, allocation **matches crop-only** — so you do
not have to discard the scene to get the benefit.

## 3. Numbers that changed
| budget | uniform | **alloc_query** | alloc_random | crop_only |
|---|---|---|---|---|
| B=300 | 56.5% | **85.9%** | 48.7% | 93.7% |
| B=600 | 66.0% | **90.1%** | 60.2% | 93.7% |
| B=1200 | 70.2% | **94.2%** | 63.9% | — |

Decider: **+29.8pp** [+22.5,+37.2] at B=600; **+30.4pp** [+23.0,+37.7] at B=1200.
Headline: **292 tokens (85.9%) > 1176 tokens (70.2%)**, +15.7pp CI [+7.9,+23.6].

## 4. Keep in paper: 9/10
The main-benchmark version of the centrepiece, on the full 191 items with no confidence selection.
Positioning note that must travel with it: V\*Bench was built by the V\*/SEAL authors, so "cropping
helps here" is *their* result — ours differs by the **matched realized budget** and the
**size-matched random-placement control**.

## 5. Experiment, step by step
1. Fix the read-out **before coding** (the CUB lesson): 4-way MCQ → softmax over {A,B,C,D} logits at
   the final position; metric = accuracy; all 191 items, no confidence selection.
2. Verify realized-token match at every budget (spread 1.7–2.7%).
3. Run `uniform`, `alloc_query` (GT region), `alloc_random` (size-matched), `crop_only`.
4. Repeat at B = 300 / 600 / 1200.
5. Note why no FP arm is needed: 4-way argmax is immune to the bias-shift trap, since uniformly
   inflating one option cannot raise argmax accuracy.
