# Phase 53 — Prior art at matched budget, 4K, n=800

## 1. Research question
Phase 47's table has ~±7pp CIs at n=191 and V\*Bench is not where tree search is supposed to shine.
Rerun at 4× the power on 4032² images with CircularEval.

## 2. Finding and contribution (plain English)
The negative **replicates with significance** — and our own headline **does not**.

Tree search loses decisively in the venue it was designed for: **−15.3pp** against its own bar, and
**−10.4pp against uniform@1200 while spending 2.2× the tokens**. Per token it is the worst arm in the
table.

But Phase 47's "ours is the only positive margin" **does not hold at 4K**: we are exactly break-even
(−0.2pp), grounding beats us by 5.2pp, and our fixed policy is as bad as random placement because a
W=0.15 crop of a 4032² image discards 97.75% of the scene.

Grounding flips sign across scales (−6.2 → +1.3) exactly as the coverage account predicts — it must
localise from a 300-token view, which works at 4032px and fails at 1500px.

## 3. Numbers that changed
| policy | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| grounding | 2.00 | 590 | 61.3% | 59.9% | +1.3 n.s. |
| **ours, adaptive** | **1.40** | **411** | 56.0% | 56.2% | **−0.2** n.s. |
| **zoom_eye** | **9.00** | **2636** | 54.1% | 69.4% | **−15.3** [−18.6,−11.8] SIG |
| ours, fixed | 2.00 | 587 | 42.6% | 59.9% | −17.3 SIG |

CircularEval: uniform@1200 **52.0%** · grounding 47.0% · ours 45.5% · zoom_eye 42.0%.

## 4. Keep in paper: 9/10
The significant version of the paper's headline negative. The retraction of our own cross-benchmark
claim must be kept alongside it.

## 5. Experiment, step by step
1. Key instances on the image **content hash** and assert 4 rows each.
2. Charge per-instance proposal cost to **each** of its 4 rows — the deployment view, applied
   identically to every search-based method. Report the amortised variant too.
3. Build the bar from uniform arms measured in-run on these items.
4. Use the **transferred** firing rate for the adaptive arm, never refit here.
5. Report per-row accuracy **and** CircularEval, the benchmark's own metric.
