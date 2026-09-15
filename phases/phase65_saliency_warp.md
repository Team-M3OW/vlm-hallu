# Phase 65 — Warp instead of crop: AttWarp scored at a matched budget

## 1. Research question
Cropping is a step function over the token budget — all tokens inside the window, none outside, and
at 4K that discards 97.75% of the scene. A saliency-guided **warp** magnifies the target and
compresses the periphery, keeping everything. Does it beat the budget axis?

## 2. Finding and contribution (plain English)
**The idea is published.** A prior-art check found **AttWarp** (arXiv 2510.09741, ICLR 2026): same
attention source, same separable inverse-CDF warp, training-free, and it explicitly claims the
fixed-token-budget interpretation. Our crop method is likewise pre-empted by **ViCrop** (2502.17422).

So the run was reframed from "our new method" to **the evaluation AttWarp's own paper does not run**:
scoring it against the budget axis at matched realized tokens. AttWarp's baselines are all
image-manipulation methods, it never asks what the same tokens buy uniformly, and it costs two passes
(1,152 vs 576 vision tokens) without being scored against `uniform@2×`.

**Result: break-even.** The best warp is +0.6pp against its own bar — its reported gains do not
survive being charged for the second pass. It also loses to our multi-crop arm.

## 3. Numbers that changed
| arm | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| **warp@0.7** | 2.00 | 591 | 64.4% | 63.8% | **+0.6** [−6.2,+7.5] |
| warp@0.9 | 2.00 | 591 | 61.8% | 63.8% | −2.0 |
| top1@0.15 | 2.00 | 590 | 60.7% | 63.7% | −3.0 |
| multi4 | 2.00 | 601 | 68.6% | 63.9% | +4.7 |

Below the cliff (single-region): uniform 32.4% · **warp 49.3%** · **multi4 63.4%** · oracle 95.8%.

## 4. Keep in paper: 7/10
Not a method contribution — a **baseline evaluation**, and a useful one: it extends the §2 finding to
a just-published ICLR method. Must carry the caveat that this is a reimplementation of the policy on
our backbone, not the released system.

## 5. Experiment, step by step
1. Marginalise the ring-masked attention map to x and y; mix each with the uniform density by λ; add
   a floor so the periphery cannot collapse; integrate to a CDF and invert to get source coordinates.
2. **Sample directly onto the output grid**, not the source grid — the first version built two
   float64 meshgrids at full resolution (up to 5759×1440), allocated ~1.1 GB per item, and was
   OOM-killed after 6 items.
3. Verify on a synthetic target that λ=0 is the identity and larger λ magnifies while retaining the
   full scene.
4. Sweep λ so a single flattering value cannot be selected.
5. Score against uniform arms measured in the same run, and against our own crop arms.
