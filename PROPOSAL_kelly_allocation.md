# Proposal: Argmax is the wrong operator for visual token allocation

## The observation nobody has connected
Every method in this space — V*/SEAL, ZoomEye, RAP, ViCrop, CropVLM, Q-CueGraph, RUTA, ViRGo —
performs **argmax allocation**: score regions, pick the best one, spend the budget there. ViRGo
even routes *between* argmax strategies. Nobody asks whether argmax is the right operator.

Our measurements say it is not, and we can derive what is.

## Step 1 — an empirical law (measured, Phase 27, n=191)
Accuracy as a function of tokens landing on the region that contains the target:

    A(b) = 9.21 * ln(b) + 3.68        R^2 = 0.976

over a 53x budget range (150 -> 7957 realized tokens). **Utility is logarithmic in tokens-on-target.**
This is the §4 sweep re-read as a utility curve rather than as a bound.

## Step 2 — the optimal policy follows in one line
Let a localizer produce a posterior `p` over K candidate regions, budget B, allocation `b_i`:

    maximise  E[A] = sum_i p_i * (c*ln(b_i) + k)      s.t.  sum_i b_i = B
    Lagrangian:  p_i / b_i = lambda   =>   **b_i ∝ p_i**

Log utility plus a fixed budget gives **proportional (Kelly) allocation**, not argmax. Argmax is
optimal only in the limit where the posterior is a point mass — i.e. **only with an oracle**, which
is exactly the regime every paper in this area evaluates in.

Define the sharpness family `b_i ∝ p_i^α`:  α=0 uniform, **α=1 proportional (predicted optimal)**,
α→∞ argmax (what the field does).

## Step 3 — a crossover prediction, and measured localizers sit on the wrong side of it
Simulating with our measured endpoints (wrong-region accuracy 38.2%, A(b) as above, K=3):

| localizer hit rate | argmax | proportional |
|---|---|---|
| 30% | 43.6% | **46.1%** |
| 40% | 45.4% | **46.2%** |
| 50% | **47.2%** | 46.6% |
| 70% | **50.8%** | 48.7% |
| 100% (oracle) | 56.2% | 56.2% |

**Crossover at ~45-50% localizer accuracy.** Measured localizers: the VLM's own grounding boxes are
right (IoU>0.5) on **28.9-43.2%** of items. **They are below the crossover.** So the field is using
argmax in precisely the regime where it is the wrong operator.

This also explains, without new assumptions, three things already in the literature:
* why a wrong crop is **worse than not cropping** (our §3: -12.9pp vs tiling);
* why ViRGo's ViCrop arm falls **below baseline** on Qwen3-VL-2B (60.5 vs 64.4) — argmax with a
  weak posterior;
* why hybrid thumbnail+crop schemes and top-2 unions (Q-CueGraph) beat top-1 — they are crude,
  hand-tuned points on the α spectrum, arrived at empirically without the framework.

## The experiments
**E1 — verify the law.** Re-fit A(b) per architecture and per benchmark. Falsified if the log fit
degrades (R^2 < 0.9) or the coefficient varies wildly across models.

**E2 — sweep α at matched realized budget.** Posterior from the model's own attention (Phase 22
showed rank marks the target +20-35pp above chance at L2/L4/L8) and from its grounding head.
α ∈ {0, 0.5, 1, 2, 4, ∞}. **Pre-registered prediction: α* ≈ 1, and α*=∞ (the field's choice) is
strictly dominated.**

**E3 — the calibration law.** Degrade the posterior synthetically (interpolate oracle -> uniform) to
sweep hit rate, and measure α*(hit rate). **Predicted monotone increasing, crossing into argmax's
favour near 50%.** This is the headline figure: *how sharp your spending should be is a function of
how good your localizer is* — a relationship no paper in this area states.

**E4 — risk, not just mean.** Report the rate of catastrophic misses (worse than uniform at the same
budget). Predicted to fall monotonically as α decreases. Argmax is a high-variance bet; the field
reports only means.

## How this dies (name it before running)
* **Format cost.** Proportional allocation needs K regions at K resolutions. Phase 23 showed
  Qwen3-VL handles 5-image input *worse* than 1 image — a format penalty that could swamp the
  allocation gain. **Mitigation:** cap K at 3 (thumbnail + top-2), which is inside the range that
  worked, and report the format penalty as its own ablation.
* **The log fit may not hold per item**, only in aggregate. E1 tests this.
* If α* = ∞ empirically, the framework is wrong and argmax is vindicated — report that.

## Why this is not the crowded part of the field
We are not proposing a better localizer — the localizer is an input. We are saying **the operator
applied to it is wrong**, deriving the correct one from a measured utility curve, and predicting
when it matters. The closest prior work argues "top-K is mis-specified" for **token pruning**
(2608.01665, 2608.09176), never for **spatial allocation at encode time**, and none of it has the
asymmetric-payoff measurement that makes the argument bite.
