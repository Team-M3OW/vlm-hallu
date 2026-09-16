# Track A — prior-art survey (2026-09-17)

Written down because the first survey was done only in conversation and had to be re-run.

## What is anticipated — do not claim

| our claim | taken by |
|---|---|
| Early-layer attention is question-blind; rank late instead | **TwigVLM** (ICCV'25, 2503.14075) — abstract: *"considerable accuracy drop due to insensitive attention signals in early layers."* Also LearnPruner (2604.23950), QCTS, ACL Findings'25 (2502.11501) |
| Early pruning is not semantically selective | **FEATHER** (ICCV'25, 2412.13180) — attributes it to RoPE long-range decay biasing shallow attention toward the image bottom; prunes at K=8 |
| The best pruning layer is architecture- and size-dependent | Reported as practical tuning in several papers: LLaVA-OV-0.5B at L2 vs LLaVA-OV-7B / Qwen2-VL-7B at L4; L8 for OV-0.5B, L11 for InternVL2.5-26B |
| "Token pruning worse than random" | **Wang et al., CVPR 2026** (2512.07580), *When Token Pruning is Worse than Random* — but for **deep** layers (>L20), explained by *vanishing token information* / an "information horizon", prescribing **random** pruning deep |
| Vision-centric benchmarks tolerate flawed early pruning | FEATHER §"Important finding on benchmarks" |

**The draft title — "VLMs Read Their Own Attention at the Wrong Depth" — is a paraphrase of
TwigVLM's abstract sentence. It cannot stand.**

## What is NOT taken

**Read depth and prune depth are two knobs, and every paper above moves them together.**

- FEATHER moves the prune point to K=8 (pays compute).
- Wang et al. move it past L20 and read attention *there* (hence "≈ random").
- TwigVLM keeps the early prune point but replaces the signal with a **trained twig**.

Our phases 75/83 **pin the prune point at FastV's K=2** — `FASTV_K = 2`, identical for every arm,
identical budget, both the ranking pass and the answer pass counted for every arm — and change
**only which of the base model's own layers the ranking is read from**:

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| late read − layer-2 read, 10% keep | **+21.4pp** | **+11.5pp** [+4.2,+18.8] |
| layer-2 read vs random selection | **−3.7pp** | **−4.7pp** |
| late read vs no pruning at all | −0.5pp | −1.6pp [−6.3,+3.1] |

That is a **controlled decomposition nobody has run**: how much of early pruning's loss is the
*prune point* and how much is the *read source*. Answer: the read source, almost all of it, with the
prune point held at 2.

### It also reconciles a live disagreement
Wang et al. find attention-based selection ≈ random when read at deep layers. We find a deep read
beats random by 17.8pp. **Both hold**, because they read deep *and* prune deep, while we read deep
and prune at 2. Their "information horizon" is a statement about tokens *after* deep processing; our
result is about the deep layers' *attention over tokens that were never pruned*. Conflating read and
prune depth is what produces the contradiction.

### And the honest catch, already measured
A late read needs a full prefill, so it is a diagnostic, not a free method. The deployable
training-free version is **cut blind early, rank late**:

| arm | acc | visual-token compute saved |
|---|---|---|
| two-stage, attention-guided 50% early | 51.8% | **64%** |
| two-stage, **random** 50% early | 50.8% | **64%** |
| two-stage, 25% early | 39.3% | 74% |

−4.7pp [−9.9,+0.5] against pure-late, at 64% saved vs pure-late's 39%. The attention-guided and
random early cuts are **the same** (+1.0pp), which is the read-depth claim applied to itself.
This is the training-free counterpart to TwigVLM's trained twig, and it quantifies what the twig buys.

## Also ours, and negative
- **Layer ranking quality does not predict pruning damage.** Pre-registered the opposite: Qwen2-VL
  ranks the target worse at layer 2 (gt_pct 0.620 vs 0.456) so its penalty should be larger. It is
  **smaller** (−13.1 vs −21.9). The verdict replicates on both models; the standard causal story
  for it does not.
- **Selection bias in read-depth sweeps.** In-sample best-of-28 reads 41.9% on Qwen3-VL and collapses
  to **36.1%** out-of-fold — *below* the block mean it appeared to beat. Every published sweep that
  picks a layer on the evaluation set is exposed to this.
- **The remedy's form does not transfer:** Qwen2-VL is fixed by one integer (L21, all 5 folds);
  Qwen3-VL needs a learned combination (no single layer stable OOF).

## Verdict
Track A is not a method paper and not "the wrong depth" paper. What survives is a **controlled
decomposition of read depth vs prune depth**, a **reconciliation of two published results that
appear to contradict**, a **failed mechanism prediction**, and a **selection-bias caution**. That is
an honest empirical/analysis paper. Retitle accordingly.

## Sources
- TwigVLM — https://arxiv.org/abs/2503.14075
- FEATHER — https://arxiv.org/abs/2412.13180
- When Token Pruning is Worse than Random (CVPR'26) — https://arxiv.org/abs/2512.07580
- Are We Solving the Right Problem? (ACL Findings'25) — https://arxiv.org/html/2502.11501v1
- LearnPruner — https://openreview.net/forum?id=Dxb6gBJHby
- Beyond Text-Visual Attention (ICCV'25) — https://arxiv.org/abs/2412.01818
