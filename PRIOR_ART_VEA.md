# Prior art: "Seeing but Not Believing" (VEA) — arXiv 2510.17771, ICLR 2026 poster

Liu et al. (UIUC · Amazon · Penn State), Oct 2025. Read in full 2026-09-17.

## What they have that we also have

| their result | our phase | verdict |
|---|---|---|
| Shallow layers attend to text, deep layers to localized evidence (RAPT, Fig 1) | 95 (question-divergence), 14L | **anticipated** — different metric, same qualitative claim |
| "deeper layers (e.g., **layers 16–26**) display sparse yet highly concentrated attention" §2.2 | our deployed block is **L16–26** | **anticipated**, and it is literally the same band |
| Deep layers attend to the correct evidence **even when the answer is wrong** — "seeing but not believing", 4 families | **4** (the founding dissociation, 43.2%) | **anticipated** — this is our project's opening observation |
| Layer selection by **AUROC against GT boxes** on a ~100-item diagnostic set (§3.1) | 45 (`peak` AUROC 0.788), 44 (layer sweep) | **anticipated** |
| Adaptive layer choice beats a fixed late block — Table 2, VEA 83.6/84.4/85.2/79.1/80.0/81.2 AUROC vs static L50–100% 78.0/76.9/79.5/67.6/65.9/68.1, on 6 configs / 4 families | §14F, 73, 82 (3 of 4 models) | **anticipated, and on more models than ours** |
| 3×3 neighbourhood denoising of the evidence map (eq. 2) | the head's 3×3 neighbourhood feature | same intuition, independently |

**VEA:** profile layers by AUROC once per model → average their attention into a patch evidence map →
3×3 outlier denoise (λ=10) → Gaussian smooth (σ=0.5) → **darken non-evidence pixels** in the original
image, `Î = (α + (1−α)ê)·I` with α=0.5 → re-answer. Training-free. **+5.67 EM** average (up to +11.1),
**+6.83 Token-F1**, 8 VLMs × 4 families, InfoVQA / DocVQA / SROIE / TextVQA.

## What they do NOT have

1. **No size analysis whatsoever.** No tokens-on-target, no cliff, no sub-token regime. Their four
   datasets are document and scene-text VQA — the evidence region is a readable text snippet, orders
   of magnitude larger than a V\*Bench target (median **0.4 merged tokens**).
2. **No oracle control.** They never test whether a perfect crop of the evidence fixes the item, so
   they cannot separate *"perceived but not used"* from *"attention ranks the right cell and the cell
   contains nothing."* Their central claim depends entirely on that distinction.
3. **Their intervention adds no resolution.** It darkens non-evidence pixels. That is the family our
   phase 13 measured at 5.6% (red box) / 0.0% (coordinates) and phase 51 measured at **16%** of what
   cropping buys, with the reason in one number: the oracle amplification set is **1 cell of 294**.
4. **No compute-matched baseline.** Baselines are other augmentation methods (Instructioning, CGR,
   VAR, AGLA). They never ask what the same tokens buy uniformly, and VEA costs a profiling pass plus
   a second inference. This is exactly the phase 47/53/65 critique and it applies to them cleanly.

## The scientific conflict, stated plainly

> **Theirs:** "VLMs encode reliable evidence internally but under-utilize it."
> **Ours (phase 66):** on sub-token targets the correct answer is decodable at **no layer**, and the
> oracle crop fixes **94.4%** of exactly those items.

Both can hold — **in different size regimes**. That is now our position, and it is stronger than what
we had before reading this: their claim acquires a boundary, we own the control that locates it, and
phase 51's 16% predicts in advance that a highlighting method must fail below the cliff.

## The experiment this creates

VEA is fully specified (eqs 1–4, α=σ=0.5, λ=10, top-10% layers by AUROC) and training-free, so it can
be **reimplemented and run on V\*Bench at matched budget**.

    PREDICTION (write before the run): VEA gains on items above the encoding cliff and is null or
    negative below it, while our crop is largest below it (+13.8pp under 0.15 tokens, §14D(b)).
    A crossing interaction is the result; VEA winning below the cliff refutes our mechanism.

That single run would give us: a published ICLR-2026 baseline scored at matched compute, a boundary
condition on their claim, and an independent test of our own.

## What this costs us

The dissociation, the text→visual depth transition, the L16–26 grounding band, and AUROC layer
profiling are all **no longer ours to claim**. What remains ours: the **cliff**, the **oracle
control**, the **seven null interventions**, **coverage as the sign-deciding mediator**, and the
**crop method** — they do not crop and do not add resolution anywhere.
