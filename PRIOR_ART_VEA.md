# Predecessor: "Seeing but Not Believing" (VEA) — arXiv 2510.17771, ICLR 2026 poster

**Framing decision (2026-09-17).** VEA is treated as this work's **predecessor**, not as a scoop.
We reached the same conclusions independently, on different model families, with different metrics,
on a different benchmark — which is corroboration of a shared phenomenon, and corroboration is worth
more than either result alone. The paper cites them as establishing it and claims only what goes
beyond: **the regime boundary, the control that locates it, and a method that works on the side of
the boundary where theirs cannot.**

Independent convergence worth stating in the paper: they identify L16-26 by AUROC against boxes on
document VQA; we identified the same band by question-swap divergence on V*Bench small-object search.
Two metrics, two benchmarks, two model families, same layers.

Liu et al. (UIUC · Amazon · Penn State), Oct 2025. Read in full 2026-09-17.

## Where we converged independently

| their result | our phase | verdict |
|---|---|---|
| Shallow layers attend to text, deep layers to localized evidence (RAPT, Fig 1) | 95 (question-divergence), 14L | **converged** — different metric, same claim. Cite them; report ours as corroboration |
| "deeper layers (e.g., **layers 16–26**) display sparse yet highly concentrated attention" §2.2 | our deployed block is **L16–26** | **converged on the same band** by a different route. Worth one sentence as independent confirmation |
| Deep layers attend to the correct evidence **even when the answer is wrong** — "seeing but not believing", 4 families | **4** (the founding dissociation, 43.2%) | **converged.** Theirs is the citable statement of the phenomenon; ours is the same observation on a different task |
| Layer selection by **AUROC against GT boxes** on a ~100-item diagnostic set (§3.1) | 45 (`peak` AUROC 0.788), 44 (layer sweep) | **converged** |
| Adaptive layer choice beats a fixed late block — Table 2, VEA 83.6/84.4/85.2/79.1/80.0/81.2 AUROC vs static L50–100% 78.0/76.9/79.5/67.6/65.9/68.1, on 6 configs / 4 families | §14F, 73, 82 (3 of 4 models) | **converged, and theirs is broader** — cite theirs for the general claim, keep ours for the Qwen-family detail |
| 3×3 neighbourhood denoising of the evidence map (eq. 2) | the head's 3×3 neighbourhood feature | same intuition, independently |

**VEA:** profile layers by AUROC once per model → average their attention into a patch evidence map →
3×3 outlier denoise (λ=10) → Gaussian smooth (σ=0.5) → **darken non-evidence pixels** in the original
image, `Î = (α + (1−α)ê)·I` with α=0.5 → re-answer. Training-free. **+5.67 EM** average (up to +11.1),
**+6.83 Token-F1**, 8 VLMs × 4 families, InfoVQA / DocVQA / SROIE / TextVQA.

## What the predecessor leaves open — and what we contribute

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

## What we do not claim

The dissociation, the text→visual depth transition, the L16-26 grounding band, and AUROC layer
profiling are **cited to VEA**, with our independent measurements reported as corroboration rather
than as discovery. Our contribution is what follows from them: the **encoding cliff**, the **oracle
control that adjudicates the predecessor's own thesis**, the **seven null interventions**,
**coverage as the sign-deciding mediator**, and a **method that adds resolution** where theirs
redistributes emphasis.

Stated as one sentence for the paper: *VEA showed VLMs often attend to evidence they fail to use;
we show that holds above a measurable encoding threshold and inverts below it, where the evidence is
not in the tokens at all and only adding resolution recovers it.*

---

# Companion: "Direct Visual Grounding by Directing Attention of Visual Tokens" (KLAL)

Esmaeilkhani & Latecki (Temple), arXiv **2511.12738** v2. Read in full 2026-09-17.

**What it is.** A **training-time** method. A KL attention loss (KLAL) is added to next-token
prediction during fine-tuning, pulling the answer token's attention over visual tokens toward a GT
map built automatically from task geometry or existing box/point annotations. No new labels, no
architectural change, no extra head.

| task | base | NTP only | **NTP + KLAL** |
|---|---|---|---|
| Grid Patch (Qwen2.5-VL-7B) | 6.12% | 28.57% | **44.90%** |
| PixMo-Points (Qwen2.5-VL-7B) | 16.79% | 26.28% | **35.77%** |
| Line Intersection | 47.62% | 62.64% | **70.23%** |
| RefCOCO testB | 86.70% | 86.90% | **87.50%** |

**Why it does not contradict our nulls.** Every one of our seven failed interventions is
**inference-time on a frozen model**. KLAL changes the weights. And their own Table 5 shows it does
not merely redirect attention — it **raises the embedding norm of target visual tokens by 6% (Qwen)
to 19% (LLaVA)**. They had to change what is *in* the tokens too, which is our claim stated from the
training side.

> Our claim, sharpened by this paper: *no inference-time operation on a frozen model recovers
> sub-token evidence — you must either add pixels (ours) or retrain the representation (theirs).*

**What it gives us.** A citable statement that "the standard NTP loss provides an insufficient signal
for directing attention to visual tokens", independent support that attention-to-evidence is causally
load-bearing (their Fig. 4: only after KLAL does the average target token outweigh the average visual
token), and a training-side counterpart that makes our inference-side scope explicit rather than
convenient.

**Where a reviewer could push.** Grid Patch targets are ~1/576 of the image — close to our cliff —
and KLAL lifts Qwen from 6.12% to 44.90% there. The honest answer is that it does so by fine-tuning
on that distribution, with GT attention maps, and that it has never been run on natural-image
sub-token search (V\*Bench). Their tasks are synthetic geometry, grid patches, point annotations and
RefCOCO, where RefCOCO objects are large.

---

# Third in the family: "Reallocating Attention Across Layers to Reduce Multimodal Hallucination"

Lu et al. (BUPT · NTU), arXiv **2510.10285v2**, Feb 2026. Training-free head-level plugin.

**Method.** Compute each head's **modality attention ratio** — the fraction of its attention mass on
visual vs textual tokens — then use depth-aware boundaries to label shallow high-visual heads as
**perception heads** and deep high-textual heads as **reasoning heads**, and apply multiplicative
gains g≥1 to each group. **+4.2pp** average over 5 benchmarks on 3 MLRMs, <1% extra compute.

**Why it matters to us.**

1. **It completes a pattern.** VEA highlights evidence pixels; KLAL supervises attention during
   training; this rescales attention heads at inference. Three independent 2025–26 papers, three
   interventions on emphasis, all reporting gains — **and not one of them measures target size or
   runs an oracle-crop control.** Our cliff predicts the entire family must fail below a threshold.
   That reframes our contribution from *competing with one paper* to *a boundary condition on a
   family of methods*.
2. **It contradicts VEA on the direction of the depth transition.** This paper: *"early layers
   emphasize visual tokens, whereas deeper layers progressively shift focus toward textual tokens"*.
   VEA: shallow layers are text-focused and deeper layers increase attention to images. The metrics
   differ (share of total mass vs relative attention per token), which may reconcile them — but on
   the surface two published papers state opposite things about the same phenomenon, and **nobody has
   reconciled them.** We have the per-layer data on two models to do it.
3. **It hands us a free untried lever.** Their modality attention ratio is a per-head statistic we do
   not use: every locator in this project **averages over heads**. Selecting or weighting heads by
   visual ratio — then applying §14V's max rule across layers — is "reallocate across heads *and*
   layers" and has never been run. It needs one pass storing per-head maps.
