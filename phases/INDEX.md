# Phase index — VLM visual token allocation project

One file per phase. Each contains: research question · finding in plain English · numbers that
changed · keep-in-paper rating · stepwise method.

## Highest value (8–10/10)
| phase | what it established | rating |
|---|---|---|
| [66](phase66_localize_vs_answer.md) | **Localisation is empty; the encoding cliff at 0.15–0.25 tokens, architecture-invariant** | **10** |
| [27](phase27_exchange_rate.md) | Exchange rate ≥26×, a lower bound set by the architecture | 9 |
| [20](phase20_vstar.md) | Allocation on V\*Bench where sub-token targets are the design | 9 |
| [17](phase17_budget_allocation.md) | The centrepiece: matched-budget query allocation | 9 |
| [41](phase41_sink_universality.md) | Row-boundary sink across four architectures; the `image_newline` prediction | 9 |
| [47](phase47_prior_art.md) | Prior art at matched budget — nobody beats the budget axis | 9 |
| [51](phase51_attention_allocation.md) | Allocation must add pixels: attention recovers only 16% | 9 |
| [53](phase53_prior_art_hrbench.md) | The negative replicates with significance at 4K, n=800 | 9 |
| [58](phase58_multicrop.md) | **Multi-crop in one pass: +7.9pp over single-crop at equal cost** | 9 |
| [37](phase37_coverage_mediator.md) | Coverage as the mediator; half of all windows miss | 9 |
| [4](phase04_dissociation.md) | The motivating dissociation | 8 |
| [13](phase13_bfs_interventions.md) | 0.4 tokens; pointing does nothing | 8 |
| [28](phase28_arch_invariance.md) | Dynamic range is architectural; "NO AXIS" reporting | 8 |
| [30](phase30_attn_proposer.md) | The localiser is in the map, the argmax is not | 8 |
| [32](phase32_conditional_allocator.md) | The conditional allocator earns its second pass | 8 |
| [33](phase33_hrbench_transfer.md) | 4K transfer + the rand control + the wrong-image bug | 8 |
| [34](phase34_column_mask.md) | The sink is columnar, not cornered | 8 |
| [42](phase42_method_qwen2vl.md) | Mechanism and method transfer to a second architecture | 8 |
| [45](phase45_adaptive_allocator.md) | `peak` predicts coverage for free | 8 |
| [59](phase59_multicrop_hrbench.md) | Multi-crop transfers (+8.0pp); the win is scale-bound | 8 |
| [64](phase64_component_ablation.md) | Component ablation null + the normalisation bug | 8 |

## Supporting (5–7/10)
1 · 5 · 6 · 7 · 9 · 11 · 14/15 · 16 · 18 · 21 · 22 · 29 · 31 · 35 · 36 · 38 · 39 · 40 · 43 · 44 ·
46 · 48 · 50 · 52 · 55 · 56 · 57 · 60 · 63 · 65

## Low / do not report (0–4/10)
0 · 2 · 3 · 8 · 10 · 12 · 19 · 23 · 24 · 25 · 26 · 49 · 54 · 61 · 62

## Retractions on the record
- **61, 62, 63 (first version)** — a double-normalisation bug in our own logit lens (caught in 64).
- **24** — `obj` probe AUROC 1.000 was an ROI-selection artefact.
- **35** — region-count explanation superseded by coverage (36/37).
- **38** — "interior optimum" withdrawn; every CI crosses zero.
- **43** — "rows carry no excess mass" is not universal (LLaVA bottom rows).
- **57** — our own scale-boundary mechanism refuted by the resolution ablation.
- **47 vs 53** — "ours is the only positive margin" holds at ~1500px, not at 4K.

## Prior art that pre-empts parts of the method
- **AttWarp** (2510.09741, ICLR 2026) — the saliency-warp idea (phase 65).
- **ViCrop** (2502.17422) — attention-guided crop + original image (phases 31/54).
- **DoLa** (2309.03883), **DeCo** (2410.11779) — layer-contrastive decoding (phase 63).

## Phases 69-89 — the method arc (2026-09-15/16)

**Standard from 2026-09-16:** every claim must hold on >1 model or be REJECTED, not caveated.
Gate: `REPLICATION_LEDGER.md`. Paper: `paper/iclr2027_submission.pdf` (19pp) · `PAPER_FLOW.md` v12 ·
`METHOD.md` · `scripts/test_invariants.py` (87 invariants).

### Track A — the read-out defect and pruning (the strong half)

| phase | class | one line | keep |
|---|---|---|---|
| [73](phase73.md) | FINDING | layers disagree; averaging dilutes. ⚠ signed-contrast part later REJECTED | 8/10 |
| 74 | REPLICATION | Qwen2-VL: averaging defect holds (+8.4pp); final-layer & signed-contrast claims **fail** | 9/10 |
| [75](phase75.md) | **★★ FINDING** | **layer-2 pruning worse than random; late read-out deletes 90% of tokens free** | **10/10** |
| [76](phase76.md) | NEGATIVE | vision tower at/below chance — localisation is built by the LM, not read off the image | 8/10 |
| 82 | INFRA | LLaVA extraction unblocked (merged-embedding separator detection) → 2nd model *family* | 7/10 |
| 83 | **REPLICATION** | **pruning result replicates on Qwen2-VL (+11.5pp); mechanism prediction FAILS** | **10/10** |
| 87 | **★★ FINDING** | **it is a THRESHOLD, not layer 2: all of L0–L14 catastrophic, L16+ free, +13.1pp step** | **10/10** |
| 88 | **★★ FINDING** | **attention quality (L17–19) and answer formation (L21) are DISSOCIATED, ρ=0.30** | **10/10** |
| 89 | ✗ ABORTED | VQA run hung 36 min at 0% CPU — disk 99% full, download blocked, no timeout | — |
| 93 | **★★ FINDING** | **collapse is LARGER on general VQA: layer-2 is 15.0pp (POPE) / 25.5pp (MMBench) BELOW random.** ⚠ "free pruning" retracted — V\*Bench-only | **10/10** |
| 94 | **★★ FINDING** | **early attention is no better than a coin-flip even for coarse culling** (+1.0pp vs random first cut). Method: cut blind early, rank late — 64% compute saved vs 39% | **9/10** |
| 95 | **★★★ MECHANISM** | **attention is QUESTION-BLIND until ~50% of depth** (divergence 0.001 → 0.14, 13× at L14→L16 on both models) — exactly the pruning threshold. Label-free locator | **10/10** |

### Track B — learned allocation (real, unfinished)

| phase | class | one line | keep |
|---|---|---|---|
| [70](phase70.md) | METHOD | learned re-ranking head: 39.3 → 52.9% evidence coverage | 9/10 |
| [71](phase71.md) | METHOD | it converts: 56.5 → 68.6% (**+12.0pp** vs vanilla) | 10/10 |
| [72](phase72.md) | MIXED | proposals transfer zero-shot to HR-Bench; the allocator still loses at 4K | 9/10 |
| [78](phase78.md) | FINDING | W=0.25 beats deployed 0.15 (in-sample); **+11.0pp sizer headroom, no predictor** | 7/10 |
| 90 | **★ FINDING** | **W=0.25 SURVIVES held-out selection** (71.7→71.6%, folds pick it 99/100). Track B's baseline moves to **+15.0pp** vs vanilla, **+7.7pp [−0.6,+16.0]** vs the bar | **9/10** |
| 91 | FINDING | restricted to per-layer maps, ALL layers beat any contiguous block or quality-selected subset — poor layers are negative evidence, not noise | 6/10 |
| 92 | ✗ VOID | premised on the head reading only L16–26. **It already reads all 28**; `BLOCK` only feeds one derived feature. No lever, and the "gain" was against a restriction we invented | 0/10 |
| [79](phase79.md) | **★★ FINDING** | **why it works: answer formation restored, only where the crop delivers** | 10/10 |
| [80](phase80.md) | MIXED | replicates vs vanilla (+10.5pp) but **not** vs the argmax it replaces | 9/10 |
| [81](phase81.md) | NEGATIVE | rank-only multi-crop −3.1pp vs a predicted +5.5pp; prompting confound excluded | 6/10 |
| 84 | **REPLICATION** | **L21 mechanism replicates on Qwen2-VL (+23.8 covered / +0.0 missed)** | **10/10** |
| 85 | INFRA | figure data from real inference: attention, masks, crop pixels, real answers | 8/10 |

### Track C — closed directions (measured, not assumed)

| phase | class | one line | keep |
|---|---|---|---|
| [69](phase69.md) | NEGATIVE | no free pass-1 signal predicts required budget (9.4% vs a 35.6% majority baseline) | 7/10 |
| [77](phase77.md) | NEGATIVE + causal | contrastive decoding fails, but proves the evidence region is load-bearing (3.32×) | 8/10 |

---

## Current state

**Survived (≥2 models):** averaging dilutes the read-out (3 of 4, 2 families) · anti-correlated
layers exist (4) · layer-2 pruning below random (2) · learned read-out improves proposals (2) ·
crop method beats vanilla (2) · answer formation restored by the crop (2) · the encoding cliff (2) ·
the serialisation sink (4).

**Rejected:** "the *final* layer is anti-correlated" (1 of 4) · "signed contrast is the mechanism"
(failed twice outside Qwen3 crop placement) · "no single layer is a good localiser" (false on
Qwen2-VL) · rank-only multi-crop · **crop method beats the equal-compute baseline (0 of 2)**.

**The two headline numbers.** Pruning: reading anywhere in **L0–L14** costs 15–24pp and L1/L2 fall
*below random*; reading **L16+** is free (late block 56.5% = no pruning 56.5%). Allocation:
**+12.0pp / +10.5pp** over an unmodified model, and **0 of 2** against simply spending the same
budget.

**Track B after phases 90–92.** W=0.25 is real: held-out selection costs 0.2pp and the folds pick it
99 times out of 100, so the honest baseline is **+15.0pp over vanilla** and **+7.7pp [−0.6,+16.0]**
against the equal-compute bar — about one point from the claim that is currently 0-for-2. The input
band is **not** a lever: phase 92 was premised on the head reading L16–26 when it already reads all
28 layers, so that path is void and the cheapest route to the missing point is gone.

**The one remaining Track B lever** is per-item window sizing: phase 78 measured **+11.0pp of stable
headroom**, target size is ruled out as a predictor, and attention-blob extent is untried. A second
option is retraining the head on post-crop *accuracy* rather than *coverage* — we optimise a proxy
for the thing we want, and the outcomes are already on disk. The method remains **unfinished, not
refuted**: it captures 27.6% of a ceiling where 88.5% of items have a covering cell available.

**Track A is now a method.** The defect: attention is question-blind through the first half of the
network — maps for different questions on the same image differ by ~0.001 until L13, then jump 13×
by L16, on both models at 50–57% of depth. Anything ranking visual tokens on attention read before
that is ranking on something that does not know what was asked, which is why layer-2 pruning is
**below random** on three benchmarks (−15.0pp POPE, −25.5pp MMBench) and why an attention-guided
coarse cut is **indistinguishable from a coin-flip**.

Three parts: **(1)** a label-free locator — run one image with several questions, find where the
maps diverge, no boxes needed; **(2)** prune past that depth: **+16.5 to +27.0pp** over the layer-2
default, no training; **(3)** to recover compute, cut half the tokens *blind* at L2 and rank the
survivors late — **64%** of visual-token compute saved against pure-late's 39%, at −4.7pp
[−9.9,+0.5].

**Open:** the locator's prediction that Qwen2-VL's pruning threshold sits at **L13–L14** is untested —
that run confirms or kills it as a general tool. Two-stage is V\*Bench-only. And the crop-placement
method (Track B) still sits at −0.6 against the equal-compute baseline.
