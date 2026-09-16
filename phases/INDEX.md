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
| 89 | **RUNNING** | pruning on POPE/GQA/TextVQA/ScienceQA — decides whether the scope is general VQA | — |

### Track B — learned allocation (real, unfinished)

| phase | class | one line | keep |
|---|---|---|---|
| [70](phase70.md) | METHOD | learned re-ranking head: 39.3 → 52.9% evidence coverage | 9/10 |
| [71](phase71.md) | METHOD | it converts: 56.5 → 68.6% (**+12.0pp** vs vanilla) | 10/10 |
| [72](phase72.md) | MIXED | proposals transfer zero-shot to HR-Bench; the allocator still loses at 4K | 9/10 |
| [78](phase78.md) | FINDING | W=0.25 beats deployed 0.15 (in-sample); **+11.0pp sizer headroom, no predictor** | 7/10 |
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

**What is untried on Track B**, and why it matters: the head still reads L16–26, a band fixed before
phase 88 showed quality peaks at L17–19 and declines past L21. Held-out validation of W=0.25 (+3.1pp
in-sample) and a sizer driven by attention-blob extent (+11.0pp stable headroom) are also open. The
allocation method is **unfinished, not refuted** — it captures 27.6% of a ceiling where 88.5% of
items have a covering cell available.
