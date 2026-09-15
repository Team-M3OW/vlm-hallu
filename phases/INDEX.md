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

## Phases 69-78 — the method arc (2026-09-15)

| phase | class | one line | keep |
|---|---|---|---|
| [69](phase69.md) | NEGATIVE | no free pass-1 signal predicts required budget; adaptive-budget family closed | 7/10 |
| [70](phase70.md) | METHOD | learned re-ranking head: 39.3% -> 52.9% top-1 evidence coverage | 9/10 |
| [71](phase71.md) | METHOD | it converts: 56.5% -> 68.6% (+12.0pp vs vanilla); +13.0pp over the 2-pass bar on single-region | 10/10 |
| [72](phase72.md) | MIXED | re-ranker transfers zero-shot to HR-Bench (+4.9pp, +6.5 circular); allocator still loses at 4K | 9/10 |
| [73](phase73.md) | FINDING | the mechanism: layers disagree, the final layer is anti-correlated, a mean can only add | 10/10 |
| 74 | — | cross-architecture replication — NOT RUN (launcher died; 7B models, GPU contended) | — |
| 75 | — | token pruning vs FastV — WRITTEN, NOT RUN | — |
| [76](phase76.md) | NEGATIVE | vision tower is at chance; localisation is built by the LM from the question | 8/10 |
| [77](phase77.md) | NEGATIVE + causal | contrastive decoding fails (-5.5pp vs bar) but proves the evidence region is load-bearing (3.32x, -10.5pp) | 8/10 |
| 78 | — | is there anything for a learned allocation function to learn? — RUNNING | — |

**Closing method:** see `METHOD.md` — gated Depth-Contrast Re-ranking, **+16.2pp [+10.5,+22.5]**
over the vanilla VLM at 1.61 passes; **+12.0pp [+4.2,+19.9]** ungated.
