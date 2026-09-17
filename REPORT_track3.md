# Track 3 — the no-training counterpart (phases 140, 140b, 141)

**Question.** How close can a fixed-constant, label-free rule get to the learned head (63.4 / 54.5),
and does any such rule clear the equal-compute bar end-task?

All rules below are **fixed constants** — no out-of-fold selection anywhere. Block L16–26 (Qwen3) /
L15–26 (Qwen2) as deployed; head fraction q=0.10; VEA λ=10; 8×8 LOO background (phase 30d);
divergence gate = layers with phase-95 question-divergence ≥ 0.5·max. Metric: top-1 coverage at
W=0.25, ring-masked. Noise floor ~1pp. `*` = CI clear of zero.

## 1. The ladder (V\*Bench, n=191 each)

| rule | needs | Qwen3-VL | Qwen2-VL |
|---|---|---|---|
| deployed: raw mean over block | — | 46.1 | 39.8 |
| raw **max** over block (CLAA) | — | 53.4 (+7.3*) | 46.6 (+6.8*) |
| norm-weighted mean over block | ‖v‖ | 50.8 (+4.7*) | 42.9 (+3.1*) |
| norm-weighted **max** over block | ‖v‖ | 56.5 (+10.5*) | 48.7 (+8.9*) |
| S_v top-10% heads, max over block | per-head | 55.0 (+8.9*) | 48.7 (+8.9*) |
| **composite** heads + ‖v‖ + max over block | both | 56.5 (+10.5*) | 49.7 (+9.9*) |
| raw, **divergence-gated max** | phase-95 curve | 56.0 (+9.9*) | **52.4 (+12.6*)** |
| **composite, divergence-gated max** | all three | **59.7 (+13.6*)** | 51.8 (+12.0*) |
| *learned head (reference)* | boxes + training | *63.4* | *54.5* |

Against the strongest pre-existing fixed rule (norm-weighted max, 56.5 / 48.7):

| | Qwen3 | Qwen2 |
|---|---|---|
| composite (heads+‖v‖+max) − nw-max | +0.0 [−3.1,+2.6] | +1.0 [−2.6,+4.7] |
| raw div-gated max − nw-max | −0.5 [−3.7,+2.6] | **+3.7 [+1.0,+6.8]*** |
| composite div-gated max − nw-max | +3.1 [+0.0,+6.8] | +3.1 [−0.5,+6.8] |

## 2. What each component does

- **Not additive.** Norm weighting, head selection and max each beat the deployed rule, but stacked
  they land on the same ~56 / ~49 — they remove the same sink by different routes. The composite of
  all three over the deployed block is worth **+0.0 / +1.0** over norm-weighted max alone.
- **The layer SET is the lever that was left.** Replacing the hand-set block with the layers where
  attention is question-conditioned (phase 95, label-free) is the only component that adds on top of
  the others: +3.1 / +3.1 vs nw-max, and on raw maps alone +3.7* on Qwen2. The gate is robust —
  thresholds 0.3 / 0.5 / 0.7 select the same layers (**L17–20** on Qwen3, **L19–22** on Qwen2) and
  move coverage by ≤1.6pp. Note the gate independently re-finds Qwen2's L21 (§14K's best single
  localiser, §14I(b)'s answer-formation layer).
- **ReAttn entropy rescaling: catastrophic** in both directions on both models (−19 to −27pp on
  norm-weighted maps, −2 to −8 on raw). Early layers are diffuse; any entropy-based reweighting
  either promotes them or over-concentrates. Do not ship.
- **VEA denoising (λ=10): hurts** on Qwen3 (−1.0 / −1.6 on the composite) and Qwen2 (−3.1 / −2.1).
  Its one positive is LLaVA-OneVision (+1.6 [+0.0,+3.7]), too small to claim. Isolated high-value
  cells on V\*Bench are usually the *target*, not noise — the opposite of document VQA.
- **Phase 30d background normalisation**: null-to-negative on the composite (−1.6 / +1.0); positive
  only on Qwen2 raw maps (+8.9* over deployed, +2.6 n.s. over nw-max). Not additive with max/heads.

## 3. Cross-family

Only raw-map components exist for LLaVA (no ‖v‖, no per-head maps stored).

| | LLaVA-NeXT | LLaVA-OneVision |
|---|---|---|
| deployed | 23.0 | 35.6 |
| raw max block | 26.2 (+3.1 n.s.) | **31.4 (−4.2, worse)** |
| raw + denoise | 25.1 | 37.2 (+1.6, lower bound 0.0) |
| raw + bgLOO | 24.6 | 33.5 |
| entropy-weighted | 18–20 (worse) | 27.2 (worse) |

No training-free rule helps LLaVA; max-over-layers actively hurts OneVision. **The divergence gate
cannot be tested on LLaVA** (no phase-95 curve exists for it) — that is the one experiment that
could change this, and it needs ~4 extra forward passes per image, no boxes. The learned head
remains 3 of 4 across families; the best training-free rule is Qwen-only as far as measured.

## 4. Gap to the head

| | Qwen3 | Qwen2 |
|---|---|---|
| learned head | 63.4 | 54.5 |
| best training-free (composite div-gated max) | 59.7 | 51.8 |
| **gap** | **3.7** | **2.7** |

Down from 17 / 15 against the deployed rule, and from 6.9 / 5.8 against norm-weighted max.

## 5. End-task (phase 141) — PENDING
