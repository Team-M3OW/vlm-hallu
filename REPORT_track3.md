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

## 5. End-task (phase 141): does any training-free rule clear the equal-compute bar?

V\*Bench, n=191 each, W=0.25, budget drift 1.2% / 0.7%. `head@0.25` re-run as pipeline control:
matches the stored value on **100%** of items on both models. Bar = uniform@600.

| single-object vs bar | Qwen3-VL | Qwen2-VL |
|---|---|---|
| **learned head** | **+15.7 [+6.1,+25.2]*** | **+11.3 [+1.7,+20.9]*** |
| raw div-gated max (label-free) | **+10.4 [+0.9,+20.9]*** | +7.0 [−2.6,+16.5] |
| composite div-gated max (pre-registered primary) | +9.6 [−0.9,+20.0] | +7.8 [−1.7,+17.4] |
| CLAA max_win4 (phase 106, reference) | +9.6 [+0.0,+19.1] | +6.1 [−3.5,+15.7] |

Pooled: rawdiv +6.8 / +3.1, compdiv +5.8 / +4.2, head +7.9 / +6.8 — none clears. Relational: all null.
head − rawdiv on single: +5.2 [−0.9,+11.3] / +4.3 [−0.9,+9.6]; head − compdiv: +6.1 [+0.9,+12.2]* / +3.5 [−2.6,+9.6].

**Verdict.** No fixed-constant training-free rule clears the equal-compute bar on both models. The best
label-free rule (max over the divergence-gated layer set, raw maps, no boxes, no per-head maps) clears
on Qwen3 only — 1 of 2, rejected under the standing rule. The pre-registered composite clears on
neither. The learned head remains the only arm clearing on both, by ~4–5pp on single-object, with
lower bounds just under zero.

**What the training-free counterpart is worth stating as:** a one-line, label-free rule — *max over the
layers where attention is question-conditioned* — that recovers **+10.4 / +7.0pp** over equal compute on
single-object questions and closes the coverage gap to the head to 3.7 / 2.7pp, but does not
significantly clear the bar on the second model. Its advantage over CLAA's max_win4 is that the layer
set comes from a label-free measurement rather than out-of-fold selection on boxes.

## 6. LLaVA: the divergence gate does not transfer (phase 142)

A phase-95 curve was measured for both LLaVA models (40 images × 4 questions, no boxes). The
**mechanism replicates on a second family**: attention is question-blind until mid-depth, then jumps
— LLaVA-NeXT 0.036 → **0.408 at L14 of 32** (44%), LLaVA-OneVision 0.049 → **0.289 at L16 of 28**
(57%). Four architectures, two families, same switch-on region. But the gated max does **not**
rescue localisation there:

| gate ≥0.5·max | LLaVA-NeXT (L14, L17) | LLaVA-OneVision (L16,19,21,22,27) |
|---|---|---|
| deployed mean block | 23.0 | 35.6 |
| div-gated max | 25.1 (+2.1 [−3.1,+7.3]) | 31.4 (−4.2 [−8.9,+0.5]) |
| div-gated mean | 23.0 (+0.0) | 35.1 (−0.5) |

Robust across gate constants 0.3–0.7. Max-over-layers hurts OneVision under every layer set, and the
gate cannot fix that: OneVision's problem is not *which* layers but that its layers agree (§14N), so
neither max nor any weighting has disagreement to exploit. **The training-free rule is Qwen-only;
the learned head stays 3 of 4.**

## 7. Verdict

1. **Not additive.** Norm weighting, head selection and max-over-layers each fix the sink; stacked,
   they land on the same ~56 / ~49 coverage. The only component that adds is the **layer set**.
2. **The best fixed-constant, label-free locator** is *max over the layers where attention is
   question-conditioned* (phase-95 gate, raw maps): 56.0 / 52.4 coverage, +9.9 / +12.6 over the
   deployed rule, and on Qwen2 it beats norm-weighted max (+3.7 [+1.0,+6.8]). With heads + ‖v‖ it
   reaches 59.7 / 51.8 — **3.7 / 2.7pp short of the learned head**.
3. **End-task: no training-free rule clears the equal-compute bar on both models.** The label-free
   rule clears on Qwen3 (+10.4 [+0.9,+20.9]) but not Qwen2 (+7.0 [−2.6,+16.5]); the pre-registered
   composite clears on neither. The head clears on both, by ~4–5pp on single-object.
4. **Dead:** ReAttn entropy rescaling (−9 to −27pp), VEA denoising (−1 to −3pp on Qwen), phase-30d
   background on top of max/heads (null to negative).
5. **Cross-family:** every training-free rule is Qwen-only. The divergence *mechanism* replicates on
   LLaVA (switch-on at 44% / 57% of depth), the *rule* does not (+2.1 n.s. / −4.2).

For the paper: the training-free counterpart is a real, citable one-liner — it recovers most of the
read-out gap with no boxes and a label-free layer set — but the learned head is what converts to a
win at equal compute, on both models and both families where proposals were tested.

**Files.** scripts/phase140_trainfree_ladder.py, phase140b_gate_sensitivity.py,
phase141_trainfree_endtask.py, phase142_llava_divergence.py, phase142b_llava_gate.py;
data/phase140_ladder_hits.json, phase140_proposals_qwen{3,2}.json,
phase141_trainfree_endtask_qwen{3,2}.jsonl, phase142_llava_divergence.json; logs/phase14{0,1,2}*.log.
Nothing committed; FINDINGS/LEDGER/PAPER_FLOW untouched.
