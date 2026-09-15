# Phase 62 / 62b — Do allocation and the read-out compose?

> **v1 ⛔ PARTLY RETRACTED.** **Re-run correctly as 62b — see §3.**

## 1. Research question
Multi-crop supplies information; the L24 read-out (apparently) stops information being discarded.
They act on different failures, so their gains should add. Do they?

## 2. Finding and contribution (plain English)
The composition arithmetic was sound, but **the read-out half was an artefact** (Phase 64), so the
composition claim is withdrawn.

What survives and is unaffected: the **allocation** main effect measured here, **+19.9pp** on
sub-token items, which agrees with Phase 58's independent measurement.

This phase also produced the first hint of the bug: the read-out gain came out **+2.9pp** here versus
**+5.1pp** in Phase 60 on identical data, differing only by the CV fold split. That fragility was the
signal that something was wrong, and chasing it led to the seed sweep and then to Phase 64.

## 3. Numbers that changed
| sub-token (n=136) | Δ vs uniform/final |
|---|---|
| allocation alone (multi4) | **+19.9** [+11.0,+28.7] ✅ stands |
| read-out alone (L24) | +2.9 [−2.9,+8.8] ⛔ artefact |
| both | +24.3 [+14.0,+34.6] ⛔ partly artefact |

Robustness check that exposed it: across **20 CV seeds** the read-out gain is +4.0pp mean, range
[+0.7,+5.1], significant in only **13/20** splits — while allocation needs no fold split at all.

**62b, corrected (final layer from `model.logits`; join checks 99.5% / 81.2% as predicted):**

| effect vs uniform/final | all items | sub-token |
|---|---|---|
| **allocation alone (multi4)** | **+12.0 [+4.2,+19.9] SIG** | **+19.9 [+11.0,+28.7] SIG** |
| read-out alone | −1.0 [−2.6,+0.0] | −1.5 [−3.7,+0.0] |

**Nothing to compose with** — the read-out term is negative. But the allocation effect is
**identical to the original**, because both arms were corrupted the same way and their difference
survived. The measurement was never wrong; the interpretation built on the read-out half was.

## 4. Keep in paper: 4/10 (62b independently confirms the allocation effect)
Cite only the allocation main effect, which is better sourced from Phase 58. The seed-sweep
robustness check is worth a line in the methods section.

## 5. Experiment, step by step
1. Capture per-layer distributions for uniform, top1 and multi4 on the same items.
2. Build the 2×2: {arm} × {final layer, CV-chosen layer}.
3. Compare the observed combined effect against the sum of the main effects.
4. **The check that mattered:** re-run the same quantity under different fold seeds. A real effect
   should not swing from significant to null on the shuffle.
