# Phase 61 / 61b — Why do the last layers hurt, and can a free rule exploit it?

> **v1 ⛔ RETRACTED** (double-normalisation bug). **Re-run correctly as 61b — see §3 and §4.**

## 1. Research question
Phase 60 appeared to show that decoding at L24 beats the final layer on sub-token items. What in
L25–27 destroys the signal, and can a label-free rule exploit it?

## 2. Finding and contribution (plain English)
**Every finding in this phase is an artefact** of the double-normalisation bug in Phase 60's lens —
L24 was being compared against a corrupted final layer. See Phase 64.

For the record, the artefact was coherent and mechanistically plausible, which is what made it
dangerous: the damage appeared to concentrate where visual evidence is weak, to reverse where it is
strong, and the flips concentrated on one option letter. It even survived its own falsification test
(a static prior correction failed while truncation worked, suggesting an input-dependent bias). All
of it dissolved when the final layer was computed correctly.

The one durable lesson is in the design, not the result: the "prior removal" ablation was the right
instrument, and it *correctly* reported that a static vector does not explain the pattern.

## 3. Numbers that changed
**v1 (all retracted):** L24 vs final +5.1pp on sub-token; drop reversing on oracle/resolvable (−1.8);
flips 78% on 'B' vs a 37% base rate.

**61b, corrected (final layer from `model.logits`):**

| stratum | final | CV-chosen layer | delta |
|---|---|---|---|
| all items | 56.5% | 56.0% | −0.5 |
| sub-token | 52.9% | 51.5% | −1.5 |
| **resolvable** | 65.5% | 58.2% | **−7.3 [−14.5,−1.8] SIG** |

L24 vs final is **+0.0 in all four condition × stratum cells**. Flips fall from 9 to **2**, so the
letter-concentration finding disappears. Churn is symmetric (2 vs 2). No free depth rule beats the
final layer. `oracle_layer` ceiling +30.4pp — real headroom, unreachable by any label-free rule.

**The corrected result is stronger than "no gap": on resolvable items, choosing a layer is
significantly WORSE than reading the final one.**

## 4. Keep in paper: 5/10 (as the corrected negative) · 6/10 as a methods cautionary note
Do not report any of it as a finding. It is worth one paragraph in the discipline section: a
plausible mechanism, a supporting ablation, and a falsification test can all be built on one
corrupted number.

## 5. Experiment, step by step
1. Four hypotheses with distinct offline signatures: evidence-dependent prior fallback, evidence-
   independent degradation, a sharpening artefact, and noise.
2. Test evidence-dependence by crossing condition (uniform / oracle) with stratum (sub-token /
   resolvable).
3. Test the prior hypothesis by estimating the option prior on training folds and dividing it out.
4. Test depth-selection rules out of fold: fixed layer, max confidence, stability over k layers.
5. **What was missing:** an independent recomputation of the baseline. Added in Phase 64.
