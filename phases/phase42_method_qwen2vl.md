# Phase 42 — Do the mechanism and the method transfer to a second architecture?

## 1. Research question
Coverage, the sink-masked localiser and the confidence gate are all measured on Qwen3-VL-2B. Do they
hold on a different model generation with 3.5× the parameters, with **nothing refitted**?

## 2. Finding and contribution (plain English)
All three transfer. The localiser beats random placement, the coverage dose–response reproduces with
the same sign flip, and the confidence gate detects coverage at AUROC 0.885 — better than on the
model it was designed for.

This removed the paper's main scope limit at the time.

It also produced the project's most instructive bug: loaded in fp16, this **bfloat16** checkpoint
produced all-NaN logits. `argmax` fell through to index 0, and because 71/191 V\*Bench labels are "A",
every arm scored an identical **37.2%** with CI **[+0.0,+0.0]** — which the analyzer reported as
"the localiser does not transfer". Two false negatives that looked exactly like findings.

## 3. Numbers that changed
| claim | Qwen2-VL-7B | Qwen3-VL-2B |
|---|---|---|
| localiser transfers (attn − rand) | **+19.9pp** [+11.5,+28.3] | +40.0pp |
| coverage: missed / full | **−9.4pp** / **+41.8pp** | −15.6 / +37.1 |
| gate detects coverage (AUROC) | **0.885** | 0.835 |
| deployed gate | **+7.9pp** [+2.1,+14.1] | +12.0pp |
| oracle | **91.6%**, +38.7pp | 97.4% |

## 4. Keep in paper: 8/10
The generality result. And the NaN bug belongs in the methods-discipline section: a dtype error that
masquerades as a clean scientific negative is worth warning people about.

## 5. Experiment, step by step
1. Refit **nothing**: W, the ring mask, B₀ and the relative layer block all come from Qwen3-VL.
2. Load in **bfloat16** and assert finiteness on both logits and attention before anything is logged.
3. Run uniform, oracle, attn@{0.15,0.25}, rand@{0.15,0.25} at a matched budget.
4. Log `peak_frac` and the GT box so coverage can be recomputed offline exactly as on the first model.
5. Check for the failure signature explicitly: all arms bit-identical **and** zero-width CIs.
