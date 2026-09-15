# Phase 51 — Allocate in ATTENTION space instead of pixel space

## 1. Research question
Phase 50 tested the destructive half. This is the constructive half: instead of giving the evidence
more **pixels**, give its tokens more **attention weight**. One pass, same budget, no crop. The oracle
arm decides it.

## 2. Finding and contribution (plain English)
This is the paper's cleanest causal result, and it is a negative.

Amplifying attention on **exactly the right cells** gains +5.8pp — but the pixel-space oracle crop
gains +36.2pp at the same budget. Attention recovers **16%** of what cropping recovers. And every
*practical* arm (top-1, top-5, top-15, the deployed window) is **indistinguishable from random**.

The reason is visible in one number: **the oracle amplification set is 1 cell of 294.** The target is
sub-token, so there is nothing to amplify. **Cropping creates tokens; attention can only reweight
tokens that already exist.**

This rules out "the model just isn't looking in the right place" — we *made* it look there and got a
sixth of the benefit.

## 3. Numbers that changed
| arm | +1 | +2 | **+4** | +8 |
|---|---|---|---|---|
| **amp_oracle** | +1.0 | +3.1 | **+5.8\*** [+1,+10] | −13.1\* |
| amp_win / top1 / top5 / top15 | ≈0 | ≈0 | ≈0 | −10 to −14\* |
| amp_rand (control) | +1.6 | +1.0 | −1.6 | −9.4\* |

Pixel-space oracle: **+36.2pp**. Attention recovers **15.9%** of it.

## 4. Keep in paper: 9/10
The causal core. It converts "allocation works" from an empirical observation into a claim about
*why* it must be pixels.

## 5. Experiment, step by step
1. Reuse Phase 50's verified attention-bias patch, now with positive bias.
2. Build `amp_oracle` (cells whose centres fall inside the GT box) as the ceiling — it is the
   attention-space counterpart of the oracle crop.
3. Build practical arms from the attention ranking (top-1/5/15) and from the deployed window.
4. Include `amp_rand` at matched cell count.
5. Sweep the amplification strength; report the full grid.
6. Compare the oracle arm against the **pixel-space** oracle at the same budget — that ratio is the
   result.
