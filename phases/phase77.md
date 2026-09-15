# Phase 77 — A decoding-time fix: contrast the evidence region

**Classification: NEGATIVE as a method, POSITIVE as causal evidence**

## 1. Research question

Everything so far needs a crop. Is there a fix that works purely at decoding time — no extra pixels,
no re-encoding, no cutting the image?

The idea: run the model normally, run it again with the evidence region masked out of its attention,
and push the logits away from the masked version. Six earlier internal interventions failed, but all
six either amplified attention (there is nothing to amplify when the target is sub-token) or steered
to a bad location. This one uses a working localiser and a different operation — contrast, which only
needs the region to *contribute* something, not to be attended more.

## 2. Finding, in simple English

**As a method it fails. As evidence it is the cleanest thing we have.**

First the evidence. Masking the evidence region moves the model's output **3.32× more** than masking
a random region of the same size, and costs **10.5 points** of accuracy. So the model genuinely does
read the right place, and the right place genuinely matters.

That closes a hole. An earlier phase had concluded image-token attention was "not a scarce resource"
because blocking random, interior, or corner cells did nothing — but it had no working localiser.
With one, the evidence region is plainly load-bearing.

Now the failure. Pushing the logits away from the masked version buys **+2.1pp**, and the CI includes
zero. Even the **ceiling** — doing this with the ground-truth region instead of a predicted one —
reaches only +2.6pp. Meanwhile the method costs two forward passes, and spending those same two
passes on a bigger image buys +7.4pp. It loses to its own bar by 5.5 points.

The falsification test came out directionally right: the gain is larger where the proposal covers
(+3.0pp) than where it misses (+1.1pp), so the effect really is about evidence — it is just far too
small to matter.

**Why this is worth reporting.** It is the seventh internal intervention to fail, and the first to
fail *while aiming at the ground truth*. The previous six could always be dismissed as "you steered
to the wrong place." This one steered perfectly, proved the place was causally live, and still could
not convert it. That turns a pile of nulls into a statement:

> The model looks in the right place, and the right place is load-bearing. It still cannot answer,
> because what is encoded there is not enough — and no decoding-time operation can add information
> that was never encoded.

## 3. Numbers that changed

| region masked | logit shift (L1) |
|---|---|
| **evidence (head)** | **3.348** |
| evidence (oracle GT box) | 2.906 |
| **random, same size** | **0.874** |

Masked-only pass: **45.8%**, i.e. −10.5pp against baseline's 56.3%.

| arm (best α) | acc | vs baseline |
|---|---|---|
| baseline | 56.3% | — |
| cd_oracle (ceiling) | 58.9% | +2.6pp [−0.5,+5.8] |
| cd_head | 58.4% | +2.1pp [−1.1,+5.3] |
| cd_rand (control) | 56.3% | +0.0pp [−1.6,+1.6] |
| uniform@600 (bar) | **63.9%** | method is **−5.5pp** |

## 4. Keep in paper: **8/10**

The method half is a footnote. The causal half belongs in the mechanism section — it is the
strongest single piece of evidence for why allocation has to add pixels.

## 5. Experiment, stepwise

1. **Mask by attention bias, not by deleting tokens.** A large negative additive bias on the
   region's image-token columns, applied at every layer, so positions and token count are untouched
   and the two passes are exactly comparable.
2. **Four regions, each the same window geometry:** oracle (GT box centre), head, argmax, random.
   The random arm is what separates evidence-grounding from generic logit sharpening.
3. **Sweep α** over 0.25–2.0 and report the whole curve rather than the best point.
4. **Record the raw logit shift** for each region, not just accuracy — that is what detects whether
   masking does anything at all, independently of whether it helps.
5. **Fix the decision order before looking.** Does masking do anything → does the oracle ceiling move
   → does the head beat random → does the gain track coverage. Stop at the first failure rather than
   reporting downstream numbers that cannot mean anything.
6. **Set the bar at two passes.** The method runs the model twice, so `uniform@600` is the floor,
   not `uniform@300`.
7. **Flag the prior art as unchecked.** Contrastive decoding for VLMs is crowded — VCD, ICD,
   Pensieve, RITUAL. Grounding the contrast in a learned localiser looks unclaimed, but that is
   unverified and must not be asserted before a literature sweep.
