# Phase 52 — Residual-stream steering: is the oracle-crop state reachable by a linear shift?

## 1. Research question
Phase 51 predicted residual steering would fail for the same reason. Test it: take the displacement
`h_oracle − h_uniform` at the last token, and inject it into the uniform pass.

## 2. Finding and contribution (plain English)
The prediction was **half wrong**, which is why it was worth running.

A shared "allocate to the evidence" direction **does exist** — the displacements do not cancel
(cosine 0.53–0.75) — and injecting an item's **own** displacement at L20 recovers 47% of the crop's
benefit after seven more layers of processing.

But the transferable **mean** direction gives **+0.0pp at every layer**, losing to a norm-matched
random vector. The shared direction is geometrically real and **answer-irrelevant**: it carries the
*statistics* of a cropped image, not the evidence that makes the target legible.

One arm must not be misread: injecting at L26 recovers the entire gap, and that is **near-tautological**
— L26 is second-to-last of 28, so the injection sets the residual equal to `h_oracle` two layers from
the head. It is labelled and excluded from the headline.

## 3. Numbers that changed
| injection layer | v_item | v_mean | v_rand |
|---|---|---|---|
| L8 | −1.2 | +0.0 | +0.0 |
| L14 | −0.6 | −1.2 | −0.6 |
| **L20** | **+17.3\*** | +0.0 | −0.6 |
| L26 *(tautological)* | +36.6\* | +0.0 | +1.6 |

`v_mean − v_rand = −1.6pp` [−5.2,+2.1], n.s. Displacement geometry: |mean d| / mean|dᵢ| =
0.78 / 0.75 / 0.68 / 0.58 at L8/L14/L20/L26.

## 4. Keep in paper: 7/10
Completes the three-intervention argument. The tautological-arm caveat is essential — without it the
+36.6pp line reads as a triumph.

## 5. Experiment, step by step
1. **Stage 1:** run each item twice at the same budget (uniform, oracle crop), capturing the last
   token's residual at several layers. Store the displacement.
2. **Stage 2:** inject three vectors — the item's **own** displacement (a reachability test, not a
   method), the **out-of-fold mean** (the actual method), and a **norm-matched random** vector.
3. Sweep injection layer and strength.
4. Exclude layers adjacent to the read-out from the headline, and say why.
5. Report the displacement geometry (mean norm vs norm of the mean) to show whether a shared
   direction even exists.
