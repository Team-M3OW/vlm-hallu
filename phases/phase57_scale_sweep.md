# Phase 57 — The scale sweep: same items, only resolution varies

## 1. Research question
Is the 4K failure about **image scale**? Downsample HR-Bench's 4032² images to 1008 and 2016px —
same items, same questions, same answers — so scale is isolated with everything else fixed.

## 2. Finding and contribution (plain English)
It **refuted our own stated mechanism**. We had claimed that larger images mean coarser localiser
cells and a worse proposal. At **4× finer cells the margin is worse**, and it is not monotone in
scale. Proposal precision is not the binding constraint.

Then stratifying by category dissolved the boundary into the coverage account: `cross` is negative at
every scale, `single` turns positive at native resolution, and the pooled −0.2pp is simply their
average because HR-Bench is 50% multi-region. **No separate mechanism is needed.**

A confound I had to state: downsampling is not the same as a natively smaller image — it destroys the
detail a crop would reveal — so the sweep varies available detail as well as scale.

## 3. Numbers that changed
| scale | cell px | gated margin | **cross** | **single** |
|---|---|---|---|---|
| 1008px | 59 | −3.6 | −4.4 | −2.4 |
| 2016px | 119 | +0.1 | −1.6 | **+1.6** |
| 4032px | 237 | −0.2 | −4.0 | **+3.5** |

## 4. Keep in paper: 7/10
Keep for the retraction and the unification. A refuted own-hypothesis that resolves into the existing
mechanism is a strong thing to be able to show.

## 5. Experiment, step by step
1. Reject MMBench first, on measurement: its images cap at **512px**, so uniform@300 already resolves
   them and a null there would be a benchmark-selection artefact.
2. Downsample the same 800 HR-Bench rows to 1008 and 2016px; join the native 4032 arm from an earlier
   run on identical rows.
3. Measure the bar **at each scale** from that scale's own uniform arms.
4. Pre-register the prediction (margin rises as scale falls) so the refutation is unambiguous.
5. Stratify by category, because "tokens on target" means something different when the box is a union
   of two objects.
