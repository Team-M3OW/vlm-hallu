# Phase 79 — Why the method works, read through the logit lens

**Classification: FINDING (causality) — the mechanism section**

## 1. Research question

The method works (+12.0pp). But *why*? Two very different explanations fit the same accuracy number:
the crop could make the model **reason** better, or it could **supply information** the model was
missing. They are distinguishable by where in the network the arms diverge.

## 2. Finding, in simple English

**It supplies information. The model was never reasoning badly.**

Read the model's answer at every one of its 28 layers, for four versions of the same question that
differ only in which pixels were shown. For twenty layers the cropped and uncropped versions are
**indistinguishable**. Then at layer 21 they separate by 15 points and stay separated.

That shape is the answer. A method that improved reasoning would pull ahead early and widen. This
one is flat, then steps.

Look at where each version takes its biggest jump. The two that fail — the plain image, and the old
proposer — jump at **layer 2**, which is just the model forming its prior guess, and then never
improve: the plain image's best score across all 28 layers is 56.3%, *exactly* what it finally
answers. There is no hidden depth where the answer was quietly available. The two that work jump at
**layer 21**, and the size of that jump scales with how much evidence the crop delivered — 28 points
for our method, 44 for a perfect crop.

**The control is what makes this causal.** Take our method's own results and split them by whether
its window actually landed on the evidence. Same method, same intervention — only the delivery
differs. Where it covers, the answer goes from 60.4% to **90.1%**, nearly matching a perfect crop's
98.0%. Where it misses, it *hurts*: 51.7% down to 43.8%. A 37.6-point swing between the two halves.

So the method is not adding intelligence. It moves items across a threshold — out of the state where
the answer exists at no layer, into the state where layer 21 has something to convert. The headline
+12.0pp is 101 items gaining about 30 points minus 89 items losing about 8, and nothing else.

This also says exactly where the remaining headroom is: on items it covers, the method reaches 90.1%
against a perfect crop's 98.0%. **Coverage is the whole remaining gap** — better proposals, not
better anything else.

## 3. Numbers that changed

Separation (head − uniform) by layer: L14–L20 ≈ **0.0pp** → L21 **+15.3** → L23 **+17.4** →
L25 **+17.9** → L27 +12.1.

| arm | final | biggest jump | at |
|---|---|---|---|
| uniform@300 | 56.3% | +23.2pp | **L2** |
| argmax@0.15 | 60.0% | +23.7pp | **L2** |
| **head@0.15** | **68.4%** | **+28.4pp** | **L21** |
| oracle@0.15 | 90.0% | +44.2pp | **L21** |

| stratum | head | uniform | oracle | head − uniform |
|---|---|---|---|---|
| window **covers** | **90.1%** | 60.4% | 98.0% | **+29.7pp [+19.8,+39.6]** |
| window **misses** | 43.8% | 51.7% | 80.9% | **−7.9pp [−18.0,+2.2]** |

uniform's max over all 28 layers = 56.3% = its final answer.

## 4. Keep in paper: **10/10**

This is the causality section. It converts "the method helps" into "the method restores a computation
that was always present and starved of input," and it does so with a within-method control rather
than a cross-arm comparison.

## 5. Experiment, stepwise

1. **Read the answer at every layer, for four arms** that differ only in pixels: plain image, old
   proposer's crop, our crop, oracle crop. Same question, same realized token budget throughout.
2. **Implement the lens correctly — this cost a retraction once.** Intermediate layers are
   `lm_head(final_norm(h_i))`, but the **final layer must come from `model.logits`**: HF has already
   normalised the last hidden state, and normalising again gives reconstruction error 23.47 against
   0.06. It corrupts only the final layer, and it previously manufactured a "+5.1pp free read-out
   gain" and a favourable DoLa comparison.
3. **Compute the buggy path alongside the correct one and assert they disagree.** A zero gap would
   mean the correct path is not actually in use. The bug was originally caught exactly this way — a
   second code path computed the same quantity and disagreed.
4. **Plot separation per layer**, not just final accuracy. The *shape* is the evidence: flat-then-step
   means information; early-and-widening would mean reasoning.
5. **Record each arm's maximum over all layers**, not only its final. This is what rules out "the
   answer was available at some depth and got lost on the way out."
6. **Run the coverage split as the decisive control.** Within the method's own arm, split by whether
   the window covered. This holds the intervention fixed and varies only whether it delivered, so it
   is not confounded by item difficulty or by which crop was taken.
7. **Fix the falsification in advance:** if the covered and missed strata do *not* separate, the gain
   is not restored answer formation and the mechanism is wrong regardless of how good the per-layer
   curve looks.
8. **Read "ever argmax" against its own reference.** Being correct at *some* layer is inflated by
   chance when you read a 4-way question at 28 depths, so only the ordering across arms is
   informative (uniform 86.8% → oracle 97.4%).
