# Phase 66 — The founding question, answered: localisation is EMPTY

## 1. Research question
The project began from a dissociation: attention ranks the GT cell in the top 3.4%, yet the model is
still wrong. Is the correct answer **present internally and mis-routed**, or **never there at all**?

## 2. Finding and contribution (plain English)
**Never there.** On items the model localises but answers wrong, the correct option is decodable at
**no layer** — 72.2% "ever argmax at some layer" versus 68.1% for items where the window missed
entirely, which is no difference. Localising buys essentially nothing internally.

A trap had to be ruled out: the correct answer is argmax on 41.7% at L20 and collapses to 0% from
L22, which reads exactly like propagation failure. It is not — at L18–L20 these items score +1.7 and
+1.9pp relative to those layers' **overall** accuracy, i.e. precisely what a near-chance layer
produces on any wrong-answered subset.

And the decisive contrast: the **oracle crop fixes 94.4%** of them, at the same budget and the same
question. Median tokens on target for this cell: **0.14**.

> The model looks in the right place and the tokens there do not contain the answer. "Localises well"
> never implied "has the answer" — and the gap between them is **resolution**, not routing.

**The cliff.** Locating the threshold: on cleanly-localised single-region items the step is
**19.0% → 78.6%** at **0.15 merged tokens**, with disjoint CIs — and the **oracle is flat across it**
(100.0% vs 97.6%), so it is encoding, not difficulty. Below the cliff the model scores *below* the
25% chance level.

**It is architecture-invariant.** Both Qwen3-VL-2B and Qwen2-VL-7B step at **0.25 tokens** on all
single-region items, with disjoint CIs and an oracle flat at **identical values** (95.8% / 100.0%).

## 3. Numbers that changed
| | answered RIGHT | answered WRONG |
|---|---|---|
| localised | 59 | **36** |
| window missed | 49 | 47 |

| cell | ever argmax at some layer |
|---|---|
| localised & right | 100.0% |
| **localised & WRONG** | **72.2%** |
| missed & WRONG | 68.1% |

Cliff (localised, single-region, n=63): **19.0% → 78.6%** at 0.15 tokens; oracle 100.0% → 97.6%.
Cross-architecture (all single-region, n=115 each): cliff at **0.25 tokens** on both models.

## 4. Keep in paper: 10/10
This is the paper's core finding and the resolution of the question it started from. Everything else
is downstream: the exchange rate is large because allocation moves targets across this cliff; coverage
governs the sign because a missed window leaves the target below it; the internal interventions all
failed because below the cliff nothing is encoded; the method works because adding pixels is the only
operation that crosses it.

It also survives both prior-art hits: AttWarp and ViCrop are **methods for crossing** the cliff;
neither characterises the cliff itself.

## 5. Experiment, step by step
1. Build the 2×2: localised (the deployed window covers the GT) × answered correctly.
2. For the localised-but-wrong cell, ask whether the correct option is argmax at **any** layer.
3. **Rule out the near-chance artefact:** compare each layer's accuracy on this cell against that
   layer's accuracy on **all** items. A difference of ≈0 means the "correctness" is what an
   uninformative layer produces by chance.
4. Run the decisive contrast: the oracle crop on the same items, same budget, same question.
5. Locate the cliff with a sliding threshold, reporting the **whole sweep** rather than one value.
6. **Remove the union-box confound** — multi-region boxes are the union of two objects, so
   tokens-on-target means something different there. Restrict to single-region items.
7. Check the **oracle is flat** across the split; that is what makes it a claim about encoding rather
   than difficulty.
8. Repeat on a second architecture.
