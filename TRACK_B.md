# Track B — visual allocation: what VLMs fail to encode, and why cropping fixes it

**Status: paused pending a prior-art survey.** Track A's core claims turned out to be anticipated
(TwigVLM ICCV'25, LearnPruner ICLR'26, QCTS, ACL Findings'25) *after* five phases were built on them.
**Nothing below should be built on until it has been surveyed the same way.** That is the first task
in this file, not the last.

---

## 1. The thesis

The vision encoder hands the language model a representation that is **resolution-limited** and
**question-agnostic**. Track B is about the first limitation: when the queried object is small, it is
not encoded at all, and no operation downstream can recover it.

This is a claim about **what is in the tokens**, and is distinct from Track A, which is about
**reading attention over tokens that already exist**. They were presented as one "wrong depth"
narrative for part of this project; they are not one narrative and should not be merged again.

---

## 2. What is measured and replicated

### 2.1 The encoding cliff — 2 models ✅
Define `tokens_on_target = area_fraction × total_visual_tokens`.

| model | cliff | uniform below | uniform above | **oracle crop below / above** |
|---|---|---|---|---|
| Qwen3-VL-2B | 0.25 tok | **32.4%** [21,44] | 77.3% [64,89] | **95.8% / 100.0%** |
| Qwen2-VL-7B | 0.25 tok | **36.6%** [25,48] | 63.6% [50,77] | **95.8% / 100.0%** |

On the cleanest subset (localised, single-region, n=63) the step is **19.0% → 78.6%**, and 19.0% is
*below the 25% chance level* of a four-way question.

**The oracle arm is the control that makes this a claim about ENCODING rather than difficulty.** It
is flat across the split on both models at identical values. Items below the cliff are not harder —
given the same 300-token budget spent on the resolved region, both models answer them near-perfectly.
Only the uniform arm steps, and it steps at the same target size in both.

### 2.2 Cropping restores an answer-formation step — 2 models ✅
Reading the answer at every layer, four arms at identical budget, only pixels differ:
separation is **≈0 through L20**, then **+15.3pp at L21**, and stays. The failing arms jump at L2 —
the answer prior forming — and never improve: **the uncropped arm's maximum over all 28 layers
equals its final answer** (56.3% = 56.3%, 50.8% = 50.8%). There is no depth at which the answer was
available and lost.

**The causal control** — same arm, split only by whether its window contained the evidence:

| | Qwen3-VL | Qwen2-VL |
|---|---|---|
| window covers | **+29.7pp** [+19.8,+39.6] | **+23.8pp** [+13.1,+34.5] |
| window misses | −7.9pp [−18.0,+2.2] | **+0.0pp** [−11.2,+11.2] |

⚠ Cross-model claim is *"the gain appears only where the crop delivers"* — **not** "a miss actively
hurts", which held on one model only.

### 2.3 Coverage is the mediator and decides the SIGN — 2 models ✅
| coverage of the evidence set | n | no crop | crop | Δ |
|---|---|---|---|---|
| 0% (missed) | 96 | 52.1% | 36.5% | **−15.6pp** [−26.0,−5.2] |
| 100% (full) | 62 | 58.1% | 95.2% | **+37.1pp** [+24.2,+50.0] |

Crosses zero near 25% coverage; **half of all windows miss**. Within the wrong-at-uniform stratum the
gap reaches **70.8pp** (83.8% covered vs 13.0% missed, below chance). Replicates on Qwen2-VL
(−9.4 / +41.8pp).

### 2.4 The method pays exactly where the deficit is — 1 model
| stratum | n | learned | standard | Δ |
|---|---|---|---|---|
| below the cliff (<0.15) | 58 | 67.2% | 53.4% | **+13.8pp** [+1.7,+25.9] |
| cliff zone (0.15–0.25) | 25 | 64.0% | 48.0% | **+16.0pp** [+4.0,+32.0] |
| above the cliff (≥0.25) | 108 | 70.4% | 66.7% | +3.7pp [−3.7,+11.1] n.s. |

The cliff predicts the method's own operating regime. This also **refutes** a tempting account of the
weak 4K transfer — "below the cliff there is nothing to re-rank" is false by its own sign.

### 2.5 Seven internal interventions, all null — 1 model
Sink suppression · attention amplification (oracle-targeted recovers **16%** of what cropping the
same region buys) · residual-stream steering · layer read-out selection · DoLa/DeCo · component
ablation · **contrastive decoding steered to the ground-truth region** (ceiling **+2.6pp**
[−0.5,+5.8]).

The seventh is the decisive one: it aimed **perfectly**, verified the target is causally live
(masking the evidence region shifts logits **3.32×** more than a random region and costs **10.5pp**),
and still could not convert it.

> The model looks in the right place; the right place is load-bearing; it still cannot answer,
> because no operation on the residual stream can supply information the encoder never wrote.

### 2.6 Localisation is built by the LM, not read off the image — 1 model
Every vision-tower arm is **at or below the 2.3% chance rate** (best single vision layer 1.6%,
learned combination over 24 vision layers 1.0%, mean of all 0.5%) while the LM read-out on the same
items reaches 39.3–45.5%. The vision encoder never sees the question.

### 2.7 Attention quality and answer formation are DISSOCIATED — 1 model
Attention localisation peaks at **L17–19**; the answer forms at **L21**. Rank correlation across
layers only **ρ = +0.297**. Attention top-1 is *higher* in the six layers before the answer forms
(30.1%) than after (24.8%). Raised in external review, tested, confirmed.

---

## 3. The method, and why it is not finished

**Depth-Contrast Re-ranking.** A learned head over the per-layer attention profile replaces the
block-mean argmax; crop at its pick; no extra forward pass.

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| evidence coverage | 39.3 → **52.9%** | 35.1 → **44.0%** |
| vs unmodified model | **+15.0pp** (held-out W) | +10.5pp [+2.6,+18.3] |
| vs random placement | +28.3pp | +20.9pp [+11.5,+30.4] |
| **vs equal-compute baseline** | **+7.7pp [−0.6,+16.0]** ✗ | **+3.1pp [−5.2,+11.0]** ✗ |

**The blocking fact: 0 of 2 models beat simply spending the same compute on a bigger image.** The
lower bound is **−0.6** — about one point short.

**Ceiling context:** the oracle reaches 92.7% at the same budget, and **88.5%** of items have *some*
cell whose window would cover. We capture **27.6%** of that headroom.

### 3.1 Untried levers
1. **Per-item window size from attention-blob extent.** Phase 78 measured **+11.0pp of stable
   headroom** (survives the monotonicity check that killed a similar-looking ceiling in §14A).
   Target size is ruled out as a predictor; blob extent is untried. *Ranking-flavoured, which is the
   distinction separating everything that worked here from everything that failed.*
2. **Retrain the head on post-crop accuracy rather than coverage.** We optimise a proxy for what we
   want; the outcomes are already on disk.

### 3.2 Closed — do not revisit
Budget from a perfect size oracle (loses to flat uniform 11/12 cells) · budget from any free pass-1
signal (9.4% exact-rung vs a 35.6% majority baseline — worse than a constant) · window size from
target size (best W identical across a 100× area range) · attention-space reallocation (16% of the
crop) · rank-only multi-crop (−3.1pp against a predicted +5.5pp; prompting confound excluded) ·
the head's input band (phase 92 was **void** — the head already reads all 28 layers).

---

## 4. FIRST TASK: survey these claims before building anything

Track A's claims were anticipated by work we had not read. The same survey must be run here. Search
specifically for prior art on:

1. **A sharp encoding threshold in target size** below which VLM accuracy falls below chance, with
   an oracle-crop control demonstrating it is encoding rather than difficulty.
2. **Layer-wise answer formation in VLMs** — whether the abrupt single-layer emergence, and the
   split by whether the intervention delivered evidence, is reported.
3. **Coverage of the evidence set as the mediator** that decides the *sign* of a crop's effect.
4. **The dissociation** between where attention localises best and where the answer becomes
   decodable.
5. **Null results for internal interventions** on sub-token visual evidence — especially contrastive
   decoding steered to a ground-truth region.
6. Whether **vision-encoder attention at chance for question-conditioned localisation** is reported.

Suggested starting points from Track A's survey that touch this ground: **FEATHER** (2412.13180,
ICCV'25) on positional bias and localisation collapsing faster than VQA; **ACL Findings 2025**
(2502.11501) §4.2 on RefCOCO grounding under pruning, where random beats every attention method;
**VisPruner** (2412.01818) on positional bias. Also check V\*/SEAL, Zoom Eye, ViCrop, SmartRes,
ViRGo/RAP for the cliff and coverage claims.

---

## 5. The merge with ram

His accessibility / localization / causal-use hierarchy maps onto **Track B**, not Track A — his
selectors are about *where to look*, which is this problem. §2.6 also **qualifies** his Vision-CLS
result: that selector is vision-tower attention, which he reports at box-inside 0.808 on CUB (large,
centred birds) and which we measure **at chance** on V\*Bench (small, query-selected targets). Both
correct; the scope differs, and a merged paper should own that boundary.

---

## 6. Honest framing if the method stays where it is

Not *"a better cropping method"* — it does not beat the trivial baseline. Rather:

> **Why VLMs fail on small objects, seven interventions that do not fix it, and the ceiling nobody
> has reached.**

A findings-and-negatives paper with a strong mechanism section. That is publishable and it is what
the evidence currently supports. It becomes a methods paper only if §3.1 closes the −0.6 gap.
