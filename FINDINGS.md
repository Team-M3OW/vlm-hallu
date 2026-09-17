# VLM Hallucination Project — Consolidated Findings

**Status as of 2026-09-07.** Self-contained handoff: everything needed to resume without prior chat
context. Organized around the intended paper flow: **problem → verify problem → find causality →
find fix.**

---

## 0. Standing constraints (do not violate)

7. **No router.** (2026-09-18) The method must be question-type agnostic in mechanism: no question
   classifier, no keyword rule, no text-only gate, no category oracle — in the method or in how the
   paper frames its headline. Adaptation to question type may come only from the attention map the
   method already computes. Phases 150/153 (routers) are diagnostic context only, never the method.

1. **Public benchmarks only. Do not build datasets.** Everything uses POPE (COCO subset) + COCO
   `val2014` instance annotations, already on disk.
2. **Preprocessing is NOT the headline.** The LLaVA center-crop finding (§5.1) is a footnote.
   User's words: *"no preprocessing as thats not architectural flaw rather just a flawed pipeline."*
   Any preprocessing-specific result stays supporting-evidence, never the architectural claim.
3. **Zoom / attention-guided recropping is prior art** (V*/SEAL, Zoom Eye, Chain-of-Spot, Visual
   CoT, DualFocus, …). Phase 7 (§4.4) is retained as *causal evidence locating the failure*, NOT as
   the paper's contributed fix. Do not pitch recropping as novel.
4. **Report negative results honestly.** Several dramatic-looking early results were killed by
   controls (§6). That discipline is the project's main asset — preserve it.

---

## 1. Problem statement

**Thesis:** VLMs localize well but answer / perceive / classify poorly. Concretely: a VLM will
confidently deny the presence of an object that is plainly in the image *and that the same model
can draw a correct bounding box around.*

Models: `llava-hf/llava-1.5-7b-hf`, `Qwen/Qwen3-VL-2B-Instruct`.
Benchmark: POPE (adversarial / popular / random splits), 5553 items, joined to COCO `val2014`
instance annotations for ground-truth area, bbox, centrality.

---

## 2. Verify the problem

### 2.1 Scaling law: accuracy collapses with object size — and it is NOT a calibration bug
- Forced-choice `P(yes)` degrades monotonically as object area shrinks, in **both** architectures
  (LLaVA-1.5-7B and Qwen3-VL-2B) — cross-architecture, not a single-model quirk.
- **Oracle-threshold ceiling analysis** (the key rigor step): the maximum balanced-accuracy gain
  from a *perfect per-bin* decision threshold over a single global threshold is tiny —
  **LLaVA +0.0162** (`patch_token_frac`), **LLaVA +0.0123** (`pixel_area_frac`), **Qwen +0.0098**.
  ⇒ This is a real evidence-integration failure, not fixable by recalibrating a threshold.

### 2.2 Centrality bias, independent of size, in a model that never crops
- Area-weighted centroid distance from image center predicts accuracy, surviving **category fixed
  effects** and an **aspect-ratio covariate**. Bootstrap 95% CI **[-0.227, -0.092]**.
- Critically it **survives in Qwen3-VL**, which performs *no* destructive center-cropping ⇒ rules
  out "it's just LLaVA's preprocessing" as the whole story.

### 2.3 The dissociation itself (Phase 4) — the core verification
`data/phase4_localize_results.jsonl` (486 rows). Cohort: items where Qwen3-VL confidently denies
(`P(yes) < 0.01`) an object that IS present, plus a matched control of items where the object is
genuinely ABSENT and also confidently denied.

| group | n | emits a box | mean best IoU | IoU > 0.5 |
|---|---|---|---|---|
| confident_denial (object present) | 243 | **95.1%** | 0.368 | **43.2%** |
| negative_control (object absent) | 243 | 45.3% | 0.000 | 0.000 |

⇒ On the exact items where its verbal judgment is confidently wrong, the model still localizes the
object correctly 43.2% of the time. **Localization and verbalized judgment are dissociable in the
same model, on the same input, in the same forward-pass budget.**

*Caveat:* IoU is circular for the negative arm (absent objects have no GT box, so IoU≡0). The
**non-circular** signal is box-*emission* (95.1% vs 45.3%) — see §5 for why this matters a lot.

---

## 3. Find causality — what the mechanism is

### 3.1 Phase 6c: a controlled attention-routing signature ✅ (query-conditional, but see the mass-vs-rank caveat below)
Measure: **enrichment** = (attention mass on the queried object's own image tokens / #object tokens)
÷ (attention mass on all image tokens / #image tokens). 1.0 = no preference. Read from the last
input position (where yes/no is emitted), averaged over heads and the last 4 layers.

Three query conditions, **same image, same object-X token set held fixed**, only the *question*
changes (n=271 3-way joined: 123 confident_denial, 148 confident_correct_small):

| group | query = X (own object) | query = absent Y | query = present-other Z |
|---|---|---|---|
| confident_denial | 0.730 (ans: NO) | 0.336 (ans: NO) | 0.592 (ans: YES) |
| confident_correct_small | **3.145** (ans: YES) | 0.576 (ans: NO) | **0.713** (ans: YES) |

**The tautology-killer** (a reviewer's first objection — "of course attention points at evidence
when about to say yes"): within `confident_correct_small` alone, **polarity held fixed at YES**,
attention on X collapses 3.145 → 0.713 (**~4.4×**) purely because the *question* moved from X to Z.
⇒ Attention genuinely tracks the **query target**, not the answer polarity.

> **⚠️ Mass-vs-rank caveat (added after Phase 22; this DEMOTES 6c).** Everything above measures
> attention **mass** enrichment. Phase 22 measured attention **rank** — whether the target's tokens
> survive top-k retention — and found the target is retained **+20 to +35pp above chance** at
> L2/L4/L8. These are not contradictory: mass on the target can be sub-proportional (enrichment
> below 1.0) while the target still sits near the top of the ranking. But it means 6c **cannot**
> support the stronger reading it was previously used for — that attention fails to find the
> object, or that "there is nothing at that location to attend to." 6c shows the attention
> distribution is **query-conditional**; it does not show the target goes unattended. The label
> "strongest mechanistic result" has been withdrawn accordingly (bug #19).

**Given that, the finding:** confident_denial's attention on X *under its own query* (0.730) is
statistically indistinguishable from correct items' attention on X *when X isn't being asked about
at all* (0.713). X/Z target-modulation index: denial **[0.93, 1.28]** (includes 1.0 — sharply
reduced target-conditioning), correct **[3.65, 5.39]**; difference **[2.52, 4.30]**, 100% of 3000
resamples favor correct.

> **When Qwen3-VL is about to confidently deny an object that is present, its attention to that
> object's own location looks like attention to an object nobody asked about.**

*Phrasing discipline:* say **"sharply reduced target-conditioning,"** not "flat/invariant" — the
absent-Y condition (0.336) does differ from X and Z, so denial-group attention is not literally
query-invariant.

### 3.2 Three decoder-side causal hypotheses — all eliminated ❌
Each was a genuine attempt at a *fix*, each properly controlled, each negative. Collectively they
are what licenses the upstream conclusion in §3.3.

| phase | hypothesis | intervention | result |
|---|---|---|---|
| **5** | "detected then suppressed" in late layers | logit-lens readout at layers 15/18/20/22/25/28 | **Falsified.** Final layer AUROC **0.956** beats every intermediate layer on an unbiased 500-pos/500-neg sample; the apparent mid-layer gain was a calibration coincidence; per-layer oracle ceilings statistically tied (~0.895–0.897). |
| **6d** | too few autoregressive steps (grounding gets ~15–20 forward passes, yes/no gets 1) | force extra decode steps before the answer: image-grounded CoT vs. **token-count-matched content-free filler** | **Refuted.** Both raise P(yes) a lot (median 0.00019 → cot 0.056 / filler 0.119) but **filler beats CoT on 108/141 items, p=8.4e-11**, and object-token attention enrichment *drops* under both (0.848 → 0.604/0.643, CI on drop [0.154, 0.372]). ⇒ generic template/distribution-shift effect, not re-perception. Confident denials are **fragile to task-irrelevant prompt perturbation**. |
| **6e** | attention allocation is the causal bottleneck | monkeypatch `Qwen3VLTextAttention.forward`; redistribute post-softmax attention at the last position so object tokens get a target share F | **Null / inconclusive.** At single-layer, realistic F (0.094 = correct items' *natural* median share; and 0.20): only **6/141** items improved; object-target vs **random-target** indistinguishable (58/141 and 67/141; CIs [-0.00004, 0.00003] and [-0.00009, 0.00004]). |

**6e phrasing discipline:** do **not** write "attention causality refuted." The object-vs-random
symmetry means the manipulation isn't discriminating between targets at all — a statement about the
*tool*, not the mechanism. Correct line: *we could not construct an intervention that tests this
cleanly.* It does not undo §3.1, which compared **naturally occurring** attention.

### 3.3 Phase 7: the locus is upstream — evidence is present but under-resolved ✅
Give the model a second image: a crop of the **same photograph** around the object's own region
(COCO bbox + 25% pad), which the processor upscales so the region gets many more patches than it
did embedded in the full scene. **Zero new information** — same pixels. n=141.

| condition | median P(yes) | mean P(yes) | flips to "yes" |
|---|---|---|---|
| baseline | 0.000185 | 0.00143 | 0% |
| **oracle_zoom** (true object's region) | 0.0078 | 0.162 | **12.8%** (18/141) |
| random_zoom (size-matched other region) | 0.00047 | 0.0103 | 0% |

oracle > random on **105/141** (one-sided binomial **p=2.5e-9**); CI on mean difference
**[0.104, 0.202]**; CI on flip-rate difference **[0.078, 0.184]**. **Dose-response:** items that
flipped are *smaller* (median area frac 0.00145) than items that didn't (0.00401) — exactly what
resolution-starvation predicts.

**Two mandatory caveats:**
1. The clean comparison is **oracle-vs-random** (both carry the identical connector sentence and
   two-image structure). `oracle > baseline` (111/141) is contaminated by the template-shift effect
   independently measured in 6d — do not headline it.
2. This is an **oracle demonstration, not a fix**: the crop location comes from COCO ground truth.
   The honest claim is *the evidence is present in the original pixels but under-resolved at the
   patch grid the model actually processes.* Combined with §3.2, this **locates the failure
   upstream of the decoder** — in visual representation/resolution, not attention, layer choice, or
   decoding pace.

> ⚠️ **AMENDED 2026-09-07 (Phase 9/10).** The 12.8% figure **understates the resolution effect by
> ~3×**, and the cause is Phase 7's own two-image format. Handing the model the crop **alone**
> recovers **40.4%** (FP on absent objects 3.5%), vs **12.8%** for `[full + connector + crop]`.
> Paired diff **+27.7pp, CI [+20.6, +34.8], 100% of resamples**. Phase 7 therefore measured
> "resolution + a distractor" and reported it as "resolution". The correct reading of Phase 7 is
> that under-resolution is a **substantially larger** cause than this section claimed — and the
> shortfall it accidentally measured is itself the Phase 10 finding (§4A). See §6 row 14.

---

## 4. Find fix — OPEN. This is where the next session starts.

Zoom/recrop is prior art and is therefore **not** the contribution. The fix should follow from
*this project's own thesis* rather than from the zoom literature.

### 4.1 Lead candidate: answer-by-grounding (use the localization channel as the presence detector)
The dissociation says the localization channel is more reliable than the yes/no channel. So *read
the answer off the localization channel.* **The signal is already visible in data on disk:**

On the Phase 4 cohort — where **both** groups confidently answer "no", so the yes/no readout is at
**0.500 balanced accuracy by construction** — box-emission alone gives:

| detector | sensitivity | specificity | balanced accuracy |
|---|---|---|---|
| yes/no readout `P(yes)` | — | — | **0.500** (by construction) |
| "does it emit a box?" | 0.951 | 0.547 | **0.749** [95% CI 0.716–0.782] |

> ❌ **REFUTED 2026-09-07 by Phase 8** (`phase8_grounding_detector.py`, n=1000, broad unselected
> 500 pos / 500 neg, `random.Random(7)`, fp16, **matched compute — one forward pass each**).
> The 0.749 was **entirely a selection artifact**: that cohort was chosen for `P(yes)<0.01`, i.e.
> chosen for the yes/no channel failing.
>
> | score (1 forward pass) | AUROC |
> |---|---|
> | `P(yes)` | **0.9725** |
> | `p_ground` (grounding channel, first-token box-vs-abstain readout) | 0.9635 |
>
> Paired ΔAUROC **−0.0090, CI [−0.0163, −0.0018]**, only **0.5%** of resamples favor grounding.
> Grounding is **worse**, not better. Combining the two channels gives +0.002 (CI spans 0).
>
> **Why it fails — the mechanism:** on `P(yes)`'s 42 false negatives `p_ground` says "box" 90.5% of
> the time (looks like recovery), but on its 35 false **positives** it says "box" **100%** of the
> time. It asserts presence on nearly everything, so it recovers misses for free. Confirmed by the
> negative arms everywhere: box emission on genuinely absent objects is **46.1%** (full image),
> **44.0%** (two-image), **75.9%** (crop), and Phase 8's blank-image readout is **0.74–0.99 on
> black images**.
>
> ⇒ **The grounding channel is not a complementary evidence channel. It is a yes-biased channel.**
> Any "grounding recovers denials" figure based on *emission* is uninterpretable; only *IoU*
> (correct localization, which a yes-bias cannot fake) survives.
>
> **Clean controls that did hold:** blank-image AUROC floors are **0.407** (`P(yes)`) and **0.356**
> (`p_ground`) — both **below chance**, so neither channel rides a POPE category-frequency prior.
> The co-occurrence-prior objection is dead. Per-split deltas are uniform (random/popular/adversarial).

### 4.2 Secondary candidate: grounding↔yes/no disagreement as a detector
> ❌ **DEAD with §4.1.** The channels do not fail independently — grounding says "present" almost
> always, so disagreement is dominated by grounding's yes-bias. Channel combination: +0.002 AUROC,
> CI spans zero.

### 4.3 Ruled out — do not re-run
Layer-readout selection (§3.2 Phase 5); extra decode steps / CoT (Phase 6d); attention patching as
implemented (Phase 6e); anything framed as recropping/zoom being novel (Phase 7 + prior art);
**answer-by-grounding as a detector, and grounding↔yes/no disagreement (§4.1, §4.2, Phase 8)**.

---

## 4A. Phase 10 — visual context interference ✅ (the live lead, 2026-09-07)

**The observation.** Phase 9's factorial found the resolution intervention is worth **40.4%**
recovery when the crop is given **alone**, but only **12.8%** when the *same crop* is given
alongside the full scene. Phase 10 (`phase10_context_interference.py`, n=141 positives + 141
negatives, all arms **paired on the same stored crop box**) varies the three confounded factors
one at a time. Every recovery number carries its false-positive rate on genuinely absent objects.

| arm | recovery | FP | discrimination |
|---|---|---|---|
| full image only (baseline) | 0.0% | 0.0% | +0.0 |
| **crop only** | **40.4%** | 3.5% | **+36.9** |
| crop + connector text | 48.9% | 4.3% | +44.7 |
| **[crop + crop]** — image count held at 2 | **38.3%** | 3.5% | **+34.8** |
| [black + crop] | 34.0% | 2.1% | +31.9 |
| [**other** image's scene + crop] | 29.8% | **19.1%** | +10.6 |
| **[full + crop]** — Phase 7 replication | **12.8%** | 0.0% | **+12.8** |
| [full + crop], no connector | 17.0% | 0.7% | +16.3 |
| [crop + full] — order swapped | 4.3% | 0.0% | +4.3 |

**Finding.** *The suppression is caused by the scene content itself, not by prompt format.*
Holding image count at two, adding the full scene costs **+25.5pp raw** and **+14.2 to +29.8pp on
discrimination**. Exonerated as causes: **image count** (`[crop+crop]` 38.3% ≈ crop alone 40.4%),
**prompt format**, and the **connector sentence** (which if anything *helps*: 48.9%).

> **A high-resolution crop that lets the model answer correctly 40% of the time becomes nearly
> useless (13%) when the full scene sits in the same prompt. Same pixels, same crop, same model.**

**Secondary: strong recency.** Full image *last* (`[crop + full]`, 4.3%) is worse than full image
*first* (`[full + crop]`, 12.8%): **−8.5pp, CI [−12.8, −4.3]**. The final image dominates.

**Pre-registered hypothesis REFUTED — "re-anchoring on the model's own failed percept."**
On raw recovery, the other-scene arm (29.8%) beat the own-scene arm (12.8%) by **+17.0pp, CI
[+11.3, +23.4], 100% of resamples** — which reads as clean evidence for re-anchoring. It is not:
that arm carries **FP 19.1% against ~0–4% everywhere else**, i.e. it is simply more yes-biased. On
**discrimination the CI is [−10.6, +6.4], spanning zero.** ⇒ **Any full scene interferes about
equally. Generic visual context interference, not re-anchoring.** (`phase10_analyze.py` now renders
verdicts only on discrimination, never raw recovery — see §6 row 15.)

**Required before this is framed as a contribution:** a **prior-art check** on multi-image
distraction / positional bias in VLMs (standing constraint #3 — check prior art *before* pitching).

### 4A.1 Open questions
- Does it hold on a **fine-grained** task (CUB-200-2011, downloaded to `data/cub/`)? CUB birds fill
  much of the frame, which separates **object size** from **fine detail resolution** — POPE
  conflates them.
- Cross-architecture (`Qwen2-VL-7B`, `InternVL2-8B`, `llava-onevision-7b` all cached).

---

## 4B. Phase 11 — RePOPE audit ✅✅ (2026-09-07). **Read this before trusting any cohort number.**

`phase11_repope_audit.py`, CPU-only. RePOPE (arXiv 2504.15707, `github.com/YanNeu/RePOPE`)
re-annotated POPE's 500 COCO images with dual independent three-way labels (yes/no/ambiguous).
Joined on `(split, question_id)`; **0 missing**.

### 4B.1 Confident model disagreement is a 7.2× enriched detector of benchmark label error

| cohort | mislabeled (RePOPE says absent) | ambiguous | **not clean** |
|---|---|---|---|
| all POPE positives we evaluate (n=4053) | **4.3%** [3.7, 5.0] | 12.1% | 16.4% |
| Phase 8 **broad unselected** positives (n=500) | 4.8% [3.2, 7.0] | 12.2% | 17.0% |
| **confident_denial cohort, P(yes)<0.01** (n=243) | **30.9%** [25.4, 36.9] | 27.2% | **58.1%** |
| Phase 7/9/10 crop cohort (n=141) | **40.4%** [32.7, 48.7] | 21.3% | **61.7%** |

**Enrichment factor 7.2×** over the base rate. The broad sample sits exactly at base rate, so this
is selection, not a join artifact.

> **When Qwen3-VL-2B confidently contradicts POPE, POPE is wrong ~31% of the time and ambiguous a
> further ~27%. Nearly 60% of the canonical "confident hallucination" cohort is not a model error
> at all.**

Two consequences:
1. **Methodological, for the whole field.** Any paper that studies VLM hallucination by selecting
   high-confidence errors on POPE is studying a cohort that is majority label noise. This *extends*
   RePOPE (which found the errors by manual re-annotation) into an automatic detector.
2. **For us.** Every cohort-based number in §2.3, §3, §4A must be reported RePOPE-clean. Fortunately
   this makes the findings **stronger**, not weaker (§4B.2).

### 4B.2 On correctly-labeled items the findings get substantially STRONGER

**Dissociation (Phase 4)** — essentially unchanged, so it was never label noise:
emits box 95.1% → 94.1%; IoU>0.5 43.2% → **41.2%** (n 243 → 102 clean).

**Context interference (Phase 10)** — the effect roughly doubles:

| arm | all items | **RePOPE-clean** (n=54) |
|---|---|---|
| crop alone | 40.4% | **72.2%** |
| [crop + crop] | 38.3% | **72.2%** |
| [full + crop] | 12.8% | **22.2%** |
| [crop + full] | 4.3% | 5.6% |
| [other scene + crop] | 29.8% | 42.6% |

> **THE GAP on correctly-labeled items: 72.2% vs 22.2% — +50.0pp, CI [+37.0, +63.0], 100% of
> resamples.** Image count still exonerated: `[crop+crop]` 72.2% is identical to crop-alone, so the
> cause remains **scene content** (same CI vs `[full+crop]`).

⇒ The model **can** perceive the object in ~72% of genuine confident denials. The full scene in the
same prompt destroys about two-thirds of that.

**Label noise was diluting the effect**, because mislabeled items (object truly absent) can never
be "recovered" by any arm and were being counted as failures.

### 4B.3 The residual is NOT primarily label noise
Recovered group 38.5% flipped vs residual group 42.1% flipped — indistinguishable. So "no
intervention recovers it" does **not** mostly mean "the object was never there". On clean items the
residual shrinks to ~28% (crop-alone recovers 72.2%), which is the honest open question.

---

## 4C. Phase 13 — breadth-first sweep of NON-CROP interventions ✅✅ (2026-09-07)

`phase13_bfs_interventions.py` / `phase13_analyze.py`. 11 arms, one forward pass each, n=141 (+141
negatives for FP). **Gate passed:** the upscale arms genuinely raised visual-token count on
**141/141** items (median 300 → 1200 → 2700), so FINDINGS bug #5 did not recur.

**RePOPE-clean results (n=54 — the trustworthy numbers), with tokens-on-object computed from the
logged `image_grid_thw` and the COCO bbox:**

| arm | family | tokens on object | recovery | FP | disc |
|---|---|---|---|---|---|
| baseline | reference | **0.4** | 0.0% | 0.0% | +0.0 |
| `dim_outside` | scene removed, res const | 0.4 | 0.0% | 1.4% | −1.4 |
| `gray_outside` | scene removed, res const | 0.4 | 16.7% | 1.4% | +15.2 |
| `blur_outside` | scene removed, res const | 0.4 | 16.7% | **12.3%** | +4.3 |
| **`black_outside`** | **scene removed, res const** | **0.4** | **27.8%** | 2.2% | **+25.6** |
| `upscale2x` | **res raised, scene KEPT** | 1.8 | 27.8% | 1.4% | +26.3 |
| **`upscale3x`** | **res raised, scene KEPT** | **4.1** | **50.0%** | 0.7% | **+49.3** |
| `redbox` | pointing only | 0.4 | 5.6% | 0.7% | +4.8 |
| `prompt_focus` | pointing only | 0.4 | **0.0%** | 0.0% | +0.0 |
| `prompt_coords` | pointing only | 0.4 | **0.0%** | 0.0% | +0.0 |
| `crop_alone` | both | **32.0** | 72.2% | 0.7% | +71.5 |

### Three findings

**1. The median confident-denial object occupies 0.4 merged visual tokens — less than ONE token.**
These objects are *sub-token*. That single number reframes the whole phenomenon.

**2. Two independent, additive mechanisms — and resolution is the larger one.**
- *Token budget*, monotone dose-response **with the scene fully retained**:
  0.4 tok → 0.0% · 1.8 tok → 27.8% · 4.1 tok → 50.0% · 32 tok → 72.2%.
- *Scene interference*, at **identical** 0.4-token budget: baseline 0.0% → `black_outside` 27.8%.

  This **corrects §4A's emphasis**: the interference effect is real (0% → 27.8% with zero extra
  pixels) but the resolution effect is larger. Neither alone reaches crop's 72.2%.

**3. Pointing does nothing. ⭐ (the surprising one)**
`redbox` 5.6%, `prompt_focus` 0.0%, `prompt_coords` 0.0% — telling the model *exactly* where to
look, either by drawing a box on the pixels or by giving normalized coordinates in the prompt,
recovers **essentially nothing**. This converges with **Phase 6e** (attention patching: null) and
**Phase 6c** (attention target-conditioning collapse).

> **The failure is not "the model looks in the wrong place." You cannot point a VLM at what it has
> not encoded — you can only spend more tokens on it, or remove what competes with it.**

> **⚠️ Corrected after Phase 22.** This block originally continued: *"the object occupies less than
> one visual token, so there is nothing at that location to attend to."* **That sentence is wrong
> and has been removed.** Phase 22 tested it directly and refuted it: attention-guided top-k
> retention keeps the target **+20 to +35pp above chance** at L2/L4/L8, so the target *is* attended
> — it is findable in the attention ranking. The surviving explanation for why pointing does nothing
> is therefore **not** absence of attention but insufficiency of what is encoded *at* the attended
> location: the model attends to the right tokens and those tokens do not carry the detail. This was
> a prediction of mine that the experiment killed; see §4J and bug #19.

**Practical corollary worth its own experiment:** `upscale3x` needs **no bounding box**. Every crop
method (V*/SEAL, CropVLM, Chain-of-Spot) requires *predicting where to crop*; naive whole-image
upscaling recovers **50%** — about 70% of crop's benefit — with no localizer at all.

*Caveats:* n=54 clean (§4B.4 rebuild fixes this); `blur_outside` carries FP 12.3%, so it is
bias-shifted — trust `black_outside`/`gray_outside`.

---

## 4D. Phase 14/15 — is anything decodable inside the model? ✅ (2026-09-07)

The gate for every internal (activation/attention/gradient) intervention, since the user ruled out
image preprocessing. `phase14_probe_gate.py` + `phase15_category_specificity.py`, RePOPE-clean.

**Phase 14** (n=1400): held-out linear probe on the object's own visual tokens reaches **0.88 AUROC
in the sub-token stratum** (<1 merged token), vs **0.59** for a paired random region in the same
image. Zero-coverage fallback never fired. Last-position probe 0.974 vs `P(yes)` 0.916 in that
stratum — a **readout gap**. ⇒ There is **no capacity floor**; the prediction from §4C (0.4 tokens,
pointing at 0.0%, 6e null) was **wrong**.

**Phase 15** — the control that makes 14 interpretable. Phase 14 compared *true bbox* (positives)
against *random region* (negatives), so it could be reading **objectness**, not the queried
category. Phase 15 probes **within positives, paired inside one image and one forward pass**:
queried-category region vs a **size-matched other-category** region (n=893).

| pairs | L10 | L16 | L22 | L27 |
|---|---|---|---|---|
| all (median size ratio 2.2×) | 0.588 | 0.539 | 0.629 | 0.627 |
| **well size-matched** (\|log2 ratio\|<0.5, n=275) | 0.629 | **0.709** [0.637, 0.778] | 0.689 | 0.671 |

**Verdict: category-specific encoding is REAL but MODEST.** Most of Phase 14's 0.88 was objectness;
what survives a same-image, size-matched, other-category control is **~0.71 at L16**, CI excluding
chance. Size mismatch was *diluting* the signal, not creating it (well-matched pairs score higher).

⇒ Internal steering has a genuine target, but a **faint** one. Expect modest effects, and treat
Phase 6e's null as the realistic prior. Derivation layer should be **L16–L22**, not the last layer.

---

## 4E. Phase 16 — activation steering at L16 ❌ NULL (well-controlled) (2026-09-07)

`phase16_steering.py`. Direction v = mean(queried-category tokens) − mean(size-matched
other-category tokens) at L16 (where Phase 15 put the category signal), derived on items **disjoint**
from the eval set, injected via a forward hook on `layers[15]` as `α·‖h‖·v̂`.
Cohort: **54 RePOPE-clean fp16 confident denials** + 220 clean negatives. α ∈ {0.25, 0.5, 1, 2}.

| arm | α=0.25 | α=0.5 | α=1.0 | α=2.0 |
|---|---|---|---|---|
| `steer_obj` | 0.0/4.1 | 0.0/4.1 | 0.0/4.1 | 0.0/4.1 |
| `rand_obj` (norm-matched control) | 0.0/4.1 | 0.0/4.1 | 0.0/4.1 | 0.0/4.1 |
| `steer_all_img` | 0.0/4.1 | 0.0/3.2 | 0.0/2.7 | 0.0/2.3 |
| `steer_last` | 0.0/3.6 | 0.0/3.6 | 0.0/3.2 | **55.6/26.4** |
| `rand_last` (control) | 0.0/4.1 | 0.0/4.1 | 0.0/11.4 | **100.0/78.2** |

*(recovery% on denials / false-positive% on true absences)*

**Verdict: NULL.** `steer_obj` vs `rand_obj` diff-in-discrimination CI **[+0.0, +0.0] pp at every α**.

**Not a bug — verified.** Identical binarized outcomes with a [0,0] CI is a no-op signature, so the
hook was checked directly: raw P(yes) *does* move (1.245e-03 → 1.226e-03 etc.), and differs between
the real and random directions. The intervention fires; it is simply far too weak.

**Why, quantitatively — this is the useful part.** Baseline P(yes) on these items is ~1e-4 to 1e-3.
Flipping to >0.5 needs a **3–4 order-of-magnitude** odds change. The median denial has
**2.5 object tokens**; perturbing 2–3 positions out of ~1500 at one layer moves P(yes) by **~2%**.
The lever is ~3 orders of magnitude short of the gap. Steering **all** image tokens moves it more
(5.4e-07 → 9.2e-06) — still short, and it pushes *toward* "no" (FP 4.1% → 2.3%).

**The controls earned their keep.** `steer_last` at α=2.0 looks like a triumph (55.6% recovery) —
but the **norm-matched random direction at the same position does it harder** (100% recovery,
78.2% FP). Pure bias shift: at high α the last position is simply being pushed to "yes"
irrespective of direction. Without `rand_last` this would have been a false headline (bugs #7/#10,
#15 — the fourth instance of this trap in the project).

**Reading, consistent with Phase 6e:** this is a statement about the *intervention*, not proof the
mechanism is absent. Phase 15's L16 category signal (AUROC ~0.71) is real but faint, and a single
mean-difference direction is too blunt to exploit it against a 1000× odds gap.

⇒ **Internal inference-time interventions now have two independent, well-controlled nulls
(6e, 16).** Combined with §4C (pointing recovers 0.0%) and §4D (the signal exists but is weak), the
defensible position is that the confident-denial failure is **not correctable at inference time by
redistributing or nudging activations** — only by changing what the encoder spends tokens on.

---

## 4F. Phase 17 (E1) — budget-matched query-conditional allocation ✅✅✅ **THE CENTERPIECE**

`phase17_budget_allocation.py` / `phase17_analyze.py`. n=54 RePOPE-clean fp16 confident denials +
160 clean negatives. **All arms matched on REALIZED merged tokens** (`image_grid_thw`), verified:
spread 0.7% at B=300, 5.6% at B=150, 2.2% at B=600.

**Question:** "more tokens helps" is not a finding (AnyRes / dynamic-resolution already assert it).
At a **fixed total budget**, does allocating **by query** beat allocating **uniformly**?

| budget | `uniform` | **`alloc_query`** | `alloc_random` (decider) | `crop_only` (upper bd) |
|---|---|---|---|---|
| **B=150** | 5.6% | **50.0%** | 18.5% | 77.8% |
| **B=300** | 11.1% | **38.9%** | 14.8% | 83.3% |
| **B=600** | 16.7% | **50.0%** | 14.8% | 66.7% |

False-positive rates 0.0–4.4% everywhere ⇒ **no bias shift** (contrast Phase 16's `rand_last`).

**The decider — `alloc_query` vs a size-matched RANDOM region at identical realized budget:**

| budget | diff | 95% CI | resamples favoring |
|---|---|---|---|
| B=150 | **+31.5pp** | [+18.5, +44.4] | 100% |
| B=300 | **+24.1pp** | [+13.0, +35.2] | 100% |
| B=600 | **+35.2pp** | [+22.2, +48.1] | 100% |

Significant at **every** budget. So the win is **allocation**, not merely "non-uniform layouts help"
— which is exactly the alternative that `alloc_random` was pre-registered to rule out (Phase 16's
`rand_obj` lesson moved to the input level).

### The headline number

> **Query-conditional allocation at 150 tokens (50.0%) beats uniform allocation at 589 tokens
> (16.7%).** Roughly **4× fewer visual tokens, ~3× the recovery.** Uniform allocation cannot be
> fixed by spending more — it is spending in the wrong places.

⇒ This converts the project's four nulls (§4C pointing 0.0%, §4E steering, 6e attention patching)
from "things that failed" into **"the allocator is upstream of all of them."** You cannot redirect
attention to capacity that was never allocated — but you *can* allocate it differently, at no extra
cost.

**Caveats to keep attached:** (1) **oracle** region, from COCO GT, exactly like Phase 7 — legitimate
for a mechanism claim, and what keeps this out of re-implementing V*/SEAL's localizer; a practical
system needs a query-conditional proposer. (2) `crop_only` is higher still but **discards the
scene**, so it is not a viable general allocator (and it is non-monotonic: 83.3% at B=300 vs 66.7%
at B=600). (3) n=54.

**Two bugs caught in v1, both of which would have voided the experiment (§6 rows 17–18).**

---

## 4G. Phase 18 (E2a) — CUB-200-2011: the tail does NOT transfer, and that BOUNDS the claim

`phase18_cub_tail.py`. n=997, CUB **test split only**, published labels. Same prompt/readout as
everywhere. Three questions per image: true species, **same-genus** confusable species (hard
negative), random different-genus species (easy negative).

**Geometry first — CUB is the size/detail separator we wanted:**
median bbox area fraction **0.306** and **53.8 tokens on object** (POPE confident denials: ~0.0015
and ~0.16). CUB birds are *large*. Any failure here cannot be "the object is too small".

| measure | CUB | POPE (reference) |
|---|---|---|
| AUROC true vs **easy** negative (diff genus) | 0.9618 | 0.9725 |
| **AUROC true vs HARD negative (same genus)** | **0.6716** | — |
| confident-denial tail, P(yes)<0.01 on true species | **2.2%** | 1.6% (RePOPE-clean) |
| **false-positive rate on same-genus negatives** | **70.7%** | — |
| false-positive rate on different-genus negatives | 9.0% | — |

### Two findings, one of which is against us

**1. The confident-denial tail does NOT get bigger (2.2% vs 1.6%).** So fine-grained difficulty does
*not* inflate the phenomenon, and CUB does **not** rescue the "1.6% is a thin base" problem. The
plan's premise for running CUB (§3 of PAPER_PLAN v2) was wrong.

**2. The fine-grained failure is a DIFFERENT failure, of the opposite polarity, and it is huge.**
The model does not deny birds — it **over-accepts** them: it says "yes" to a *wrong same-genus
species* **70.7%** of the time, giving near-useless discrimination (AUROC 0.672). And it does this
with **53.8 tokens on the object** — a generous budget.

⇒ **This bounds the thesis, which is more useful than a replication would have been.** The
token-allocation account (§4C, §4F) explains **confident denial of under-resolved small objects**.
It does **not** explain fine-grained over-acceptance, where capacity is ample and the failure runs
the other way. The paper must say so explicitly rather than implying a general theory of VLM
hallucination — otherwise a reviewer finds this in one experiment.

### 4G.1 The one CUB experiment still worth running
The bird is large, but the *discriminative evidence* (beak shape, wing bars, eye ring) may itself be
sub-token. So: does **budget-matched allocation** (Phase 17's design) to the bird's bbox — or to CUB
`parts/` annotations — improve same-genus AUROC above 0.672? If yes, allocation generalizes from
*object size* to *diagnostic detail*, which is a materially stronger claim. If no, the claim is
specifically about object size, and that is the honest scope.

---

## 4H. Phase 20 — V*Bench: the allocation result in a regime where it is NOT a tail ✅✅✅

`phase20_vstar_allocation.py` / `phase20_analyze.py`. **n=191 (full benchmark)**, median image
2250×1500, **median target area fraction 0.001081** — comparable to, and often smaller than, our
POPE confident denials. Categories: 115 `direct_attributes`, 76 `relative_position`.

**Why this benchmark:** POPE's phenomenon is 1.6% of positives and CUB failed to enlarge it (2.2%).
On V*Bench sub-token targets are the **design**, so the effect base is ~100% of items rather than a
tail. Budget match verified on realized tokens: spread **1.7–2.7%** at all three budgets.

**Readout mapping** (fixed before coding, per the CUB lesson): 4-way MCQ → softmax over `{A,B,C,D}`
logits at the final position; metric = **accuracy**; all 191 items, no confidence selection.
A false-positive arm is **not needed here** — 4-way argmax is intrinsically immune to the
bias-shift trap that caught Phase 16's `rand_last` (uniformly inflating one option cannot raise
argmax accuracy). `alloc_random` remains the decider.

| budget | `uniform` | **`alloc_query`** | `alloc_random` | `crop_only` |
|---|---|---|---|---|
| B=300 | 56.5% | **85.9%** | 48.7% | 93.7% |
| B=600 | 66.0% | **90.1%** | 60.2% | 93.7% |
| B=1200 | 70.2% | **94.2%** | 63.9% | — |

**The decider — `alloc_query` vs a size-matched RANDOM region at identical realized budget:**
B=600 **+29.8pp** [+22.5, +37.2]; B=1200 **+30.4pp** [+23.0, +37.7]; 100% of resamples at every
budget. (POPE gave +24 to +35pp — the effect is at least as large here, on ~100% of items.)

### The headline
> **Query-placed allocation at 292 tokens (85.9%) beats uniform allocation at 1176 tokens (70.2%)**
> — **4× fewer visual tokens, +15.7pp**, CI [+7.9, +23.6]. Also 610 tok (90.1%) > 1176 tok (70.2%).

### On items uniform allocation gets WRONG
`alloc_query` recovers **86.2–93.0%**; `alloc_random` recovers **15.7–19.3%**. That contrast is the
thesis in one line: it is not *that* you allocate non-uniformly, it is **where**.

### Unexpected, and practically important
At B=1200, `alloc_query` (94.2%) **matches or exceeds** `crop_only` (contrast −2.6pp, CI spans 0);
at B=300 `crop_only` is still ahead (+7.8pp). ⇒ **At adequate budget you do not have to discard the
scene to get the benefit.** This matters because crop-only is not a viable general allocator (it
throws away context other questions need) — and now it does not have to be.

### Positioning vs V*/SEAL
V*Bench was built by the V*/SEAL authors to motivate their crop method, so "cropping helps here" is
already their result. Our claim is different and rests on two things they do not do: (1) the
**matched realized budget**, and (2) the **size-matched random-placement control**. Allocation, not
cropping, and not extra tokens.

---

## 4I. Phase 21 — where does allocation stop paying? (the crossing-point experiment)

`phase21_crossing_point.py`. RePOPE-clean positives **stratified by design** on `tokens_on_object`
(range 0.08–290), 120 per stratum, **all confidence levels** (a negative delta is unobservable on
items uniform already gets wrong). 200 clean negatives for FP. B=300, realized 294–299 (**1.7%
spread**).

| stratum | med tokens | uniform | **alloc_query** | alloc_query25 | alloc_random | Δ(q − uniform) |
|---|---|---|---|---|---|---|
| [0, 0.5) | 0.26 | 57.5% | **74.2%** | **80.0%** | 59.2% | **[+7.5, +25.8]** ✅ |
| [0.5, 2) | 1.03 | 89.2% | **95.8%** | 92.5% | 82.5% | **[+0.8, +12.5]** ✅ |
| [2, 8) | 4.19 | 98.3% | 100.0% | 100.0% | 95.0% | [+0.0, +4.2] |
| [8, 32) | 18.1 | 97.5% | 98.3% | 98.3% | 95.8% | [+0.0, +2.5] |
| [32, ∞) | 104.9 | 99.2% | 98.3% | 98.3% | 98.3% | [−3.3, +1.7] |

False-positive rate is **flat at 7.0–7.5% across all four arms** ⇒ no bias shift anywhere.

### Findings
1. **The benefit is confined to the token-scarce regime.** Significant below ~2 tokens on object,
   statistically indistinguishable from uniform above ~2, and *nominally negative* (−0.8pp) above
   32 tokens. This converts §4C/§4F from a claim into a **law with a domain**.
2. **The predicted harm did NOT clearly appear** — and the honest reason is a **ceiling artifact**:
   uniform already scores 97–99% on large objects, leaving no headroom for a negative delta to be
   detected. The pre-registered prediction (allocation halves scene resolution to fund a crop it
   does not need) is therefore **untested, not refuted**. Testing it needs a hard task with large
   targets, which POPE does not provide.
3. **The 50/50 split is not optimal.** Giving the crop only 25% of budget (`alloc_query25`) *beats*
   50/50 in the scarcest stratum (**80.0% vs 74.2%**) while matching it elsewhere ⇒ the scene
   deserves more budget than we gave it, and the split is a tunable we never tuned.

### Reconciliation with V*Bench
V*Bench still shows +13.9pp at ~6 tokens where POPE shows nothing at 4.19. Not a contradiction:
POPE is at ceiling there (uniform 98.3%) while V*Bench uniform is at 66–77%. ⇒ **benefit tracks
token scarcity × task headroom**, and POPE cannot measure the large-object end.

---

## 4J. Phase 22 (E2) — does attention-guided pruning drop small targets? ❌ PREDICTION REFUTED

`phase22_pruning_retention.py`, n=600 (the Phase 21 stratified sample), layer-2 attention from the
final prompt position, keep-rates r ∈ {0.10, 0.25, 0.50}.

**The hypothesis** (ours): the ACL'25 pruning paper reports that attention-guided pruning is beaten
by random/uniform selection and "catastrophically fails on fine-grained localization" without
explaining why. We predicted Phase 6c supplied the mechanism — attention does not mark small
targets, so an attention-guided selector cannot keep them. **Pre-registered test:** retention of the
target's tokens below the keep-rate r (which is chance by construction for random/uniform).

**Result: the opposite.** Attention-guided retention is ABOVE chance everywhere, and *most* above
chance exactly where we predicted it would be worst:

| stratum (tokens on object) | r=0.10 | r=0.25 | r=0.50 |
|---|---|---|---|
| <0.5 (med 0.26) | **+20.2** [+12.9, +27.9] | **+34.8** [+27.3, +42.5] | **+33.3** [+28.7, +37.9] |
| 0.5–2 | +16.2 [+10.6, +22.4] | +31.8 [+26.0, +37.7] | +33.8 [+29.9, +37.4] |
| 2–8 | +10.6 [+7.0, +14.3] | +23.9 [+18.9, +29.0] | +26.6 [+22.8, +30.2] |
| 8–32 | +1.7 [−0.2, +3.7] | +10.8 [+6.9, +14.9] | +14.1 [+10.0, +18.2] |
| >32 (med 104.9) | −1.1 [−2.3, +0.4] | +3.1 [+0.7, +5.6] | +6.0 [+3.3, +8.7] |

(values are retention minus keep-rate, in pp; chance = 0)

⇒ **Attention-guided selection preferentially PROTECTS small targets** (+20 to +35pp over chance)
and is at chance for large ones. Per the pre-registration this is reported as a refutation: our
Phase 6c-based explanation for the ACL result **does not hold**, and that failure needs a different
cause (plausibly loss of spatial uniformity for scene-level context — which is what the ACL paper
itself suggests — not loss of the target).

### 4J.1 The subtlety that misled us — MASS vs RANK
Attention *mass* on the target IS sub-proportional to its grid share, at every stratum
(enrichment **0.49–0.95×**, i.e. below 1.0), which is consistent with Phase 6c. But top-k pruning
selects on **rank**, not mass: the target's few tokens can rank inside the top-k while carrying
less than their proportional share of mass (mass is dominated by attention sinks elsewhere).

> **Phase 6c's low attention enrichment does NOT imply attention-guided selection drops the target.**
> We conflated the two, and this experiment caught it. Any future argument from Phase 6c must
> specify whether it depends on attention mass or on attention rank.

⇒ E2's intended contribution (supplying the mechanism for a published unexplained result) is
**withdrawn**. What remains is a clean negative that prevents a wrong claim, plus the mass/rank
distinction.

### 4B.4 Required next step
n=54 clean items is thin. **Rebuild the confident-denial cohort RePOPE-clean from the start** —
re-identify `P(yes)<0.01` in **fp16** (cached values are 4-bit, bug #6) across all RePOPE-clean
positives — then re-run the Phase 10 arms at proper n.

---

## 5. Supporting / footnote material

### 5.1 LLaVA center-crop (FOOTNOTE ONLY — see constraint #2)
LLaVA-1.5's CLIP preprocessing (resize-shortest-edge-336 + center-crop-336) physically deletes
peripheral image content; quantified per-object as `crop_survival_frac`. A pad-instead-of-crop fix
(`PIL.ImageOps.pad`) was validated with a negative control (crop-intact items) and a mathematical
counterfactual (a pure logit bias-shift cannot reproduce one subgroup improving while another
degrades). Explains why LLaVA is *more severe* than Qwen on peripheral objects. **Not the headline.**
A survey of ~16 VLM preprocessor configs is in `phase3_model_survey.py`; note LLaVA-NeXT was
initially mis-bucketed as center-crop but actually uses AnyRes multi-tile processing.

---

### 4K. Phase 23 (E0): AnyRes tiling at matched realized budget, V*Bench n=191
Budget gate passes (spread 2.6% at B=600, 5.0% at B=1200). `anyres@300` is INFEASIBLE and void --
see bug #21: every image costs >=64 merged tokens, so a 2x2 grid + thumbnail cannot cost <320.

| contrast (paired, matched realized tokens) | B=600 | B=1200 |
|---|---|---|
| alloc_query vs **anyres** | **+36.6pp** [+28.8, +44.5] | **+33.0pp** [+25.1, +40.3] |
| alloc_query vs uniform | +27.2pp [+19.9, +34.6] | +22.0pp [+14.7, +29.3] |
| **anyres vs uniform** | **-9.4pp** [-16.2, -2.6] | **-11.0pp** [-17.8, -4.2] |
| anyres_pick vs anyres | +9.4pp [+2.6, +16.2] | +14.1pp [+7.9, +20.9] |
| **alloc_query vs anyres_pick** | **+27.2pp** [+19.4, +35.1] | **+18.8pp** [+11.5, +26.2] |
| alloc_query vs alloc_random | +38.7pp [+30.9, +46.6] | +28.8pp [+21.5, +36.1] |

**The pre-registered "our baseline was a strawman" outcome did NOT occur -- it inverted.** `anyres`
recovers **-35%** (B=600) and **-50%** (B=1200) of the uniform->alloc_query gap. Tiling is *worse*
than plain downsampling at matched budget, in all three target-size terciles, and worst on the
LARGEST targets (57.8% vs uniform's 78.1% at B=600). So `uniform` was a generous baseline, not a
weak one.

#### ⚠️ THE CONFOUND -- do not quote the headline without it
Arms differ in **image count**, and Qwen3-VL was never trained on tiled input. Real AnyRes
concatenates tile features into ONE token sequence with spatial position encoding; our arm presents
tiles as N separate images. So contrasts that change the image count partly measure **format
mismatch**, not allocation policy:

| contrast | images | format-clean? |
|---|---|---|
| alloc_query vs alloc_random | 2 vs 2 | **YES** -- the query-conditionality decider is safe |
| **alloc_query vs anyres_pick** | 2 vs 2 | **YES** |
| alloc_query vs anyres | 2 vs 5 | **NO** |
| anyres vs uniform | 5 vs 1 | **NO** |
| anyres_pick vs anyres | 2 vs 5 | **NO** |

⇒ **"Tiling loses to downsampling" is the confounded contrast and must NOT be written as a claim
about production VLMs.** It is currently: *on a model not trained for tiled input, tiled input
underperforms.* Gate 1 in the Phase 23 docstring anticipated exactly this. The fix is to run the
arm on **InternVL2-8B / llava-onevision** (both cached), where tiling is the native path -- a
narrow, targeted resurrection of E1, NOT the cancelled cross-architecture sweep.

#### What Phase 23 DOES establish, cleanly and newly
**At matched budget AND matched image count, a tight query-placed crop beats a grid-aligned tile
containing the same target: +27.2pp (B=600), +18.8pp (B=1200).** Both arms are query-informed and
both use 2 images, so this isolates *grid alignment* as the cost. It says the constraint that
regions must be grid cells is itself expensive -- which neither Q-CueGraph nor RUTA tests, since
both select free-form boxes. This is the one contribution in Phase 23 that survives the confound.

Cross-budget (format-confounded, recorded for completeness): `alloc_query@300` (301 tok, 91.1%)
beats `anyres@1200` (1235 tok, 59.2%).

### 4N. Phase 26: the ANSWER-MATCHED probe -- C2 is a USE failure, not a routing failure
Phase 24's `last` probe was circular (4L). Fix: condition on the model's answer, probe the truth
inside it. **ANSWERED-NO cohort, n=612** -- the model said "no" to every item, so its answer is
constant across the probe's two classes and decoding it yields nothing.
153 objects actually PRESENT (false negatives) vs 459 actually ABSENT (true negatives).
Built from Phase 24's saved features; no new forward passes.

| | AUROC |
|---|---|
| permutation null (label-shuffled) | 0.531 |
| **baseline: `p_yes` alone, a single scalar** | **0.899** |
| `last_blank` (same position, BLANK image = language prior) | 0.70 - 0.83 |
| **`last` (answer position), L8 onward** | **0.958 - 0.982** |

⇒ **Among items the model answers "no", the answer position linearly separates objects that are
actually present from ones that are actually absent at ~0.98** -- above the language-prior floor
(~0.80), above the model's own confidence scalar (0.899), and far above the null (0.531).

**This is pre-registered outcome #3, and it rewrites C2.** The evidence is present *at the readout
position itself*, on the items the model gets wrong, and the model does not act on it. C2 is a
**USE failure, not a routing failure** -- "irreversible" was the wrong word. Same picture in the
sub-token stratum (n=555, baseline 0.870, `last` 0.916-0.963 from L6).

**Caveats that must travel with this.**
* `last_blank` at 0.70-0.83 is HIGH: within a "said no" cohort the question text alone predicts a
  lot (category priors). The claim rests on the `last` - `last_blank` gap, not on 0.98 absolute.
* The probe beats `p_yes` (0.899 -> 0.98), so it is not merely re-reading the logit margin -- but
  0.899 shows the model's own confidence ALREADY carries most of this signal. The honest framing is
  "graded evidence survives to the readout and is not thresholded into the answer", not "the model
  secretly knows".
* `obj` (~0.99 at every layer including L0) remains uninterpretable here for 4L's reason and is not
  cited.

### 4O. Phase 27: query-independent budget does NOT substitute for placement (bound: 26.5x)
Qwen3-VL-2B, V*Bench n=191. `crop_only` held FIXED at 300 tokens; `uniform` swept over a MEASURED
ladder. **Both arms single image**, so the sweep is format-clean at every rung (Phase 23's confound
cannot recur). Reference: `crop_only@300` = **93.7%**, `crop_random@300` = 38.2% (same budget, wrong
place).

| uniform, realized tokens | accuracy | vs `crop_only@300` |
|---|---|---|
| 150 | 46.6% | -47.1pp [-54.5, -39.3] |
| 294 | 56.5% | -37.2pp [-45.0, -29.3] |
| 600 | 66.0% | -27.7pp [-35.1, -20.4] |
| 1176 | 70.2% | -23.6pp [-30.9, -16.2] |
| 2400 | 75.4% | -18.3pp [-25.1, -11.0] |
| 4760 | 81.2% | -12.6pp [-19.4, -6.3] |
| **7957** | **84.8%** | **-8.9pp [-15.2, -2.6]** still behind |

⇒ **At 26.5x the budget, query-independent spending is still significantly behind a 300-token
query-placed crop.** Stated as a BOUND -- no crossing up to 7957 realized tokens (26.5x) -- never as a claim
about all budgets. The gap closes ~3.7pp per budget doubling in the last rungs, so parity by
extrapolation lies beyond ~40k tokens (>100x); **that extrapolation is NOT a measured result and
must be labelled as extrapolation if used at all.**

#### RQ-B (the 1/area scaling law) is NOT demonstrable on this benchmark -- report the failure
All three target-size terciles fail to cross by 7957 tokens (smallest 83% vs 90.5%; largest 86% vs
96.9%). V*Bench's *largest* targets are still tiny in absolute terms (overall median area fraction
0.001), so the whole benchmark sits in one regime and the size-dependence of the crossing budget
cannot be resolved here. Testing it needs a benchmark spanning a wide target-size range --
**RePOPE-clean spans 0.08 to 290 merged tokens** and is the natural venue. Until then the scaling
claim is unsupported and must not be written.

### 4P. Phase 29: our V*Bench readout is NOT inflating anything — the gap to prior work is the ORACLE
ViRGo (2606.21968) reports, on **Qwen3-VL-2B / V*Bench / all 191 items — our exact setup**:
global perception **64.2%**, RAP (patch retrieval) **78.9%**. We report `uniform@2400` 75.4% and
`crop_only@300` **93.7%**. A reviewer who knows ViRGo will read 93.7% as inflated. Two candidate
explanations, and they had to be separated before either number could be written:

1. **Readout.** We take argmax over pooled {A,B,C,D} logits; ViRGo follows each method's original
   protocol, i.e. **generating** an answer and matching it. Constrained argmax *cannot* emit an
   unparseable answer; free generation can. That would inflate every arm, baseline included.
2. **Oracle premium.** Their RAP is a training-free *proposer*; our crop is the benchmark's own GT box.

**Measured (1) directly** — same items, same images, same budgets, same model, only the readout
changes, unparseable generations counted as **WRONG**:

| arm | logit argmax | generate + exact-match | offset | unparseable |
|---|---|---|---|---|
| `uniform@2400` | 75.4% | 75.4% | **+0.0pp** | 0 |
| `crop_only@300` | 93.7% | 93.7% | **+0.0pp** | 0 |

n=191; only **2 of 191** items disagree at all, and they cancel. The model always emits a parseable
"(A) rubber"-style answer.

⇒ **The readout explains none of the gap. The oracle explains it** — and that converts section 9's
oracle caveat from an apology into a measurement:

> A published training-free proposer (ViRGo's RAP, 78.9%) captures roughly **31-44% of the oracle
> gain** on this backbone and benchmark — 44% against ViRGo's baseline (64.2%), 31% against ours
> (75.4%). The remaining ~56-69% is what knowing the ground-truth box buys.

**Caveat that must travel with it:** this is a cross-paper comparison and budgets are NOT matched
between their RAP and our arms, so it is a bound, not a controlled contrast. A residual ~11pp gap on
the *baseline* arm (our 75.4% at 2400 tokens vs their 64.2%) is unexplained by readout and is most
likely a different native operating point; it is reported, not explained away.

### 4M. Phase 25 (E0 done properly): query placement vs NATIVE AnyRes, LLaVA-NeXT, n=191
The Phase 23 confound fix, run on the architecture that **introduced** AnyRes. Every arm is a
SINGLE image through `LlavaNextProcessor` unmodified, so there is no image-count / format
mismatch. Budget gate: **0.0% spread** -- `anyres`, `alloc_query`, `alloc_random` all realize
exactly **2144** image tokens (median), matched per item to that image's own native AnyRes cost.

#### ⚠️ CORRECTED 2026-09-11: the original gate was MEDIAN-based and too lenient
Medians agreed exactly (0.0% spread), but **39.3% of individual items were mismatched by >10%**
(per-item spread p90 = 46.4%, max = 115.2%). The first-reported numbers below aggregated matched and
unmatched items into one figure. **The defensible numbers are the per-item-matched subset, n=116.**

| arm | as first reported (n=191, median gate) | **CORRECTED (n=116, per-item gate)** |
|---|---|---|
| uniform (unmatched reference) | 47.6% | 50.0% |
| **anyres** (shipped high-resolution path) | 55.0% | **60.3%** |
| **alloc_query** (same budget, question's region) | 88.5% | **86.2%** |
| **alloc_random** (same budget, wrong region) | 39.8% | **47.4%** |

| contrast (paired) | as first reported | **CORRECTED (n=116)** |
|---|---|---|
| query placement vs native AnyRes | +33.5pp [+25.7, +40.8] | **+25.9pp [+17.2, +34.5]** |
| query vs size-matched random (DECIDER) | +48.7pp [+41.4, +56.0] | **+38.8pp [+29.3, +48.3]** |
| random sub-region vs tiling | -15.2pp [-23.0, -7.9] | **-12.9pp [-22.4, -4.3]** |

**All three conclusions survive, all three shrink.** Use the corrected column. The analyzer now gates
per item and reports how many items it drops.

**The third row is the most important control in this project.** Spending the identical budget on a
size-matched RANDOM region is **15.2pp WORSE than tiling**. So the gain is not "cropping beats
tiling" and not "sub-regions beat whole images" -- a placement-free sub-region actively HURTS. The
budget is not the constraint; **the placement policy is**.

⇒ **At matched realized token budget, on the model that ships AnyRes and in its own native format,
query-conditional placement beats the shipped high-resolution pathway by +33.5pp, while random
placement at the same budget loses to it by 15.2pp.**

#### What this does to 4K
Phase 23's *format-confounded* rows (Qwen tiles-as-separate-images) are superseded, not rescued:
4M is the citable version and 4K's tiling contrasts should not be quoted. 4K's one clean finding --
tight crop vs grid-aligned tile at matched image count, +27.2/+18.8pp -- still stands independently.

#### Relation to prior art (checked, LITERATURE_GAPS.md)
Q-CueGraph reports V*Bench 0.696 full-image -> 0.833 with an anti-region control at 0.393; our
`alloc_random` at 39.8% lands on essentially their anti-region number, which is a healthy external
replication. What is **not** in Q-CueGraph or RUTA: (i) matching on **realized tokens** rather than
image AREA, and (ii) a **native-tiling arm** at that matched budget. Those two are 4M's contribution;
the phenomenon that query-conditioned regions beat generic ones is **theirs** and must be cited as
such.

#### Method note worth reporting -- three void versions preceded this table
The budget gate rejected three earlier builds, all of which produced `alloc_query` 95-96% vs
`anyres` ~50%: (1) a proportional scale solver that could not cross LLaVA-NeXT's token-count
plateaus; (2) a global budget axis this architecture cannot honour (it realizes only a handful of
counts, aspect-dependent); (3) crops whose ASPECT differed from the source, which can never realize
the same count because LLaVA-NeXT **unpads tile features by aspect ratio**. The fix was to expand
every region to the source aspect ratio -- which also means `alloc_query` here is *the query region
expanded to source aspect*, NOT a tight crop, and is therefore not a tight-crop upper bound.
**The accuracy numbers barely moved across all four versions; only their admissibility did.**

### 4L. Phase 24 probe sweep -- ⚠️ PARTIALLY RETRACTED, see the circularity note below
Full-depth probe sweep, 29 layer outputs x 4 pooled positions, 5-fold cross-validated, n=1113
(failure-enriched: all 153 wrong positives + all 42 wrong negatives + 459 tokens-on-object-matched
correct controls). Permutation null at n=1113: **0.525** -- so 0.5 is nearly the right chance level
at this sample size, and the estimator is not badly overfitting.

**ALL-ITEMS cohort, the `last` position (where yes/no is emitted):**

| layer | `last` | `last_blank` (language-prior floor) |
|---|---|---|
| L0-L5 | 0.453 -> 0.683 | at or **above** `last` |
| **L6** | **0.780** | 0.719 |
| **L7** | **0.878** | 0.722 |
| **L8** | **0.943** | 0.736 |
| **L9** | **0.966** | 0.744 |
| L10-L19 | ~0.98 plateau | 0.707-0.769, flat |

⇒ **Below L6 the answer position carries nothing the language prior does not already supply.**
Across L6-L9 it goes 0.68 -> 0.97 and saturates. `last_blank` is flat throughout, so this is not
drift in probe difficulty -- it is visual evidence arriving, and the arrival is localized to a
four-layer window.

#### ⚠️ The `obj` curve is NOT part of this claim
`obj` sits at ~0.96 **from L0, the embedding layer**, before any transformer block runs. For
positives it pools the GT bbox tokens; for negatives the object is absent so it pools a RANDOM
region -- so it largely separates "patch embeddings of a real object" from "patch embeddings of a
random patch of scene", a low-level distinction, NOT evidence that the model encodes whether the
QUERIED category is present. Same failure family as bug #19. The query-conditional version is
Phase 15's within-positive design (queried vs size-matched other category) extended across depth;
see RESEARCH_PLAN.md section 9. Do not cite `obj` as C2 evidence.

#### 🛑 RETRACTED SAME DAY: the `last` curve is CIRCULAR. Do not cite section 4L as written.
The wrong-answer cohort (n=195, permutation null 0.595) came back with `last` at **0.997-1.000 from
L16 to L28** -- on the very items the model answers WRONG. Taken at face value that is
pre-registered outcome #3 ("the readout has the information and ignores it"), which would refute
C2's "irreversible" framing. **It is an artifact, and the artifact also undermines the L6-L9 window
above.**

**The circularity.** `last` is the position the yes/no logits are read from, so the model's OWN
ANSWER is linearly decodable there essentially by construction. Now note how the cohorts are built:
* wrong-only cohort: every item is wrong, so `true label` == NOT(`model answer`) **exactly**.
* correct-only cohort: every item is right, so `true label` == `model answer` **exactly**.
* all-items cohort: the model is right 82.5% of the time, so label and answer agree 82.5% of the time.

In every one of them a probe can score near-ceiling by decoding **the model's own answer** and, where
required, flipping the sign. It never has to represent visual evidence at all. The high
`last_blank` values in the failure cohort (0.75-0.82, well above the 0.595 null) are a further
symptom: even with NO image the answer position predicts the label, because the cohort was selected
using the answer.

**Diagnostic (pre-registered before cohort 3 ran) -- ✅ CONFIRMED.** Predicted: if the CORRECT-only
cohort also returns `last` ~ 1.0, circularity is proven, because the two cohorts have OPPOSITE
label-answer signs and cannot both reflect genuine evidence with the same probe. Result:

| cohort | label vs answer | `last` @ L16-L28 | `last_blank` |
|---|---|---|---|
| wrong-only (n=195) | label == NOT answer | 0.997 - 1.000 | 0.735 - 0.823 |
| **correct-only (n=918)** | label == answer | **1.000 at L16, L20, L24, L28** | 0.723 - 0.725 |

Both ceilings, opposite signs. The probe is decoding **the model's answer**, not visual evidence.
Confirmed, not merely suspected.

**What survives:** the `last` curve measures **where the model's decision becomes linearly
decodable**, which is a real quantity but NOT "where visual presence information arrives". Those are
different claims and 4L conflated them.

**The fix -- an ANSWER-MATCHED probe (not yet run).** Condition on the model's answer and probe the
truth within it: among items the model answered **"no"**, positives are objects that are actually
present (false negatives) and negatives are objects that are actually absent (true negatives). The
answer is then CONSTANT across the probe's two classes, so decoding it yields nothing, and any
remaining AUROC is genuine evidence about the input. That is the design C2 needs; the cohort sizes
(153 FN vs the true-negative pool) make it feasible. See RESEARCH_PLAN.md section 9.

This is the third time in this project that a strong-looking number came from the selection rule
rather than the model (cf. Phase 10 re-anchoring, Phase 16 `steer_last`, bug #19).

### C3 strengthened: error rate vs target token budget, whole RePOPE-clean pool (n=3387)
Computed offline (2026-09-09) from `phase12_clean_pyes.jsonl`'s stored `image_grid_thw` plus the
COCO bboxes -- **no new GPU time**. This is a cleaner statement of the domain claim than anything
previously in this file, because it is the FULL clean positive pool rather than a sampled cohort,
and the trend is monotone across all seven strata:

| tokens on object | n | wrong | error rate |
|---|---|---|---|
| [0, 0.5) | 174 | 57 | **32.8%** |
| [0.5, 1) | 129 | 24 | 18.6% |
| [1, 2) | 189 | 15 | 7.9% |
| [2, 4) | 219 | 12 | 5.5% |
| [4, 8) | 252 | 12 | 4.8% |
| [8, 32) | 678 | 15 | 2.2% |
| [32, inf) | 1746 | 18 | **1.0%** |

A **33x** swing in error rate across the budget range, monotone, on a benchmark whose overall
accuracy is 95.5% on these items. The aggregate number hides the entire phenomenon.

**It also caps the C2 experiment.** The whole clean pool holds only **153 wrong positives and 42
wrong negatives** -- 195 items total. That is the hard ceiling on any failure-cohort analysis
built on POPE, which is why Phase 24 takes *all* of them, matches correct controls on
`tokens_on_object` (small objects are both likelier to fail AND harder to decode -- an unmatched
comparison would manufacture C2's predicted result), and reports every cell against a
**permutation null** instead of 0.500.

### Audit: Qwen3-VL uses patch=16, not 14 -- and why `tokens_on_object` is UNAFFECTED
Measured from the processor: `patch_size=16, merge_size=2`, so one merged token covers **32x32 px**
of the resized frame, not the 28x28 that `PATCH=14, MERGE=2` in `phase6_attention_entanglement.py`
assumes (that constant is right for Qwen2-VL, wrong for Qwen3-VL). I checked whether this
invalidates the stratification used by Phases 13/17/21/24. **It does not**, and the reason is worth
recording so nobody "fixes" it later and breaks it:

    tokens_on_object = (dx * rz_w/W) / (PATCH*MERGE) * (dy * rz_h/H) / (PATCH*MERGE)
    with rz_w = g[2]*PATCH, rz_h = g[1]*PATCH
                  = (dx * g[2]) / (2W) * (dy * g[1]) / (2H)

`PATCH` cancels exactly, because it appears in both the resized-frame size and the divisor. The same
cancellation makes `object_token_indices()` map bboxes to the correct grid cells. So every
`tokens_on_object` value and every probe token index in this project is correct as computed.

The one place the constant does bite is `TOK_PX = 28` in `fit_to_budget`'s **analytic seed** and in
`_resize`'s rounding granularity. Both are only starting guesses -- the loop converges on the
processor's *measured* `image_grid_thw` -- so budget matching is unaffected. Left as-is deliberately.

## 6. Bugs and near-misses caught (methodological credibility — worth a paper appendix)

| # | issue | consequence if missed |
|---|---|---|
| 1 | Threshold-scan loop broke on first iteration | bogus headline threshold/FPR |
| 2 | LLaVA-NeXT mis-bucketed from raw config (`crop_size:336`) but actually AnyRes multi-tile | wrong cross-model claim |
| 3 | Model-survey substring match `"no crop" in verdict` also caught `"needs manual check"` | inflated no-crop count |
| 4 | `random.seed()` before `build_items()` polluted its global RNG → different population (88/300 uids resolved) | silent cohort mismatch |
| 5 | `max_pixels` silently ignored; fast processor reads `image_processor.size` | intended resolution control was a no-op |
| 6 | **fp16 vs 4-bit mismatch**: cached `p_yes_real` came from a 4-bit model; one item was 0.0067 (4-bit) vs 0.169 (fp16) — crosses the `<0.01` cohort threshold | all interpretability cohorts silently wrong |
| 7 | Phase 5's dramatic 80/80-vs-0/80 trajectory result | would have been a **false headline**; killed by the negatives-control AUROC validation |
| 8 | Phase 6a used only the *largest* COCO instance while Phase 4 scored against *any* instance | measured attention to a different region than was verified localizable → fixed to union-of-instances |
| 9 | Per-item enrichment *ratios* dominated by noise when objects span 1–4 tokens | switched to **pooled ratio** (Σnumerators/Σdenominators) + saved raw mass components |
| 10 | Phase 6b modulation index not polarity-matched across groups | would have been a **false headline**; fixed by the Phase 6c present-Z condition |
| 11 | Phase 6d CoT prompt named the category → model answered "No, there is no {cat}" *inside the CoT*, anchoring the final answer | self-fulfilling result; fixed to a generic scene-description prompt |
| 12 | Phase 6e v1 patched 4 layers (compounding renormalizations) at F=0.30/0.60 (3–6× above any real item's natural share, median 0.094) | uninterpretable "refutation"; fixed to single-layer + realistic F |
| 13 | Phase 7 baseline lacked the connector sentence the zoom arms had | contaminated by 6d's template-shift effect; fixed by relying on oracle-vs-random |
| 14 | **Phase 7's two-image format measured "resolution + a distractor" and reported it as "resolution"** — understating the effect ~3× (12.8% vs 40.4% crop-alone) | would have **under-sold the resolution mechanism** and propped up a wrong "readout beats resolution" headline; caught by the Phase 9 factorial |
| 15 | **Phase 10's other-scene arm looked like +17.0pp evidence for "re-anchoring"** on raw recovery, but carried FP 19.1% vs ~0–4% elsewhere | would have been a **false headline** (a pure yes-bias shift — the §5.1 trap again); killed by reading verdicts on discrimination, CI [−10.6, +6.4] |
| 16 | Phase 8's `p_ground` oracle-threshold balanced accuracy (0.9280) sits at t=0.999998, inside a tie mass of a saturating score | not comparable to `P(yes)`'s 0.9230 at t=0.34; do not quote the pair — AUROC is the honest comparison |
| 17 | **Phase 17 v1 required a positives-only lookup (`pope_coco_area_joined.json`) for every item, silently dropping all 160 negatives** — the entire false-positive arm | every recovery number would have been unfalsifiable; caught because the run wrote 54 rows for 214 targets |
| 18 | **Phase 17 v1 solved budget analytically; the processor's own `smart_resize` overrode it** (asked 150 tokens → uniform got 120, two-image arms got 143) | a **19% budget advantage to the very arm we wanted to win** — the whole matched-budget claim would have been void; fixed by iterating on *measured* `image_grid_thw` |
| 19 | **I labelled Phase 6c the "strongest mechanistic result" and used it to assert "there is nothing at that location to attend to"** — conflating attention **mass** (what 6c measures) with attention **rank** (what retention depends on) | Phase 22 refuted the assertion: attention-guided top-k retention keeps the target **+20 to +35pp above chance** at L2/L4/L8. The target *is* attended. §3.1 label withdrawn and the §4C sentence removed; the surviving reading is that the attended tokens do not carry the detail, not that attention misses them |
| 20 | **Phase 23 v1: the AnyRes arm was calibrated per-image only, so `smart_resize` granularity compounded across 5 images** -- measured B=300 spread **19%**, `anyres`=350 tokens vs every other arm's 300 | a **17% token advantage to the very arm under test** (bug #18's cousin, and caught by the analyzer's budget gate at 27 rows rather than at 191). Fixed with a two-stage calibration: per-image to B/(N+1) to establish AnyRes's equal-share property, then one common rescale of the whole list to hit the total. Run restarted clean; the 27 pre-fix rows were parked, not analyzed |
| 21 | **I chased a "calibration bug" that was a hard model constraint.** Qwen3-VL's processor gives EVERY image a floor of **64 merged tokens** regardless of size (28x28 px -> 64; 112x112 px -> 64), so a 2x2 AnyRes grid + thumbnail = 5 images costs **>=320 tokens** and `anyres@300` pins at 350 no matter how it is calibrated | I restarted the run once (bug #20) before measuring the floor -- I should have measured it first. The two-stage fix from #20 is still correct and kept, but it was not what #20's symptom needed. **`anyres@300` is now marked INFEASIBLE rather than mismatched**, and the floor is reported as a result: *tiling has a minimum budget of n_tiles x 64 tokens, so the AnyRes pathway cannot be run at a small budget at all.* B=600/1200 gates pass at 2.6%/5.0% |

---

## 7. Infrastructure and reproduction notes

### Files
**Data** (`data/`): `phase1_results.jsonl` (LLaVA, 5553) · `phase1_results_qwen_dedup.jsonl` (Qwen,
5553) · `pope_coco_area_joined.json` · `coco_ann/instances_val2014.json` ·
`phase4_localize_results.jsonl` (486) · `phase5_logit_lens_results.jsonl` (160) ·
`phase5_layer_auroc_results.jsonl` (1000) · `phase6_attention_results.jsonl` (291) ·
`phase6b_query_swap_results.jsonl` (291) · `phase6c_present_other_results.jsonl` (271) ·
`phase6d_decode_steps_results.jsonl` (141) · `phase6e_attention_patch_results.jsonl` (141) ·
`phase7_vision_zoom_results.jsonl` (141)

**Scripts** (`scripts/`): `phase1_eval.py` (`build_items()` — the canonical item loader, used
everywhere) · `phase1_eval_qwen.py` · `phase1_threshold_analysis.py` · `phase3_centrality.py` ·
`phase3_crop_check.py` · `phase3_fix_pad{,_negatives}.py` · `phase3_model_survey.py` ·
`phase4_{ground_format_check,localize_vs_answer}.py` · `phase5_{logit_lens,layer_auroc_validation}.py` ·
`phase6_attention_entanglement.py` (**reusable core**: `Ctx`, `grid_from_image`,
`object_token_indices`, `enrichment_from_attention`, `load_bbox_lookup`) ·
`phase6b_query_swap.py` · `phase6c_present_other.py` · `phase6d_decode_steps.py` ·
`phase6e_attention_patch.py` · `phase7_vision_zoom.py`

Full chronological narrative (including reasoning at each fork) is in `experiment_log.md`. Note its
section order is not strictly chronological — Phase 6c/6e were inserted at their topical positions.

### Method invariants (keep identical for comparability)
- **`P(yes)`**: one forward pass; logsumexp-pool logits over `{yes,Yes," yes"," Yes"}` vs
  `{no,No," no"," No"}` at the final position; softmax over the two pooled logits.
- **Prompt**: `f"{question} Please answer this question with yes or no."` via
  `apply_chat_template(..., add_generation_prompt=True)`.
- **Precision**: fp16 for all interpretability work. Cached `p_yes_real` in the phase1 files is
  **4-bit** — always re-verify cohort membership in fp16 (bug #6).
- **Qwen3-VL image-token geometry**: `image_grid_thw` gives the *pre-merge* patch grid; with
  `merge_size=2`, `patch=14`, the token grid is `(h/2, w/2)`, each token covers 28×28px of the
  **resized** image, row-major, contiguous in the sequence. Verified by hand: bbox `[100,100,50,50]`
  in 640×480→560×420 with a 15×20 grid → indices `{63,64,83,84}`.
- **Grounding output format**: ` ```json [{"bbox_2d":[x1,y1,x2,y2],"label":...}] ``` `, coords
  normalized to **0–1000** regardless of true image size — rescale by `img_w/1000, img_h/1000`.
- **Attention capture** requires `attn_implementation="eager"` + `output_attentions=True`.
- **Small-n statistics**: use pooled ratios and sign/win-rate tests; P(yes) is heavily skewed, so
  bootstrap CIs on *means* can cross zero while the win-rate is overwhelming (Phase 6d).

### Running long jobs
```
tmux new-session -d -s <name> "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/<script>.py 2>&1 | tee -a logs/<name>.log"
```
Monitor with `wc -l` on the output JSONL (stdout is block-buffered through `tee`). All phase
scripts are **resumable** (they skip uids already in their output file). `logs/` must exist before
launching or `tee` fails silently. GPU0 (49GB) is shared with unrelated jobs — check
`nvidia-smi` first and never kill another user's process; GPU1 (4GB) cannot fit the model in fp16.

---

## 8. One-paragraph summary of where this stands

*(rewritten 2026-09-07 after Phases 8–10.)*

**Scope correction first.** `P(yes)` scores **0.9725 AUROC** on a broad POPE sample, and **0.948**
even in the smallest area quartile. So *"VLMs localize well but answer poorly"* is **not supported
as a capability claim**, and no section should assert it. The real object of study is a **confident-
error tail**: ~6% of positives get `P(yes) < 0.01` while the object is plainly present. Not
uncertain — confidently wrong. That tail is what matters for deployment and is the paper's subject.

**What survives, with controls:** (1) the **dissociation** — on confident denials the model still
puts a box in the right place (28.9% IoU>0.5 vs a 6.4% trivial-box floor); note only *IoU* survives,
since *emission* is a yes-bias (§4.1). (2) The **attention target-conditioning collapse** (§3.1),
with the polarity confound ruled out. (3) **Under-resolution is a large, clean cause**: a crop of
the same pixels recovers **40.4%** of confident denials at **3.5% FP** (§4A). (4) **Visual context
interference** (§4A): the same crop recovers only **12.8%** when the full scene shares the prompt —
attributable to **scene content**, with image count, format, and connector text each exonerated;
plus a **recency** effect (full image last is worst, 4.3%).

**What is dead:** answer-by-grounding as a detector and channel-disagreement (§4.1/§4.2, Phase 8);
three decoder-side mechanisms (§3.2); "re-anchoring on the model's own failed percept" (§4A, killed
by its own FP control); and the "readout beats resolution" framing drafted earlier this session
(killed by Phase 9 — Phase 7 had understated resolution ~3×).

**Where it stands (updated 2026-09-09).** The live lead is no longer context interference — it is
**budget-matched query-conditional allocation** (§4F/§4H): at equal *realized* tokens, placing the
budget where the question points beats both uniform downsampling and a **size-matched random
region** (V*Bench +30.4pp [+23.0, +37.7], n=191). The open liabilities are, in order:

1. **The `uniform` baseline may be a strawman.** No production VLM downsamples the whole image;
   LLaVA-NeXT/InternVL2/Qwen-VL tile it (AnyRes / dynamic tiling) — still uniform over space and
   still chosen before the question, but at a much better operating point. **Phase 23 is running
   this arm now.** Until it lands, the sentence "modern VLMs allocate uniformly" is an inference
   about those systems, not a measurement, and must not be written.
2. **One model.** Everything above is Qwen3-VL-2B. Cross-architecture + patch-size invariance (E1)
   decides whether the effect is about *tokens* or merely about *pixels*.
3. **An oracle region.** `alloc_query` uses ground-truth boxes and is stated as an oracle. A
   training-free proposer would be reported as *% of oracle gain captured*, not as a method claim —
   beating V*/SEAL on their own leaderboard is **not** reachable with these assets.
4. Unchanged: the **42% of confident denials that neither resolution nor readout recovers**.

---

## §4Q. Token-budget dynamic range is an architectural property, and it bounds what can be asked

Measured with the processor alone (no model load), on a 2250x1500 V*Bench source. "Floor" is the
realized merged-token count for a 64x48 input; "ceiling" is the count at 2x the source resolution.

| model | budget mechanism | floor | ceiling | dynamic range |
|---|---|---|---|---|
| Qwen2-VL-7B | continuous | 4 | 7776 | **1944x** |
| Qwen3-VL-2B | continuous | 64 | 7957 | **124x** |
| llava-onevision-7b | tiles, 384px | 1261 | 7329 | **5.8x** |
| LLaVA-NeXT-vicuna-7b | tiles, 336px | 1416 | 2144 | **1.5x** |

The ceiling is structural, not a search artifact. LLaVA-NeXT returns **exactly 2144 tokens** for
inputs of 562x375, 1125x750, 2250x1500 and 4500x3000 -- four inputs spanning 64x in area. Its
`image_grid_pinpoints` set has five entries topping out at 1008x1008, so no input can express more.
(I checked this specifically against the phantom-cap error, where a coarse ladder *manufactured* an
apparent cap that refinement dissolved. This one survives refinement and survives 4x oversampling.)

### Two consequences, one of them a correction to §4

**1. The exchange-rate experiment cannot be run on LLaVA-NeXT at all.** With floor 1368 and ceiling
2340 on the probe image, the realizable ladder reduces to a **single rung**. There is no second
budget to sweep to. This is not a null result about token scaling; it is the absence of an axis.

**2. The onevision "no crossing at 3.9x" row was reported as a weaker result than Qwen's 25.9x and
26.5x. That reading was wrong.** onevision's *entire* dynamic range is 5.8x, so 3.9x is ~67% of
everything the architecture can express. Qwen's 26x sweeps cover ~2% and ~21% of their ranges. The
three models were never sweeping comparable fractions of their own capacity, and the raw multiples
are not comparable across them. The correct statement is that **no architecture crossed within the
range it can realize**, and the multiple at which each was tested should be reported as a fraction
of that model's dynamic range, not as a bare number.

### Why this was found
The Phase 28 LLaVA-NeXT leg was OOM-killed by the host (26 GB RSS, 30 GB machine, pid 381958). Cause:
the ladder's early-stop is `realized > 1.3 * max_target`, and with `max_target = 26 * B0 = 35568` on
a model capped at 2144 that condition can never fire -- so the ladder climbed to scale 4.0 on a
2250x1500 source (54 megapixels) and the image processor took the machine down. Fixed by measuring
the ceiling the way the floor is already measured, skipping unreachable rungs, stopping the ladder
when two consecutive scales realize the same count, and adding a hard 24 MP resize guard.

## §4R. NEGATIVE (recorded, not run): region packing has no valid venue on the available benchmarks

Considered as the paper's method contribution: instead of cropping the **union** of the GT boxes
(which is what Q-CueGraph's operation amounts to, and what this paper's own `alloc_query` arm does
at line 232 of `phase28_arch_invariance.py`), crop each region separately and compose them onto one
canvas, discarding the dead space between them, at matched realized token budget.

**The motivating measurement is real and stands on its own.** On V*Bench's multi-box items (n=54):

| quantity | median | mean | range |
|---|---|---|---|
| fraction of the union bbox that is dead space | **0.810** | 0.771 | 0.085 - 0.995 |
| union bbox as fraction of image | 0.0143 | | |
| sum of boxes as fraction of image | 0.00169 | | |

So a published region-selection method spends roughly **four-fifths of its already-tiny budget on
image area containing none of the evidence.** Deciles run 0.47 -> 0.97, i.e. this has real spread
and would serve as a per-item predictor.

**The method is nevertheless void, for a structural reason.** Packing is computed from the oracle
boxes. On a *relational* question ("is the umbrella left or right of the traffic light?"), the
arrangement the packer reconstructs **is the answer** -- the transformation performs the very
discrimination the benchmark measures. That is categorically worse than this paper's other oracle
arms, which say where to look and leave the perceiving to the model. And the venue is gone:

- V*Bench `direct_attributes`: 115/115 items have a **single** box. Packing reduces to cropping.
- V*Bench `relative_position`: all 54 multi-box items are relational. **Void by construction.**
- HR-Bench (4k/8k): columns are `index, answer, question, A, B, C, D, category, cycle_category,
  image`. **No box annotations at all.** No packing possible.

Packing needs a multi-region, non-relational, box-annotated split. No benchmark available here has
one, and the standing constraint is public benchmarks only -- so this is not buildable, not merely
unproven. Recorded as a negative with its mechanism, and the 81% dead-space number is retained as a
measurement about the existing operation.

**Rejected diagnostic, for the record:** a `packed_flipped` control (pack regions in reversed
arrangement) cannot discriminate. Flipping removes the true arrangement, so the canvas carries no
correct-answer signal and *both* an order-reading model and a content-reading model fail. Collapse
is predetermined; the run would confirm nothing.

### Prior-art check performed (narrow, one query)
Composing disjoint crops onto a canvas is not taken. The two nearest: **AwaRes** (2603.16932,
IBM/TAU) appends retrieved high-res crops to the dialogue as *separate images* on top of a retained
low-res view -- token count **grows**, RTR is what they reduce rather than hold fixed, crop set is a
fixed quadrant/half/center menu, and the policy is trained (SFT + multi-turn GRPO). **Progressive
Visual Scaffolds** (2608.21170) overlays markers, axis labels and coordinates on the *original*
image with no crop-and-repack and no token control, on grid puzzles. Neither composes a canvas and
neither holds realized tokens fixed. The gap is real; the benchmark to test it in is not.

## §4S. NEGATIVE (pre-registered, n=191): attention is concentrated but not CO-LOCATED, so a bounding-box read-out cannot allocate

Phase 30a. Qwen3-VL-2B, V*Bench, B0=300, one forward pass per item. Proposal = min/max bounding box
of the top-r image tokens by attention from the final prompt position.

**Result: median area_frac = 1.0000.** The proposal is the entire image on essentially every item,
at **every** layer (2/4/8) and **every** keep rate (0.10/0.25/0.50) -- 0.0% of items produced a box
covering <= 25% of the image. Under the rule fixed before running (`median area_frac > 0.8 -> DEAD`)
this read-out is dead: a matched-budget re-render of "the whole image" IS the uniform baseline.

**This is not "attention is diffuse" -- it is the opposite, and the distinction is the finding.**
Attention is extremely concentrated: on a 7x10 grid, 50% of image-attention mass sits on the **top 2
of 70 tokens** (max/median ratio 155x at L2). But the high-attention tokens are not co-located --
the top-2 box is compact (area 0.143) while the top-6 box (80% of mass) spans the full grid. A few
positions absorb large attention largely independent of content (attention-sink / register-token
behaviour). A min/max bbox is an extreme-order statistic, so a handful of scattered sinks destroy it
no matter how well the ranking works.
  *(Those concentration figures are from a 7x10 grid in a one-off diagnostic, NOT from the 15x20
  grid the experiment runs at. They are quoted with that grid attached and are not carried forward
  as B0=300 facts; Phase 30b re-measures on the real grid.)*

**Reconciles with Phase 22 rather than contradicting it.** Phase 22 measured token *rank* (do the
target's tokens rank above chance? yes, +34.8pp for the smallest objects). It never measured the
*area* those tokens span, which is what a proposer must pay for. **Rank quality and spatial
compactness are different properties, and only the first was ever established.**

**`attn_mass_in_gt` (median 0.0000) is undefined here, not low** -- do not cite it as evidence.
At B0=300 a V*Bench target spans **0.04-2.4 merged tokens** (most under one), so the set of grid
cells whose centre falls inside the GT box is frequently EMPTY. Phase 30b replaces it with the rank
of the single cell containing the GT centre, which is always defined.

### Sub-token regime, sharper than the C3 dose-response table
Measured directly: at B0=300 the V*Bench target occupies **0.04 to 2.4 merged tokens**, median well
under one, against a median GT box area of 0.108% of the image. **The evidence the question asks
about is smaller than one visual token.** This is why `uniform@300` sits at 56.5% while
`crop_only@300` reaches 93.7% at the same cost, and it belongs wherever C3 is stated.

## §4T. Phase 28 LLaVA-NeXT leg: placement contrast only, and the uniform comparison is VOID

n=191. crop_only 80.6% vs crop_random 38.2%, **placement +42.4pp CI [+34.6, +50.3]** -- a fourth
architecture where *where* the budget goes dominates. But `crop_only` realized 1176 tokens against
B0=1368, **14.0% off, which fails this paper's own 10% budget gate**: the crop-vs-uniform row is
VOID and is not reported. The crop_only-vs-crop_random contrast survives because both arms are
equal-size crops matched to each other.

Updated RQ-C table, now with each test multiple expressed as a fraction of the model's own
dynamic range (see 4Q -- bare multiples are not comparable across architectures):

| model | xB0 tested | dyn range | % of range | crossed? | crop_only | uniform@max |
|---|---|---|---|---|---|---|
| qwen2vl | 25.9 | 1944x | 1% | NO | 92.4 | 75.4 |
| onevision | 3.9 | 5.8x | 67% | NO | 86.9 | 69.6 |
| llavanext | 1.2 | 1.5x | 83% | n/a (no axis) | 80.6 | 38.2 (void vs B0) |

## §4U. The localizer IS in the attention map -- the argmax is not, and that is the whole problem

Phase 30b, n=191, Qwen3-VL-2B, V*Bench, B0=300. Post-hoc repair attempt on 4S; 4S remains the
pre-registered headline.

`attn_mass_in_gt` was undefined in 4S because 71% of targets are sub-token. Replaced with the
attention RANK of the single grid cell containing the GT centre, always defined, **chance = 0.500
exactly by construction** (a uniformly random ranking puts the GT cell at a uniform percentile).

| layer | median gt_pct | p10 | P(top-1 cell inside GT) |
|---|---|---|---|
| L2 | 0.289 | 0.072 | **0.0%** |
| L4 | 0.167 | 0.059 | **0.0%** |
| L8 | **0.157** | 0.041 | **0.0%** |

**Two facts that have to be stated together.** The GT cell ranks around the 16th percentile of ~294
tokens against a chance of 50 -- the signal transferred from Phase 22's RePOPE setting to V*Bench.
And the argmax is **never** the target: 0/191 at every layer (chance ~1/925, so ~0.2 expected --
0 is consistent with chance). The top of the ranking is not the evidence. Any read-out that trusts
the peak, or the extremes of the top-k, is reading sinks.

### The size gradient replicates Phase 22 on a different benchmark with a different statistic
| GT size (merged tokens) | n | median gt_pct |
|---|---|---|
| 0.01 - 0.09 | 47 | **0.082** |
| 0.09 - 0.32 | 48 | 0.141 |
| 0.32 - 1.25 | 48 | 0.153 |
| 1.31 - 35.20 | 48 | 0.347 |

Attention ranks the SMALLEST targets HIGHEST -- the same direction Phase 22 measured via retention
(+34.8pp at the smallest stratum, decaying to +3.1pp at the largest). Two benchmarks, two
statistics, same gradient. (Mechanically sensible: for a large object the GT *centre* cell is one of
many on the object and attention may sit elsewhere on it; for a sub-token object there is one cell.)

### The repair that failed, and the one that has not been tested yet
Percentile box (10th-90th of top-k coordinates instead of min/max): median area 0.929 vs 1.000,
headroom 1.08 vs 1.00, still **0.0%** of items under 25% area. Robustifying the geometry is not
enough.

**The sinks are positionally stable and sit at the image corners.** Pooling selected positions onto
a normalised 7x7 grid over 143 images:

| position | L2 | L4 | L8 |
|---|---|---|---|
| (0,6) top-right | 5.5x | 6.3x | 5.1x |
| (0,0) top-left | 4.4x | 5.7x | 4.7x |
| (6,6) bottom-right | 4.2x | 3.8x | 3.9x |
| top-5 positions' share of all selected tokens | 40.8% | 44.6% | 41.4% |

Uniform would be 10.2%. A content-independent, position-fixed background is **subtractable without
training**, and that must be tested before any learned read-out is justified. Phase 30c dumps the
full attention map once so every candidate read-out becomes an offline experiment.

**Leakage caveat, attached now so it travels with the result:** a background estimated from these
same 191 items uses the test set to build its normaliser. Acceptable for establishing WHETHER a fix
exists; if it works, re-estimate the background leave-one-out before quoting any accuracy.

## §4V. Layer choice dominates read-out choice for localization (curve NON-MONOTONE, peak not yet established)

Phase 30d, offline on the Phase 30c attention dumps, n=191, Qwen3-VL-2B, V*Bench, B0=300. No
forward passes. Background normalisation is **leave-one-out by construction** (the background for
item i is estimated from every item except i), so no number here uses the test set to build its own
normaliser.

| layer | raw gt_pct | best read-out gt_pct | argmax inside GT |
|---|---|---|---|
| L2 (**FastV's choice**) | 0.289 | 0.200 | 3.7% |
| L4 | 0.167 | 0.065 | 2.1% |
| L8 | 0.157 | 0.112 | 3.1% |
| L14 | 0.276 | 0.184 | 1.0% |
| **L20** | **0.068** | **0.041** | **10.5%** |

Chance: gt_pct 0.500 exactly; argmax-inside-GT 0.108% (~1 in 925). So **L20's 10.5% is ~97x chance**,
and the GT cell sits at the 4th percentile of ~294 tokens.

**The dominant variable is the layer, not the read-out.** Positional-background removal helps
(L20: 0.068 -> 0.041, argmax-in-GT 6.8% -> 8.4%; masking the border ring reaches 10.5%), but the
layer choice moves the statistic far more: L2 -> L20 is 0.289 -> 0.068 on raw attention alone.

**The curve is NOT monotone in depth, so "late layers are better" is NOT what this measures.**
L2 0.289 -> L4 0.167 -> L8 0.157 -> **L14 0.276** -> L20 0.068: L14 is worse than both L4 and L8. A
signal improving with depth does not do that. Either several of these gaps are noise at n=191, or
something structural is happening -- Qwen3-VL injects visual features at multiple layers via
deepstack, which is the same mechanism that made the FastV reimplementation unsafe to ship. **The
honest statement is that L20 is the best of five SAMPLED layers and the curve is non-monotone.**
The all-28-layer sweep is running to settle it; the test is whether L19 and L21 look like L20, since
a real effect has neighbours and an outlier does not.

**PRIMARY METRIC IS `gt_pct`**, not argmax-in-GT: it is always defined, its chance level is exactly
0.500 by construction, and it does not depend on a window size chosen by hand. argmax-in-GT and the
"near" window are secondary and must not be quoted as the headline.

### A HYPOTHESIS this is consistent with -- explicitly not a demonstrated explanation
What follows is NOT established by the measurement above and must be written as a conjecture. What
was measured is the GT-cell RANK at layer 2 versus layer 20. That is not the same as showing layer-2
attention is a bad PRUNING criterion: FastV prunes at layer 2 and the surviving tokens are still
processed by the remaining 26 layers, so a criterion can be a poor localizer and an adequate
keep-rule. (This is the same shape of overclaim as the earlier "pruning cannot recover what was
never encoded", which was false and was retracted. Do not repeat it.)

With that stated: attention-guided **pruning** must decide EARLY -- that is the entire point, since the saving comes
from dropping tokens before the expensive layers run. FastV reads layer 2. But the localization
signal at layer 2 is the weakest of any layer measured, and the argmax there is never the target.
**The layer where the evidence is identifiable is the layer at which pruning has nothing left to
save.** That is a structural tension in the pruning family, and it is a mechanism for the ACL 2502.11501
observation (attention-guided pruning beaten by random selection) that does not require attention to
be uninformative -- it only requires the information to arrive too late to be useful *for pruning*.

**Allocation genuinely does not have this constraint**, and THAT part is not a conjecture: the cheap
pass runs to completion and the SECOND pass is what spends the budget, so the proposer may read any
layer, including the last. Whether early-layer attention is thereby a worse pruning criterion is the
open question; that allocation is free to read the best layer is a fact about the two designs.

CAVEAT, and the reason 30c was re-run over all 28 layers: L20 was the LAST layer sampled in the
5-layer sweep {2,4,8,14,20}. The peak may lie beyond it, so the layer curve is being re-measured at
full resolution before any layer is named in the paper.

**SCOPE OF 4S-4V, stated so the bar is not mistaken for something already tested:** every number in
Phase 30a-30d is **localization quality**, not task accuracy. No two-pass answering arm has been run.
The pre-registered 72.9% / 44%-oracle-capture bar in PLAN.md therefore **cannot be evaluated from any
data currently in hand** -- it requires the full arm matrix, which is gated on a read-out clearing
the viability rule first.

## §4W. A selection-free localizer: block-mean late attention, background-normalised

Phase 30d, offline, n=191, Qwen3-VL-2B, V*Bench, B0=300. All 28 layers dumped; no forward passes.

### The layer curve is a REGIME SHIFT at ~L16, not a depth trend
Median `gt_pct` (chance 0.500 exactly, lower better), raw attention:

| layers | range of median gt_pct |
|---|---|
| L0-L15 | 0.153 - 0.398 |
| **L16-L26** | **0.027 - 0.119** |
| L27 (final layer) | 0.408 -- collapses |

Every layer in L16-L26 beats every layer in L0-L15. Within the late block it is jagged (L17=0.027
but L18=0.139), so **no single layer may be named as "best"**: the argmin of 28 noisy statistics at
n=191 is a test-set selection, and L17 being the winner is exactly that.

### The selection-free read-out matches the cherry-picked one
Mean of per-layer normalised attention over the whole L16-L26 block -- a block chosen by the regime
boundary, not by outcome -- then divided by the leave-one-out positional background:

| read-out | med gt_pct | p10 | argmax in GT | near (+-1.5 cell) |
|---|---|---|---|---|
| **block L16-L26 + loo_norm** | **0.034** | 0.003 | **17.3%** | **42.9%** |
| block L16-L26 + no_border | 0.037 | 0.003 | 16.2% | 41.9% |
| block L0-L15 + loo_norm (control) | 0.156 | 0.014 | 1.6% | 7.3% |
| L2 + loo_norm (FastV's layer) | 0.253 | 0.023 | 2.6% | 6.3% |
| L17 alone (**selected on test set**) | 0.029 | 0.003 | 18.8% | 44.5% |
| **chance** | **0.500** | | **0.11%** | **3.0%** |

**argmax-in-GT 17.3% is ~157x chance**, and `near` 42.9% is ~14x chance. The block mean recovers
essentially all of the cherry-picked layer's performance (0.034 vs 0.029) with **no layer selected
on the data**, so the result does not depend on a choice made after seeing outcomes.

Note what the peak metric is being asked to do: land inside a GT box that is **sub-token for 71% of
items**. 17.3% under that constraint is a strong localizer, not a marginal one.

### What this changes about the proposal mechanism
30a's read-out derived the region's SIZE from the spread of the top-k, which the sinks destroyed.
A peak-based proposer does not: it takes the argmax cell and crops a window of a chosen size around
it, so the proposal area is a design parameter (set to match the oracle crop's budget) rather than
an outlier-sensitive statistic. This is why the 30a negative does not carry over -- **the failure
was the size estimator, not the localizer.**

### STILL NOT TESTED: accuracy
Every number above is localization quality. The two-pass answering arm has not been run, so the
pre-registered 72.9% / 44%-oracle-capture bar remains un-evaluated. That is Phase 31.

## §4X. Phase 31 -- THE METHOD DOES NOT EARN ITS SECOND FORWARD PASS (pre-registered negative)

n=191, Qwen3-VL-2B, V*Bench, B0=300. **Budget gate: all 10 arms within 2.0% of B0** (nine of ten at
median 294 tokens), so every contrast is matched, not matched-on-average.

Method: one pass at B0 -> mean attention over layers 16-26 -> mask outer ring -> argmax -> square
window of side W -> re-render at the SAME budget -> second pass. Training-free, data-free.
W chosen by **cross-category holdout** (W for `direct_attributes` picked on `relative_position` and
vice versa), so no hyperparameter is fitted on its own test items.

### Headline
| arm | accuracy |
|---|---|
| **held-out attn_prop** | **60.2%** |
| uniform@300 (baseline) | 56.5% |
| same-window RANDOM centre | 37.7% |
| oracle@300 | 93.2% |

    attn - uniform : +3.7pp   CI [-4.2, +11.5]   <- CI INCLUDES ZERO
    attn - random  : +22.5pp  CI [+13.6, +31.9]  <- excludes zero decisively
    ORACLE CAPTURE : 10.0%    (published training-free proposer: 31-44%)

| bar | value | result |
|---|---|---|
| uniform@300 -- beats doing nothing | 56.5% | PASS (but CI includes zero) |
| **uniform@600 -- compute-matched** | **66.0%** | **FAIL** |
| 44% oracle capture -- published proposer | 72.9% | **FAIL** |

**VERDICT, as pre-registered: NEGATIVE.** The method beats the naive baseline but loses to spending
the same two-pass compute on plain resolution. The second forward pass is not earned.

### What nonetheless IS established: the localizer works
+22.5pp over a same-size window at a RANDOM centre, CI [+13.6, +31.9]. The gain is attributable to
*where* the window was placed, not to the magnification that cropping provides -- the placement null
was the control designed to separate exactly these, and it separates them cleanly. 4W's localization
result converts into accuracy; it just does not convert into ENOUGH accuracy.

### Where it helps and where it HURTS (POST-HOC -- reported as such, not selected on)
| split | attn | uniform | delta | CI | capture |
|---|---|---|---|---|---|
| `direct_attributes` (n=115) | 60.9% | 49.6% | **+11.3pp** | [+1.7, +20.9] | 23.6% |
| `relative_position` (n=76) | 59.2% | 67.1% | **-7.9pp** | [-21.1, +5.3] | **-40.0%** |

| GT area quartile | delta vs uniform | CI |
|---|---|---|
| 0.00002-0.00030 (smallest) | **+19.1pp** | [+2.1, +36.2] |
| 0.00031-0.00105 | +2.1pp | [-12.5, +16.7] |
| 0.00108-0.00417 | +4.2pp | [-10.4, +18.8] |
| 0.00436-0.11972 (largest) | **-10.4pp** | [-27.1, +6.2] |

**The split has a mechanism, and it is the same one that killed region packing (4R).** Cropping to a
window around a SINGLE attention peak discards inter-object layout. A relational question ("is the
umbrella left or right of the traffic light?") needs both objects and their arrangement, so the crop
destroys the evidence it is meant to concentrate -- hence -7.9pp. An attribute question needs one
object at resolution, so the same operation gains +11.3pp. Likewise by size: the method wins where
the target is smallest (+19.1pp, CI excludes zero) and loses where the target is already large
enough to resolve (-10.4pp), because there the crop only removes context.

**The headline stays the pooled held-out negative.** These splits are post-hoc and are a hypothesis
about when allocation applies, not a method claim. A conditional allocator -- apply only to
non-relational, small-target items -- is the obvious next design, but it needs a question-type and
target-size predictor that does not exist here, and selecting the favourable split after seeing
these numbers would be precisely the error this phase's holdout design was built to avoid.

## §4Y. Phase 32 -- the CONDITIONAL allocator earns its second forward pass (bounded positive)

n=191, Qwen3-VL-2B, V*Bench, B0=300. **Budget gate: all 10 arms within 2.0%.**

Phase 31's fixed policy was a negative (60.2% vs the 66.0% compute-matched bar) because it applied
allocation to every item, including those where cropping destroys the evidence. Phase 32 adds the
decision the fixed policy lacked: **whether** to reallocate.

Offline ceiling, computed before building anything: perfect per-item choice between `uniform` and
`attn` = **74.3%**, i.e. above both bars. The two arms agree on 130/191, so a gate can act on only
the **61 disagreements**; the fixed policy already wins 34 of them.

### Result (W selected by 5-fold CV on train folds only; W=0.15 chosen in every fold, sd 0.00)
| gate | CV acc | vs uniform@300 | oracle capture | vs 66.0 |
|---|---|---|---|---|
| **text+conf** | **68.6%** | **+12.0pp** CI [+6.8,+17.3] | **32.9%** | **PASS** |
| conf_route only | 66.5% | +10.0pp CI [+3.7,+17.3] | 27.2% | PASS (marginal) |
| text_gate only | 65.0% | +8.5pp CI [+3.7,+15.2] | 23.1% | FAIL |
| `conf_thresh` (FITTED) | 59.7% | +3.1pp CI [-4.7,+11.0] | -- | FAIL |
| `learned_LR` (FITTED) | 57.1% | +0.5pp CI [+0.0,+1.6] | -- | FAIL |

    uniform@300 56.5%  |  best FIXED policy 62.8%  |  bar 66.0%  |  published 72.9%  |  oracle 93.2%

**VERDICT: EARNS ITS COMPUTE.** 68.6% clears the compute-matched control (66.0%) with a CI on the
gain that excludes zero, and **oracle capture 32.9% lands inside the published training-free
proposer's 31-44% range** -- with no training and no auxiliary model. It does NOT clear 72.9%, so
this is a bounded positive, not a method win.

The gate is two rules, neither with a fitted parameter:
    1. if the QUESTION is relational (keyword), do not reallocate -- cropping destroys layout (4R/4X)
    2. otherwise, answer from whichever PASS is more confident

### The training-free gates BEAT the learned ones, and that is the interesting part
`learned_LR` (logistic regression over confidence + attention statistics + relationality, 5-fold CV)
reaches **57.1%** -- barely above `uniform` and far below the two-rule gate. `conf_thresh` reaches
59.7%. At n=191 with ~61 actionable items there is not enough signal to fit a gate; the unfitted
rules win precisely because they have nothing to overfit. **This is evidence against the learned
allocator at this data scale**, and it is the honest answer to "should this be learned?": not here.

### THE MAIN THREAT TO THIS RESULT, stated plainly
The relational keyword gate **agrees with the V*Bench `category` annotation on 189/191 items (99%)**.
It reads the question string, which is a legitimate inference-time input, so it is not label
leakage. But V*Bench's two categories are almost perfectly keyword-separable, so the gate is doing
approximately what the annotation would do, and **there is no evidence it transfers to a benchmark
whose question types are not so cleanly split.** Report the 99% figure alongside the accuracy every
time. `conf_route` alone (66.5%) is the version that carries no such dependency, and it passes the
bar only marginally.

### Second caveat: the result depends on W
Without any selection, reporting all four windows: `text+conf` spans **63.9%-68.6%**, and the worst
case FAILS the bar. CV selects W=0.15 in every fold, which is why the headline stands, but a reader
should know the spread.

## §4Z. RETRACTION: the Phase 24 `obj` probe result is an ROI-SELECTION ARTIFACT

Phase 24 reported `obj` peak AUROC **1.000** on the wrong-answer cohort and **0.999** on
sub-token-and-wrong, read in §4 as "the target is encoded at its own visual tokens even when the
model errs." **That reading is retracted.**

### The confound (phase24_probe_layer_sweep.py:190-196)
```
if group == "positive":   roi = GT bounding box of the real annotated object
else:                     roi = a UNIFORMLY RANDOM rectangle, ~U(5%,30%) per side
```
The `obj` block pools hidden states over **a GT object region for positives** and **a random
rectangle for negatives**. A probe need not represent the queried object at all -- only answer
"annotated object region, or random rectangle?" That is query-independent, which is exactly why the
wrong-answer cohort scored as high as the correct one.

**The existing `rand` control could not catch this.** For a positive, obj=GT box and rand=random box;
for a NEGATIVE, obj=random box and rand=random box -- same distribution. `rand` controls for "is this
location special", not for the positive/negative ROI-type asymmetry, which exists only in `obj`.

### Controls (offline, no forward passes; saved feature bank)
**C1 region-type probe** -- positives only, classify `obj` vs `rand` vectors (same image, same
forward pass, only region type differs):

| L0 | L4 | L8 | L12 | L16 | L20 | L24 | L26 | L28 |
|---|---|---|---|---|---|---|---|---|
| 0.967 | 0.968 | **0.971** | 0.964 | 0.966 | 0.965 | 0.967 | 0.970 | 0.969 |

**~0.97, FLAT ACROSS DEPTH, INCLUDING LAYER 0.** At L0 the LLM has computed nothing, so region type
is separable from the raw visual embeddings. The shortcut is fully available and is not a learned
semantic property.

**C2 ROI-matched presence** -- `rand` block for BOTH classes, so ROI distribution is identical:

| cohort | obj (GT vs random ROI) | **rand (ROI-matched)** | perm null |
|---|---|---|---|
| all items (n=1113) | 0.976 | **0.599** | 0.532 |
| wrong (n=195) | 0.994 | **0.742** | 0.600 |

**C3 geometry-only** (no hidden states): all items **0.629**, wrong 0.408. The shortcut is available
*before the model runs* on the full sample; it does not explain the wrong-cohort residual.

### What actually stands
1. **The 1.000 is region-type decoding, not object encoding.** Retract it everywhere.
2. **A residual survives ROI matching but is small**: 0.742 vs null 0.600 on the wrong cohort
   (+0.142); 0.599 vs 0.532 on all items (+0.067).
3. **The residual does not support the original claim.** The `rand` block reads a RANDOM region for
   both classes, so what survives is "presence is weakly decodable from ARBITRARY image tokens"
   -- global scene context -- **not** "the target is encoded at its own visual tokens." Those are
   different claims and only the weaker one has evidence.
4. **Treat the residual as provisional**: n=195, null 0.600, and its peak is at **L0** (0.742), which
   is not where a semantic representation should live and is consistent with noise at this n.

### Consequence for the paper
§4's "encoded, located, still unusable" cannot rest on this probe. The Phase 26 half -- evidence does
not reach the answer position on failures -- is a **separate experiment and is unaffected**, so the
*routing* claim survives; the *encoding* claim does not. Any future presence probe needs an ROI that
does not depend on the label, which **POPE cannot supply**, since absent objects have no box. That is
a design problem, not a control problem.

**Process note:** this is the second circularity-class defect in the same phase (the first was the
`last` block decoding the model's own answer, §4L). Both produced near-ceiling AUROCs that looked
like findings. The rule this earns: **an AUROC at or near 1.000 is a bug report until a matched
control says otherwise.**

## §5A. Phase 33 — TRANSFER TEST on HR-Bench 4k: the confidence rule transfers, the text rule does NOT

n=800 rows / 200 instances, Qwen3-VL-2B, HR-Bench 4k (4032x4032 images), B0=300.
**Nothing refitted**: W=0.15 and both gate rules carried over from V*Bench unchanged. Budget gate
passes on all arms (<=4.0% off). CircularEval complete: 200/200 instances with all 4 permutations.

**The §6.2 confound is genuinely broken here**: keyword-gate concordance with the benchmark's own
`category` label is **53.5%** (chance) vs **99.0%** on V*Bench. So this is a real transfer test.

| arm | per-row | vs uniform | CircularEval | vs uniform |
|---|---|---|---|---|
| uniform@300 | 52.5% | — | 36.5% | — |
| **uniform@600** (compute-matched) | **59.2%** | +6.8pp | **45.5%** | +9.0pp |
| always attn@0.15 | 43.8% | **-8.8pp** | 27.5% | -9.0pp |
| **conf_route@0.15** | **56.1%** | **+3.6pp** CI[+1.0,+6.2] | **44.5%** | **+8.0pp** |
| text_gate@0.15 | 45.0% | **-7.5pp** | 27.0% | -9.5pp |
| text+conf (the V*Bench gate) | 53.4% | +0.9pp CI[-1.2,+3.0] | 39.0% | +2.5pp |

**VERDICT (pre-registered): PARTIAL TRANSFER.** The combined gate beats the baseline but loses to
the compute-matched control (-5.9pp, CI [-9.0,-2.8]).

### The decomposition is the finding
* **`conf_route` TRANSFERS.** +3.6pp per-row (CI excludes zero) and **+8.0pp on CircularEval**,
  nearly matching `uniform@600` (44.5% vs 45.5%) at the same two-pass cost. The confidence rule is
  portable.
* **The text rule DOES NOT, and it is actively harmful.** Alone: -7.5pp. Combined: it drags
  `conf_route` from 56.1% down to **53.4%**. On V*Bench it was the stronger half. **§6's headline
  gate is the wrong composition off V*Bench**; `conf_route` alone is the portable method.

### The predicted mechanism was WRONG, and the diagnostic refutes it
The pre-registered guess was that HR-Bench's *referential* prepositions ("the number **above** the
entrance") would make the keyword rule skip items allocation HELPS. Measured instead:

    rows where rule FIRED (n=180):     allocation would have scored  -5.6pp vs uniform
    rows where it did NOT fire (n=620): allocation would have scored  -9.7pp vs uniform

Both negative. Allocation hurts **everywhere** on HR-Bench, and the rule is *backwards* -- it skips
the items where allocation is least harmful. The referential-preposition story is not what happened.

### The bigger qualification: the exchange rate may be V*Bench-specific
On V*Bench, budget saturates and placement dominates (no crossing at 26.5x). On HR-Bench 4k the
budget axis is **far from saturated**: merely doubling tokens gives **+6.8pp per-row / +9.0pp
circular**, while our allocator gives **-8.8pp**. By category: on `single` the gate gains +4.5pp over
uniform (58.0 vs 53.5) but `uniform@600` reaches 65.0; on `cross` the gate loses (-2.7pp), consistent
with the layout mechanism (4R/4X).

**Every architecture result in Phases 27-28 was measured on V*Bench alone.** This is the first
evidence that the exchange rate does not hold on a second benchmark, and §2's headline needs that
scope attached.

### WHAT THIS CANNOT DISTINGUISH -- the missing control
HR-Bench ships **no bounding boxes**, so there is **no oracle crop arm** and no random-crop arm was
run. Therefore we cannot separate:
   (a) allocation genuinely does not pay on HR-Bench (budget is the better axis at 4K), from
   (b) our *localizer* fails on 4032x4032 images, so the crops miss the evidence.
With W=0.15 a crop is ~605x605 px of a 4032x4032 source; a localizer miss loses everything. **The
honest claim is "our proposer does not transfer", NOT "allocation does not transfer."** The
discriminating experiment is a `rand@0.15` arm on the same items: if `attn ~= rand`, the localizer
failed; if `attn > rand` but both below uniform, allocation itself is the wrong move at this scale.

## §5B. Phase 33b — THE CONTROL RESOLVES IT: the localizer transfers, allocation does not

n=800 rows merged with Phase 33 by `row_id` (labels cross-checked), same instances, same B0, same W,
same pipeline; only the crop CENTRE changes. Budget gate: rand@0.15 median 289 tok, 3.7% off. The
centre is drawn once per instance, mirroring how the attention peak is computed once per instance.

| arm | per-row | CircularEval |
|---|---|---|
| uniform@300 | 52.5% | 36.5% |
| uniform@600 (compute-matched) | 59.2% | 45.5% |
| **attn@0.15** | **43.8%** | **27.5%** |
| **rand@0.15** | **30.1%** | **12.5%** |

    attn - rand    = +13.6pp  CI [+10.0, +17.2]   <- excludes zero
    attn - uniform =  -8.8pp  CI [-12.9,  -4.6]
    rand - uniform = -22.4pp  CI [-26.4, -18.4]
    (V*Bench, same contrast: attn - rand = +22.5pp CI [+13.6,+31.9])

### VERDICT (pre-registered): the localizer WORKS; allocation is the wrong move at this scale
The pre-registered reading `attn > rand with both below uniform` fired. This **rules out the
localizer-failure explanation** that Phase 33 could not exclude:

1. **The localizer TRANSFERS.** +13.6pp over random placement on 4032x4032 images, on a benchmark it
   was never tuned on, with a CI excluding zero. Same direction as V*Bench (+22.5pp), somewhat
   weaker. This is now a **two-benchmark** result and is the most portable component of the method.
2. **Allocation still loses (-8.8pp).** The crop finds the target often enough to beat random by a
   wide margin, but not often enough to beat seeing the whole image. At W=0.15 a crop is ~605px of a
   4032px source: **44x magnification at the cost of discarding 97.75% of the image.**

### The boundary is now a measured claim, not an uncertainty
| | V*Bench | HR-Bench 4k |
|---|---|---|
| target size at B0=300 | **71% sub-token** | larger / context-dependent |
| doubling the budget buys | +9.5pp | **+6.8pp (per-row), +9.0pp (circular)** |
| allocation at matched budget | **+37.2pp (oracle)** | **-8.8pp** |
| localizer vs random placement | +22.5pp | **+13.6pp** |

**Allocation pays when the evidence is smaller than the budget can resolve; it costs when it is not.**
On V*Bench the target is sub-token, so magnification is the binding constraint and placement wins by
26x. On HR-Bench 4k the target is resolvable enough at B0 that discarded context dominates, and the
budget axis wins. The localizer is good in both regimes; what changes is whether acting on it helps.

**This converts §5.4 from an open control into the paper's boundary condition**, and it means
"allocation does not transfer" is now *supported* — but specifically as "reallocating at a FIXED
budget does not pay at 4K", not as "the attention localizer fails", which is refuted here.

### Design implication (not tested)
W was transferred at 0.15 without adaptation. A crop of 0.15 side is ~337px on V*Bench's ~2250px
sources but ~605px on HR-Bench's 4032px sources -- absolutely larger, relatively identical. Since the
localizer's precision is what sets the usable window, a W that scales with localizer uncertainty
rather than a constant fraction is the obvious next design. Untested; recorded as a hypothesis.

---

## §6A  The sink is COLUMNAR, not cornered — and it is a serialization artifact (Phase 34)

Re-measured the Phase 30b sink, stratified by grid shape. The earlier "attention sinks sit at the
image corners" reading was **wrong**, and wrong in a way that bears on the method.

| region | attention mass | area | enrichment |
|---|---|---|---|
| **last column (col = gw−1)** | 25.3% | 5.0% | **5.1×** |
| first column (col = 0) | 14.2% | 5.0% | 2.9× |
| top row | 5.5% | 6.3% | **0.9×** |
| bottom row | 5.3% | 6.3% | **0.8×** |
| interior | 49.7% | 77.5% | 0.6× |

Rows carry **no excess mass at all**. "Corners" was the intersection of a strong column effect with
a null row effect. Per token, the last column draws 17.19 (×10⁻³) against ~2.1 in the interior — an
8× ratio — and the profile across interior columns is flat, not a rightward gradient, so the effect
is a step at the row boundary rather than a spatial bias.

Two further tests fix the mechanism:

- **Transpose test** (14×21 vs 21×14, *identical* n = 294, geometry transposed). Sink positions are
  the same in (row, col) space — col = gw−1 in both — but only 4/8 *absolute* indices coincide. A
  register token would reuse the same absolute slots. It does not.
- **Interior replay.** Modal-grid sink indices, replayed on other grid shapes where they land in the
  interior, re-fire at **1.1% against 3.3% chance** — *below* chance. Not fixed slots.
- **Onset.** Corner/column *mass* is already **39.9% at L0** and varies with depth thereafter
  (41.1% L2, 24.1% L12, 10.4% L14, 21.4% L26). The effect is present in the first layer, not built
  up with depth. (The top-10 corner *count* saturates at 40% — all four corners inside ten slots —
  so it is flat by construction and cannot show an onset; the mass series is what carries this.)

This is the signature of **raster serialization**: col gw−1 is the token before a row wrap and col 0
the token after one, while the top and bottom rows are ordinary raster positions and duly show
nothing. The "spatial" sink is a 1-D sequence artifact wearing a 2-D costume.

### The method consequence, tested and partly negative

The deployed localizer masks the whole outer ring. If the mechanism is columnar, the top/bottom rows
are being discarded for nothing. Scored against **area-matched random twins** (10 seeds), which is
required because `gt_pct` is a rank and deleting cells improves it mechanically:

| mask | cells deleted | argmax-in-GT | random twin | sink-removal effect | pp per % deleted |
|---|---|---|---|---|---|
| ring (deployed) | 22.5% | 15.7% | 4.5% | +11.2pp | 0.50 |
| **columns only** | **9.9%** | 14.7% | 4.9% | **+9.8pp** | **0.99** |
| rows only | 13.9% | 8.9% | 4.8% | +4.1pp | 0.29 |

Columns deliver **88% of the effect at 44% of the cost** — 3.4× more efficient per cell than rows.
But column-only does **not** beat the ring mask outright (gt_pct Δ +0.023, CI [+0.006,+0.037] in the
ring's favour; argmax-in-GT Δ −1.1pp, n.s.). **Honest reading: the mechanism claim is confirmed and
quantified; the method improvement is not.** Masking rows still buys a little, most of it competitor
thinning rather than sink removal. The n=3 subgroup whose GT centre lies in a masked row is far too
small to support anything and is not used.

---

## §6B  RePOPE within-benchmark: the exchange rate reproduces; the crossing does not (Phase 21)

Phase 21's stratified RePOPE-clean run (600 positives in five target-size strata + 200 negatives)
was collected but never analysed. Budget gate passes (spread 1.7%). **Yes/no read-out, chance 50% —
these accuracies are not comparable to the 4-way MCQ numbers elsewhere.**

| target size | n | median tok | uniform | alloc_query | Δ | 95% CI | vs random |
|---|---|---|---|---|---|---|---|
| <0.5 tok | 120 | 0.26 | 57.5% | 74.2% | **+16.7** | [+7.5,+25.8] | +15.0 |
| 0.5–2 tok | 120 | 1.03 | 89.2% | 95.8% | +6.7 | [+0.8,+12.5] | +13.3 |
| 2–8 tok | 120 | 4.19 | 98.3% | 100.0% | +1.7 | [+0.0,+4.2] | ceiling |
| 8–32 tok | 120 | 18.14 | 97.5% | 98.3% | +0.8 | [+0.0,+2.5] | ceiling |
| >32 tok | 120 | 104.86 | 99.2% | 98.3% | −0.8 | [−3.3,+1.7] | ceiling |

**Positive half confirmed**: monotone decay over a 400× size range, and **the false-positive rate is
unchanged (7.5% → 7.5%)**, so the gain is not yes-bias. This is §2's exchange rate reproduced on a
second benchmark with a different task format, different images, and a different annotation source.

**Negative half NOT testable here**: `uniform` saturates at 97.5–99.2% on every resolvable stratum.
No crossing can be observed where there is no headroom. Reported as uninformative, not as support.

---

## §6C  ⚠⚠ SUPERSEDED BY §6D — region count sets the SIGN (Phase 35)

> **Status: superseded.** The `oracle` arm (Phase 36) refutes the causal reading below: a perfect
> crop to the GT box WINS on `relative_position` (+19.7pp [+5.3,+34.2]). If two regions were
> intrinsically un-croppable a perfect crop would fail too. The *measurements* here stand and are
> still used; the *explanation* is replaced by evidence-set coverage in §6D, which subsumes it.

§5.5 stated the boundary as target size. That is **wrong as stated**, and HR-Bench breaks it: at
B₀ = 300 a 4032×4032 image gives ~233 px per merged token, so HR-Bench is *deeply* sub-token and the
size predictor says allocation should win by a wide margin there. It **lost by 8.8pp**.

Splitting HR-Bench by its own category — both categories sub-token, so size is held fixed — gives
the discriminating cell:

| HR-Bench 4k | n | uniform | attn@0.15 | Δ | 95% CI | circular Δ |
|---|---|---|---|---|---|---|
| single (one region) | 400 | 53.5% | 51.7% | −1.8 | [−7.5,+4.0] | +2.0pp |
| **cross (two regions)** | 400 | 51.5% | 35.8% | **−15.8** | **[−21.5,−10.0]** | −20.0pp |

The **localizer is not the failure**: attn beats random placement in *both* categories (single
+20.5pp [+15.5,+25.8]; cross +6.8pp [+1.8,+11.5]). The proposal finds a target in both cases. The
crop then **destroys the second region**, which only matters when the question needs it.

V\*Bench shows the same flip: direct_attributes **+13.9pp**, relative_position **−9.2pp**.

**Decoupling test** (V\*Bench, median split at 0.32 merged tok — size held fixed within stratum):

| | sub-token (<0.32 tok) | resolvable (≥0.32 tok) |
|---|---|---|
| **single-region** (direct_attributes) | **+18.8** [+6.2,+31.2] | +2.9 [−11.4,+17.1] |
| **multi-region** (relative_position) | **−20.0** [−46.7,+6.7] | −6.6 [−21.3,+8.2] |

The sign flip **survives size control**. Median target size is 0.18 tok for direct_attributes vs
1.40 for relative_position, so the two factors are correlated on V\*Bench — but they dissociate on
HR-Bench, where both categories are sub-token and the sign still flips.

**Restated boundary condition:**
> Reallocation pays when the answer's evidence is confined to **one region**, and costs when it
> requires **comparing two** — independently of target size, image scale, model, or task format.
> Within a sign, **target size sets the magnitude**: the smaller the evidence relative to what the
> budget can resolve, the larger the effect.

All four cells of the 2×2 are populated and consistent across three benchmarks (V\*Bench 1500px MCQ,
HR-Bench 4032px MCQ, RePOPE COCO yes/no). The `relative_position` × sub-token cell has n=15 and a CI
that crosses zero; it is the weakest cell and is carried by the HR-Bench `cross` replication.

**This also dissolves the §6.2 threat.** The relational-keyword gate was flagged as possibly being
the V\*Bench `category` annotation in disguise (99% concordance). It now reads the other way: on
V\*Bench, `category` *is* region count, so the keyword rule was an imperfect **region-count
detector** all along — which is exactly the quantity the boundary condition names. Its failure on
HR-Bench (53.5% concordance, −7.5pp) is then a *detector* failure, not a refutation of the rule's
target, and names the concrete next step: a better region-count detector.

---

## §6D  ★ THE MECHANISM: evidence-set COVERAGE is the mediating variable (Phase 36–38)

> ⚠ **Read with §6F and §6G.** This section originally said *causal*. §6F retracts that: the
> interior-optimum reading below is withdrawn (every CI crosses zero) and the exogenous instrument
> is underpowered, not supporting. §6G then supplies genuine interventional evidence, but only as a
> dose-response — the population-level effect is null. Coverage is a mediator with a dose-responsive
> intervention behind it, not a demonstrated population-level cause.

Two candidate boundaries each failed their own decisive test:

- **Target size** (§5.5) — HR-Bench is deeply sub-token (~233 px per merged token at B₀=300) so the
  predictor says allocation should win big. It **lost by 8.8pp**. Backwards on its key case.
- **Region count** (§6C) — the `oracle` arm, a crop to the GT box and thus a *perfect* single-region
  proposal, **wins** on relative_position (+19.7pp [+5.3,+34.2]). A perfect crop of one region does
  not fail, so "two regions are un-croppable" is false.

What the oracle arm shows instead: the GT box for a relational question is the **union** of the
objects involved — median area 0.0048 vs 0.0006 for direct_attributes (**7.9×**), aspect 2.09 vs
1.68. Cropping to it keeps the whole evidence set. Cropping to one attention peak does not:

| | fraction of GT box covered by the deployed W=0.15 window |
|---|---|
| direct_attributes | 52.6% (P(coverage=0) = 45.2%) |
| relative_position | 22.4% (P(coverage=0) = 57.9%) |

**The boundary is neither size nor region count. Both are causes of one quantity: the fraction of
the evidence set that the reallocated window actually covers.**

### The dose–response (categories POOLED, no size or region term)

| coverage of GT by the deployed window | n | uniform | attn@0.15 | Δ | 95% CI |
|---|---|---|---|---|---|
| **0% (missed entirely)** | 96 | 52.1% | 36.5% | **−15.6** | [−26.0,−5.2] |
| 0–25% | 11 | 63.6% | 36.4% | −27.3 | [−63.6,+9.1] |
| 25–75% | 12 | 75.0% | 100.0% | +25.0 | [+0.0,+50.0] |
| 75–100% | 10 | 60.0% | 70.0% | +10.0 | [−30.0,+50.0] |
| **100% (full)** | 62 | 58.1% | 95.2% | **+37.1** | [+24.2,+50.0] |

Monotone, **crossing zero at ~25% coverage**, with no category label used. **50% of items (96/191)
have zero coverage** — the window misses completely. The headline +12.0pp is a large win on covered
items netted against a real loss on missed ones. Within the low-coverage bin the two categories
agree (−19.2 vs −14.5), as a mediator requires; within the high bin they do not (+41.3 vs +4.8,
n=21), so coverage is the dominant term but **not the whole story**, and that is stated as such.

### Is coverage causal, or does attention just look at easy items? (Phase 38)

> ⚠ Instrument 2 below is **retracted** and instrument 3 is **inconclusive** — see §6F. Only
> instrument 1 (difficulty held fixed) stands, and it is mediation, not intervention.

1. **Difficulty held fixed — the strongest.** Stratifying on `uniform` correctness:

   | uniform was | window | n | attn@0.15 accuracy |
   |---|---|---|---|
   | WRONG | missed (0%) | 46 | **13.0%** [4.3,23.9] — *below* 25% chance |
   | WRONG | covered (>0) | 37 | **83.8%** [70.3,94.6] |
   | CORRECT | missed (0%) | 50 | 58.0% [44.0,72.0] — destroys 42% of what worked |
   | CORRECT | covered (>0) | 58 | 87.9% [79.3,94.8] |

   A **70.8pp** gap inside a single difficulty stratum. Coverage is not marking easy items.

2. **Window size (we set it, the model does not).** Growing W raises coverage mechanically while
   lowering the resolution gain, so a causal coverage account predicts an **interior optimum**. It
   appears: attn Δ = +4.7 / **+6.3** / +5.8 / +2.6 at W = 0.15 / 0.25 / 0.35 / 0.5, while coverage
   climbs monotonically 40.5% → 47.5% → 53.0% → 65.1%. Peak at W=0.25, not at the largest window.

3. **Box size as an instrument for random-placement coverage — INCONCLUSIVE, reported as such.**
   The `rand@0.15` delta does not rise with box size (−12.5 / −26.6 / −25.4). Two reasons, both
   visible in the data: `uniform` accuracy rises with box size (37.5% → 64.1% → 68.3%), so headroom
   shrinks exactly where the instrument should bite; and a W=0.15 window is 2.25% of the image while
   the *largest* box band has median area 1.2%, so random placement rarely hits even a large box.
   The instrument is too weak to test the hypothesis. It is **not** counted as evidence either way.

⚠ **This sentence is withdrawn (§6F).** Bootstrapping instrument 2 shows every CI crossing zero, and
instrument 3 fires on 11/191 items. What stands: one strong mediation result (instrument 1), plus
the multi-window dose-response in §6G.

---

## §6E  ★ WHY THE METHOD WORKS: the confidence gate is an unsupervised coverage detector

The conditional allocator (§4Y) answers from whichever pass is more confident. Nothing in its design
mentions coverage — yet coverage is what it is detecting:

| signal | AUROC for covered-vs-missed (n=191, 95 vs 96) |
|---|---|
| **conf(attn crop)** | **0.835** |
| conf(uniform) [control] | 0.675 |
| conf(attn) − conf(uniform) | 0.579 |

And it routes on it: the gate takes the crop **88.4%** of the time when the window covered the
target, versus **58.3%** when it missed (a perfect detector would be 100% / 0%).

### How much of the mechanism the method already captures

| arm | acc | vs uniform |
|---|---|---|
| uniform@300 | 56.5% | +0.0pp |
| always attn@0.15 | 61.3% | +4.7pp |
| **conf gate (deployed)** | **67.0%** | **+10.5pp** |
| ORACLE coverage gate (routes on true coverage) | 69.1% | +12.6pp |
| per-item best (upper bound) | 75.9% | +19.4pp |

⚠ **83% is the WRONG PAIRING — see §6F.** It divides the TEXT+CONF arm by the conf-only arm's
ceiling. Matched, the headlined arm captures **88%** (+12.0 of +13.6). The conclusion is unchanged
(both < 90%), but the number below is superseded.

The deployed gate recovers **83% of what a perfect coverage detector would deliver**. So the
remaining headroom is **detection, not allocation** — and it also explains the §4Y negative results:
`conf_thresh` (59.7%) and `learned_LR` (57.1%) failed because a threshold on absolute confidence,
or a linear probe on attention features, is a worse coverage detector than the two-pass confidence
comparison, which reads the crop's own downstream effect.

This is also why the relational-keyword rule worked on V\*Bench and failed on HR-Bench: keywords are
a *proxy for dispersion of the evidence set*, good where relational phrasing tracks it (V\*Bench,
99% concordance) and poor where it does not (HR-Bench, 53.5%).

---

## §6F  ⚠ CORRECTIONS to §6D/§6E after the Phase 39 audit

Four items, three of which go against the earlier write-up.

**1. Arm mismatch in the capture fraction (§6E) — FIXED.** §6E quoted "83% of what a perfect coverage
detector would deliver" by dividing one arm's gain by a *different* arm's ceiling: the headlined
method is **TEXT+CONF** (68.6%, +12.0pp) but the ceiling +12.6pp was the **conf-only** arm's. Scored
against its own ceiling — same routing rules, routing on *true* coverage — the headlined arm reaches
**+12.0 / +13.6 = 88%**. The conclusion survives (88% < 90%, so detection is still the bottleneck),
but the mismatched pairing gives 12.0/12.6 = 95%, which would have **inverted** it to "detection is
solved, the proposal is the bottleneck." `test_oracle_gate_bounds_the_deployed_gate` now asserts the
matched pairing and explicitly asserts that the mismatched one exceeds 90%.

**2. The exogenous-coverage instrument — INCONCLUSIVE, not supporting.** phase32's random centres are
recoverable: one seeded stream, `random.Random(32)`, two draws per item, every skip before the draw.
Pre-registered one-sided (a desynced replay cannot manufacture a dose-response, so a positive result
would be valid and a null ambiguous). **Null.** Only **11/191** random windows overlap the box at
all — a W=0.15 window is 2.25% of the image, the median box 0.06% — and the hit bin is n=11 with CI
[−72.7,+0.0]. Structurally underpowered. Not counted either way.

**3. The W sweep does NOT show an interior optimum — RETRACTED.** §6D(2) read Δ = +4.7/+6.3/+5.8/+2.6
as a causal signature. Bootstrapped: **every CI crosses zero**, and no paired peak-vs-neighbour
contrast separates (W=0.25 vs 0.15: +1.6 [−2.6,+5.8]; vs 0.35: +0.5 [−5.2,+6.3]; vs 0.5: +3.7
[−3.1,+10.5]). Four noisy points. `test_w_sweep_peak_is_not_significant` locks the retraction in.

**Consequence: §6D may not be called causal.** What remains is one strong mediation result with
baseline difficulty controlled (the 70.8pp within-stratum gap), one interventional test that came
out null, and one instrument too weak to fire. Stratifying on `uniform` correctness removes
*baseline difficulty*; it does **not** remove "the model represents this object well," which could
drive both where attention points and whether cropping helps. **Coverage is the dominant predictor
and mediator of the sign — not a demonstrated cause.** §6D's heading and PAPER_FLOW §3 are corrected
to "mediating variable."

**Scope, added:** coverage is measured on **V\*Bench only** (n=191). HR-Bench ships no boxes; RePOPE
was never joined to window geometry. §3 reconciles three benchmarks, but the coverage variable
itself is single-benchmark and single-model.

**4. The zero-coverage failures are FAR misses (new, and it redirects the next run).** For the 96
missed items, gap from window *edge* to box: median **0.231 of image width**, p25 0.120, p90 0.586;
**6.2%** within 0.05, **17.7%** within 0.10. A W=0.35 window at the same centre rescues only **20%**.
So a larger or box-aware window does not fix the failure mode — consistent with the null W sweep —
and **multi-window search is the correct next experiment**, not window enlargement.

---

## §6G  ★ INTERVENTION: multi-window allocation (Phase 40, V\*Bench n=191, budget gate 2.0%)

The intervention §6F said was missing. Five arms at a matched **total** realized budget; 100% of items
admitted a second peak separated by ≥0.25.

**The control that makes it decisive.** `two_same` duplicates the *first* window. Against it,
`two_diff` holds constant the image count (2), total budget (300), per-image resolution (150) and the
two-image prompt format. **Only the second window's location varies.** Comparing against `one`
instead would confound a second region with halved resolution and with the two-image format.

| arm | acc | vs uniform | 95% CI |
|---|---|---|---|
| uniform | 56.5% | — | — |
| one (deployed) | 61.3% | +4.7 | [−3.7,+13.1] |
| two_diff | 62.3% | +5.8 | [−2.6,+13.6] |
| two_same | 62.3% | +5.8 | [−2.6,+14.1] |
| two_rand | 62.3% | +5.8 | [−2.6,+13.6] |

**Pooled pre-registered contrast: `two_diff − two_same` = +0.0pp CI [−5.2,+5.2] — NULL.** By the rule
fixed before the run, the coverage account fails *as a population-level method*. Reported as such.

**The pre-registered subgroup analysis — also written before the run — is where the account said the
effect must live, and it is there:**

| subgroup | n | two_diff − two_same | 95% CI |
|---|---|---|---|
| 2nd window **added coverage** | 18 (9%) | **+38.9pp** | [+11.1,+66.7] |
| 2nd window added nothing | 173 (91%) | −4.0pp | [−8.7,+0.0] |
| **rescued from zero coverage** | 14 (7%) | **+50.0pp** | [+14.3,+78.6] |

Subgroup membership is fixed by (peak1, peak2, GT box) *before* the model answers and is independent
of the answer, so this is stratification on the **dose** of the intervention, not post-outcome
selection. Placement moved with resolution, format, image count and budget held fixed, and accuracy
moved **only** where coverage moved. **This is the interventional evidence §6F lacked.**

**Both readings are kept, because both rules were pre-registered:**
- *Mechanism:* coverage is upgraded from mediator toward cause — an intervention that changes
  coverage changes accuracy, with a dose-response and a null where the dose is zero.
- *Method:* multi-window is **not** a population-level improvement (+1.0pp over `one`, n.s.). Mean
  coverage rises only 40.5%→45.4%, because the second peak adds coverage on just 9% of items, and a
  second window placed where it adds nothing **costs 4.0pp** [−8.7,+0.0] — a distractor crop.

**Method implication, stated but not yet tested:** the second window should be *gated* on predicted
coverage gain, exactly as the first window is gated on predicted coverage (§6E). An ungated second
window pays +38.9pp on 9% of items and −4.0pp on 91%, which nets to zero. This is the same
detection-not-allocation conclusion as §6E, one level up.

By category: `two_diff` helps direct_attributes (65.2% vs uniform 49.6%) and does not rescue
relative_position (57.9% vs uniform 67.1%) — so a second window does **not** by itself fix
multi-region questions, which is further evidence against the §6C region-count reading.

---

## §7A  ★★ UNIVERSAL: the row-boundary sink holds across four architectures (Phase 41)

§6A's columnar sink was measured on Qwen3-VL-2B alone. Its *explanation* — raster serialization —
is a claim about how VLMs flatten images into sequences, which is common to all of them, so it must
generalise or be withdrawn. Tested on four architectures spanning two model families, two vision
stacks, two tokenization schemes and a 3.5× parameter range. Last-token attention, per-layer
L1-normalised, averaged over a **relative** layer block [0.55L, 0.95L] — the architecture-agnostic
form of Qwen3-VL's L16–26 of 28. Enrichment = mass share / area share, so 1.0× is a fair share.
n = 24 V\*Bench images per model; CIs are bootstrap over images.

### Models with an IMPLICIT row boundary (flattened grid, no separator token)

| model | last column | first column | top row | bottom row | interior |
|---|---|---|---|---|---|
| Qwen3-VL-2B | **3.4×** [3.0,3.8] | 2.0× [1.7,2.3] | 0.8× [0.7,1.0] | 0.8× [0.7,0.9] | 0.7× |
| Qwen2-VL-7B | **4.5×** [4.1,4.8] | 0.9× [0.9,1.0] | 1.3× [1.2,1.4] | 0.6× [0.5,0.7] | 0.7× |

### Models with an EXPLICIT row boundary (a learned `image_newline` spliced in at each row end)

⚠ **SUPERSEDED BY §7C.** The table that stood here scored these models on a single `OTHER` bucket
holding 95.4% of positions, so it only showed "the separator beats the average image token" — not the
same, stronger claim the Qwen rows make. The five-region split is in §7C and it **changes the
conclusion**: the separator result survives, but "rows carry no excess mass" does not.

**This is the discriminating result.** The two accounts differ precisely here:

- *2-D / corner account*: a newline embedding is **not** a corner, not a pixel, and not part of the
  image. It should attract nothing special.
- *Serialization account*: wherever the row boundary is made **explicit**, the sink should sit **on
  the explicit marker**.

The sink sits on the separator in both LLaVA models, with CIs excluding 1.0. Newline positions were
located by matching merged input embeddings against the model's own `image_newline` parameter, so
no unpad logic was reimplemented and nothing depends on our arithmetic.

**Universal claim — ⚠ see §7C for the corrected, weaker form.** The "rows carry no excess mass"
half of the statement below does **not** survive the five-region split on the LLaVA models, where
the bottom row is enriched 2.1–3.0×. The trailing-boundary half does survive.

Superseded text:
> Across four VLMs from two families, attention mass concentrates on the **row-boundary position of
> the raster serialization** — the last column where the boundary is implicit, the learned newline
> embedding where it is explicit — at 2.0–4.5× its fair share, with every CI excluding 1.0. Rows
> carry no excess mass (0.6–1.3×). The "attention sink at image corners" is a **1-D sequence
> artifact**, not a 2-D spatial one.

**Honest limits.** (a) The *first*-column enrichment is **not** universal: 2.0× on Qwen3-VL but 0.9×
on Qwen2-VL. Only the **trailing** boundary replicates. (b) Magnitudes vary 2.0–4.5×, so this is a
consistent phenomenon, not a constant. (c) Qwen2-VL's top row is mildly enriched (1.3×) while its
bottom row is not (0.6×) — the row effect is near-null but not exactly null in every model.
(d) n = 24 images per model, at reduced input resolution so full attention matrices fit; geometry
is unaffected but magnitudes at full resolution may differ (Qwen3-VL's last column reads 5.1× at
full size vs 3.4× here).

---

## §7B  ★★ UNIVERSAL: the mechanism and the method transfer to a second architecture (Phase 42)

Qwen2-VL-7B-Instruct on V\*Bench, n = 191, budget gate 4.8%. **Nothing refitted** — W ∈ {0.15, 0.25},
the outer-ring mask and B₀ = 300 all come from Qwen3-VL, and the layer block is the
architecture-agnostic relative form [0.55L, 0.95L] (= L15–26 of 28, matching Qwen3-VL's L16–26).
A different generation of the vision stack and 3.5× the parameters.

| arm | acc | vs uniform | 95% CI |
|---|---|---|---|
| uniform@300 | 52.9% | — | — |
| **oracle** | **91.6%** | **+38.7** | [+30.9,+47.1] |
| attn@0.15 | 58.1% | +5.2 | [−2.1,+12.6] |
| attn@0.25 | 61.8% | +8.9 | [+1.0,+16.8] |
| rand@0.15 | 38.2% | −14.7 | [−23.0,−6.3] |
| rand@0.25 | 40.8% | −12.0 | [−20.4,−3.7] |

### All three single-model claims hold

| claim | Qwen2-VL-7B | Qwen3-VL-2B |
|---|---|---|
| **1. localizer transfers** (attn − rand) | **+19.9pp** [+11.5,+28.3] @0.15; +20.9 @0.25 | +40.0pp |
| **2. coverage mediates** — missed | **−9.4pp** [−17.9,−0.9] | −15.6pp |
| **2. coverage mediates** — full | **+41.8pp** [+29.1,+54.5] | +37.1pp |
| **3. gate detects coverage** (AUROC) | **0.885** (uniform control 0.601) | 0.835 |
| deployed conf gate | **+7.9pp** [+2.1,+14.1], **71%** of its ceiling | +12.0pp, 88% |

Mean coverage achieved by the attention peak is **34.8% vs 2.3% for a random centre** — the
localizer is finding the target, not the crop shape doing the work. The coverage dose-response
reproduces with the same sign flip and a similar crossing, on a model whose accuracy profile
differs substantially (uniform 52.9% vs 56.5%; oracle 91.6% vs 97.4%).

**This removes the paper's main scope limit.** §3 (coverage), §4 (sink-masked localizer) and §5
(the gate as a coverage detector) were single-model; they are now two-architecture, and the sink
itself is four-architecture (§7A).

### ⚠ BUG FOUND AND FIXED: a dtype error that masqueraded as a clean negative

The first Phase 42 run loaded this **bfloat16** checkpoint in **fp16** and produced **all-NaN
logits**. `max(range(4), key=...)` falls through to index 0 on NaN comparisons, and because 71/191
V\*Bench labels are "A", *every arm* scored an identical **37.2%** with CI **[+0.0,+0.0]**. The
analyzer read this as "CLAIM 1 does not hold, CLAIM 2 does not hold" — two false negatives that
looked exactly like publishable architecture-specificity results.

Caught by the signature, not by luck: every arm bit-identical *and* zero-width CIs on all of them,
plus `conf(uniform)` scoring AUROC 1.000. The runner now asserts `torch.isfinite` on both logits and
attention before anything is logged, and `test_nan_logits_masquerade_as_a_uniform_negative_result`
encodes the signature so scoring code must refuse such data rather than describe it. Phase 41's
Qwen2-VL row was re-run in bf16 as a precaution: **4.5× vs 4.4×**, so its attention was unaffected
(the NaN was confined to the LM head), and its conclusion stands on verified dtype.

---

## §7C  ⚠ CORRECTION to §7A, and the layer-block assumption tested (Phase 43, 44)

### 1. The LLaVA table was not measuring the same thing as the Qwen table

§7A scored the grid models on four disjoint content regions but the newline models on one `OTHER`
bucket holding 95.4% of positions. So the Qwen rows established "the trailing boundary beats the
leading boundary and both beat rows"; the LLaVA rows established only "the separator beats the
average image token." The discriminating prediction — **rows are ordinary** — was never tested there.

**A bug in my own bucketing, found while fixing this.** Segmenting on separators naively treats the
first run as "row 0", but these models prepend a **base image carrying no separators at all**. That
put **59.8% of all positions into TOP ROW** on OneVision (one row of ~19 cannot exceed ~5%) and
mislabelled the final tile row as BOTTOM ROW, which then read as 3.0× "elevated". Fixed by keeping
only **modal-length** segments as rows and excluding the base image explicitly.

**Corrected five-region table** (n=24/model, same relative block, same normalisation):

| region | LLaVA-OneVision-7B | LLaVA-NeXT-7B | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|---|---|
| **newline separator** | **2.0×** [1.9,2.2] | **2.3×** [1.9,2.9] | — | — |
| last col (pre-separator) | 1.3× [1.1,1.5] | 1.2× [1.0,1.3] | **3.4×** | **4.5×** |
| first col | 1.0× [0.9,1.1] | 1.1× [0.9,1.3] | 2.0× | 0.9× |
| top row | **0.7×** [0.6,0.8] | **0.7×** [0.5,0.9] | 0.8× | 1.3× |
| **bottom row** | **3.0×** [2.7,3.4] | **2.1×** [1.7,2.4] | **0.6×** | **0.8×** |
| interior | 1.0× | 1.2× | 0.7× | 0.7× |
| excluded (base image) | 0.9× | 0.7× | n/a | n/a |

**What survives, and what does not:**

- ✅ **The trailing row boundary is enriched in all four models** — the last column at 3.4–4.5× where
  the boundary is implicit, the learned separator at 2.0–2.3× where it is explicit. Every CI excludes
  1.0. A newline embedding is not a corner, not a pixel and not image content, so a 2-D account
  cannot produce this; a serialization account predicts exactly it.
- ✅ **The top row is ordinary everywhere** (0.7–1.3×).
- ❌ **"Rows carry no excess mass" is RETRACTED as a universal.** The **bottom row is strongly
  enriched in both LLaVA models (3.0×, 2.1×) and not in either Qwen model (0.6×, 0.8×).** This is a
  genuine cross-family divergence and is reported as one.
- ❌ The **first**-column enrichment still does not replicate (2.0× Qwen3-VL vs 0.9–1.1× elsewhere).

A plausible but **untested** account of the bottom-row divergence: LLaVA places the base image first
and anyres tiles after, so the last tile row is the image content nearest the text in a causal LM,
whereas Qwen has no such split. Stated as a conjecture, not a result.

**Universal claim, restated at the strength the data now supports:**
> Across four VLMs from two families, two vision stacks and two tokenization schemes, attention mass
> concentrates on the **trailing row boundary of the raster serialization** — the last column where
> that boundary is implicit, the learned `image_newline` embedding where it is explicit — at 2.0–4.5×
> its fair share, every CI excluding 1.0, while the **top** row stays ordinary (0.7–1.3×). The
> corner framing is wrong in all four. A second, **family-specific** sink sits on the final image row
> in the LLaVA models only.

### 2. The relative layer block is validated, not assumed (Phase 44)

The block [0.55L, 0.95L] was reverse-engineered from Qwen3-VL's L16–26 of 28 — a **non-monotone
regime**, which is exactly what should not be rescaled to another architecture on faith. Swept every
layer of Qwen2-VL-7B on `gt_pct` (n=120, chance 0.500 exactly, ring-masked as deployed):

| | result |
|---|---|
| best 8 layers by gt_pct | 16, 18, 19, 20, 21, 22, 23, 26 — **8/8 inside the transferred block** |
| median gt_pct **inside** block | **0.041** |
| median gt_pct **outside** block | **0.390** |
| best single layer | L19: gt_pct **0.010**, argmax-in-GT **16.7%** |

The same regime structure appears: L0–13 sit at 0.22–0.52, L14 onward collapse to 0.01–0.13, and L27
rises again (0.060). **Validated.** Phase 42's +19.9pp attn−rand and +7.9pp gate stand as measured,
not as a lower bound under a transferred hyper-parameter.

### 3. The exchange rate is three architectures, and placement-vs-random is four (Phase 28)

| model | oracle crop @B₀ | random crop @B₀ | **placement effect** | uniform ceiling | exchange rate |
|---|---|---|---|---|---|
| Qwen2-VL-7B | 93.2% @300 tok | 34.0% | **+59.2pp** | 75.4% @7776 | **≥25.9×** |
| LLaVA-OneVision-7B | 86.9% @1323 | 37.7% | **+49.2pp** | 69.6% @5157 | **≥3.9×** |
| LLaVA-NeXT-7B | 80.6% @1176 | 38.2% | **+42.4pp** | 38.2% @1464 | ≥1.2× (**no axis**) |
| Qwen3-VL-2B | 97.4% @300 | — | +40.0pp (attn−rand) | — | ≥26× |

No uniform rung reaches the crop arm on any model, so every exchange rate is a **lower bound set by
the model's own token ceiling**, not an estimate. LLaVA-NeXT's 1.2× reflects its 1.5× dynamic range
— the axis does not exist there, which is why it is reported as "NO AXIS" rather than as a weak
result.

---

## §8A  ★ THE SINGLE-PASS COVERAGE PREDICTOR: adaptive allocation (Phase 45, 46)

### The problem it targets

The deployed allocator runs **two passes on every item**, so its honest bar is `uniform@600`. On
V\*Bench it clears that bar by only +1.0pp (conf-only) / +2.6pp (TEXT+CONF). On HR-Bench it **loses**:
conf_route −3.1pp, TEXT+CONF −5.9pp. A reviewer reaches that in one line — *on the transfer
benchmark, simply spending the tokens beats the method.* It is a **cost-accounting** problem: the
method pays 2× on 100% of items to discover, item by item, something it is usually wrong about.

### Coverage is predictable from pass 1, for free

§6E showed the confidence gate is a coverage detector — but it detects coverage *by cropping first*,
which is what costs the second pass. Measured on Phase 32's logged pass-1 features (n=191):

| predictor of coverage | cost | AUROC |
|---|---|---|
| `conf(attn crop)` — the deployed signal | **a second forward pass** | 0.835 |
| **`peak` attention value alone, unfitted** | **free** | **0.788** |
| all pass-1 features, 5-fold CV | free | 0.766 |
| conf(uniform) alone | free | 0.675 |
| peak geometry only | free | 0.531 |

`peak` is the max of the ring-masked, L1-normalised, block-averaged attention map — a number the
localizer already computes on its way to the argmax. It also **beats the fitted multi-feature model**,
which overfits at n=191.

### The policy, and honest compute accounting

Pass 1 gives an answer, a peak location and `peak`. If `peak < τ`, stop (cost 1×). Else crop and run
pass 2, answering by the confidence comparison (cost 2×). **The bar moves with the policy**: at
average cost c the comparison is a plain uniform image of 300·c tokens, read off the *measured*
sweep, never assumed. τ is chosen on training folds and applied out-of-fold.

**V\*Bench** (bar from phase27's measured sweep: 294→56.5%, 600→66.0%, 1176→70.2%):

| policy | cost | acc | bar | **margin** |
|---|---|---|---|---|
| deployed always-2-pass | 2.00× | 67.0% | 66.0% | **+1.0pp** |
| **adaptive, 40% firing** | **1.41×** | 66.5% | 61.3% | **+5.2pp** |

**5× the margin at 30% less compute**, and the margin is positive at *every* firing rate tested
(+1.1 to +5.2pp), so the conclusion does not depend on the rate. It routes where coverage is without
ever seeing a box: fired items have **76.9%** coverage rate vs **31.0%** for declined.

⚠ **The control that matters.** Random routing at the same cost already earns **+3.3pp**, purely
because spending 1.41× on a crop beats 1.41× more uniform tokens — a concavity effect, not routing
quality. So **`peak`'s own contribution is +1.9pp over random**, not +5.2pp. Most of the win is the
compute accounting; the signal adds a little on top. `conf(uniform)` routing is *worse* than random
(−1.5pp). Reported this way, not as +5.2pp.

### Does it rescue HR-Bench? Partly — and the transferred setting does not win

| policy | cost | margin vs its own bar |
|---|---|---|
| always-2-pass (deployed) | 2.00× | **−3.1pp** |
| **adaptive @ V\*Bench-transferred 40% firing** | 1.40× | **−0.5pp** CI[−3.9,+3.0] |
| adaptive @ HR-Bench-best 20% firing | 1.20× | +1.9pp CI[−1.5,+5.2] ← **tuned on this benchmark** |

**Honest verdict: break-even, up from a clear loss — not a win.** The adaptive fix removes the
transfer *failure* but does not convert it into a transfer *success*. The +1.9pp requires choosing
the firing rate on HR-Bench itself and is a bound, not a transfer result. Unlike V\*Bench, the margin
here is positive only at 10–30% firing and negative from 40% up, so the rate choice does matter. On
CircularEval (the benchmark's own metric) adaptive@40% reaches 44.0% against uniform@600's 45.5% —
still slightly behind. At the 20% rate, `peak` routing beats random routing by **+2.8pp** (+1.9 vs
−0.9), so the signal is doing real work here even though the policy as a whole only breaks even.

### What this changes for the paper

The method is no longer beaten by its own compute-matched control on either benchmark, and on
V\*Bench it now wins by a margin that survives the choice of operating point. The mechanism section
becomes **load-bearing** rather than explanatory: the paper's claim is that coverage predicts the
sign of the allocation effect, and the method now *acts on a prediction of coverage made before
paying for the crop*. The residual headroom is unchanged in kind — the oracle-coverage policy reaches
69.1% at 1.50×, so the adaptive policy captures **79%** of the achievable gain at lower cost.

### An error caught in this phase

The first version selected τ by maximising **accuracy per pass**. That objective is maximised by
never firing (0.565/1.00 beats 0.670/2.00), so it drove τ to fire on 2% of items and reported
+0.0pp — which reads as "the pass-1 signal is useless." The objective, not the signal, was wrong:
the question is accuracy *at* a cost, so τ is now chosen to hit a target firing rate and the full
cost-accuracy curve is reported instead of a single point.

---

## §9A  ★★ PRIOR-ART HEAD-TO-HEAD at matched realized budget (Phase 47)

The comparison the crop/zoom literature does not run: every policy charged for **every image token
it processes across every pass**, and scored against what a single uniform image of that same spend
buys **on the same items**.

### The table

n = 191 V\*Bench, Qwen3-VL-2B. Bar anchored on uniform arms measured **in this run**
(296tok→56.5%, 598→63.9%, 1182→71.7%); beyond 1182 the curve is extended with phase27's *measured*
increments (shape borrowed, level anchored here) and such rows are marked EXT.

| policy | family | passes | tokens | acc | bar | **margin** | 95% CI |
|---|---|---|---|---|---|---|---|
| uniform@300 / @600 / @1200 | budget axis | 1.00 | 296/598/1182 | 56.5/63.9/71.7% | — | — | — |
| grounding_crop | Chain-of-Spot · Visual CoT · DualFocus | 2.00 | 595 | 57.6% | 63.8% | **−6.2** | [−13.6,+0.6] |
| zoom_eye | Zoom Eye (tree search) | **9.00** | **2664** | 75.9% | 77.8% | **−1.9** EXT | [−8.1,+3.9] |
| ours_attn@0.15 | ours, fixed policy | 2.00 | 591 | 61.3% | 63.8% | −2.5 | [−9.3,+4.3] |
| **ours_adaptive** | **ours, peak-gated** | **1.40** | **414** | 62.8% | 60.0% | **+2.8** | [−4.5,+9.6] |
| oracle | ceiling (uses GT box) | 1.00 | 301 | 92.7% | 56.7% | +35.9 | [+32.3,+39.6] |
| rand@0.15 | floor | 1.00 | 296 | 33.5% | 56.6% | −23.0 | [−29.9,−16.2] |

**Cost-normalised (accuracy per 1000 image tokens):** oracle 3.07 · uniform@300 1.91 ·
**ours_adaptive 1.52** · uniform@600 1.07 · ours_fixed 1.04 · grounding 0.97 · uniform@1200 0.61 ·
**zoom_eye 0.28**.

### What it shows

**No prior-art policy beats the budget axis at its own spend.** Zoom Eye's 75.9% is the highest raw
accuracy in the table and reads as a decisive win in the usual presentation — until it is charged
for the 9 passes and 2664 tokens it spends, at which point a single uniform image of that budget
scores 77.8%. The grounding family loses by 6.2pp at 2 passes. **Ours is the only policy with a
positive margin, and it is the cheapest arm in the table** (1.40 passes, 414 tokens).

**⚠ None of the margins are individually significant at n=191.** Every CI crosses zero except the
oracle and the random floor. The defensible headline is: *no policy significantly beats the budget
axis, and ours is the only one whose point estimate is positive, at a third of Zoom Eye's cost.*
Zoom Eye does beat us on **raw** accuracy (+13.1pp, CI [+4.7,+21.5], significant) at **6.4× the
tokens**; that trade is the honest summary of tree search.

### Why the grounding family fails here — predicted by §6D

Its bounding boxes parse on 187/191 items (98%) but have **median IoU 0.029** with ground truth, and
only **22%** clear IoU > 0.5. Asking the model where to look fails precisely in the **sub-token
regime**, because it must ground on the same downscaled view that already cannot resolve the target.
The coverage account predicts a competitor's failure mode, which is the strongest form of support it
has received.

### Limitations, stated because the table will be read on its own

These are **reimplementations of the published policies on one common backbone**, not the released
checkpoints. That isolates the POLICY from the backbone — comparing our method on Qwen3-VL against
SEAL's fine-tuned vicuna-7B would confound search policy with base model — but it **cannot reproduce
gains that come from fine-tuning** rather than from search. Chain-of-Spot, Visual CoT and DualFocus
are represented by their inference-time mechanism only, on a backbone never trained to emit regions.
Zoom Eye is training-free and its core (best-first search over regions scored by the model's own
confidence, depth 2 over quadrants) is reproduced faithfully. V\*/SEAL's visual-search module is
**not** reproduced and is absent from the table.

---

## §10A  ✗ NEGATIVE: suppressing the serialization sink at inference buys nothing (Phase 50)

§7A/§7C established that the trailing row boundary of the raster serialization absorbs **25% of the
last token's attention over image tokens** at 2.0–4.5× its fair share, on four architectures, from
L0. That mass is spent on a position carrying no image content. Ring-masking removes it from our
*read-out*, but the model still spends it — so the obvious method is to suppress it **inside the
forward pass** and let softmax redistribute the freed mass.

**Implementation.** `eager_attention_forward` patched to add a per-module additive bias over key
positions before softmax. Self-tested: a zero bias is a provable no-op, a −8 bias provably moves the
output. **Every arm is ONE pass at the same 294 realized tokens on the same image**, differing only
in the bias — a pure paired comparison with no compute bar to clear.

n = 191, biasing 14 of 294 image tokens (4.8%). Δ vs baseline (56.5%), * = CI excludes zero:

| arm | b=−1 | b=−2 | b=−4 | b=−8 |
|---|---|---|---|---|
| **last_col (the universal sink)** | −0.5 | −1.6 | −1.6 | −1.6 |
| first_col (Qwen3-VL-only sink) | −2.6 | −2.6 | −2.6 | −1.6 |
| interior (real content, same count) | −0.5 | +0.0 | −1.6 | −1.6 |
| rand_cols (same count, random) | +0.5 | +0.0 | −1.0 | −0.5 |
| **text_tokens (same count, prompt)** | **−6.8\*** | **−14.7\*** | **−16.8\*** | **−17.3\*** |

**Verdict: NO BENEFIT.** Best sink arm −0.5pp CI[−2.6,+1.6], and it does not beat the `interior`
(+0.0) or `rand_cols` (−1.0) controls. The sink is real and measurable, but **the mass it holds is
not recoverable by biasing it away at inference.** The text arm's −17.3pp proves the intervention
machinery works and bites, so this is a genuine negative and not a dead hook.

### The secondary result is the more interesting one

**Biasing *any* 4.8% of image tokens changes accuracy by ≤2pp; the same number of prompt tokens costs
up to 17.3pp** (last_col − text_tokens = +6.3pp, CI [+1.0,+11.5], significant). The model's answer is
nearly insensitive to *which* image tokens receive attention, while being acutely sensitive to prompt
tokens. That is a sharp statement of how little any individual image token contributes at B₀ = 300 —
and it is the same fact the whole paper turns on, measured from the inside: when the evidence occupies
less than one merged token, no attention over the existing tokens can recover it.

**What this rules out.** The destructive half of attention-space allocation is dead: there is no
"freed capacity" to reclaim by removing the sink. Whether the *constructive* half works — amplifying
attention to the target cells instead of cropping to them — is tested in Phase 51, whose oracle arm
is decisive either way.

---

## §10B  ★★ CAUSAL: allocation must happen in PIXEL space, not attention space (Phase 51)

The constructive counterpart of §10A, and the paper's cleanest causal result. Instead of giving the
evidence more *pixels*, give its tokens more *attention weight*: an additive bias on the attention
logits of the target cells. **One pass, 294 realized tokens, no crop — identical cost to baseline**,
so this is a paired comparison with no compute bar.

n = 191. Δ vs baseline (56.5%); * = CI excludes zero.

| arm | +1 | +2 | **+4** | +8 |
|---|---|---|---|---|
| **amp_oracle** (cells inside the GT box) | +1.0 | +3.1 | **+5.8\*** [+1,+10] | −13.1\* |
| amp_win (deployed W=0.15 window at the peak) | +0.5 | +0.5 | −1.0 | −10.5\* |
| amp_top1 / top5 / top15 (attention-ranked) | −1.0 / +2.1 / +1.0 | +0.0 / +2.1 / +2.1 | +2.1 / +0.5 / +0.0 | −12.6\* / −13.6\* / −12.6\* |
| amp_rand (control) | +1.6 | +1.0 | −1.6 | −9.4\* |

### Three findings, in order of importance

**1. Attention-space allocation is real but small.** The oracle arm gains **+5.8pp [+1,+10]**,
significant. So attention *does* carry some of the effect.

**2. It cannot substitute for pixel-space allocation.** The pixel-space oracle crop reaches **92.7%
at the same 300 tokens**, i.e. **+36.2pp**. Attention amplification on exactly the same region
recovers **15.9%** of that. Resolution, not attention routing, carries five-sixths of the effect.

**3. With a real localizer it gives nothing.** Every practical arm — top-1, top-5, top-15, and the
deployed window — is **indistinguishable from the random control** (all CIs straddle zero). Only the
ground-truth-targeted arm moves, and only modestly.

### Why, and it is visible in the setup line

**The oracle amplification set is 1 cell out of 294 — 0.3% of the image.** At B₀ = 300 the target
occupies less than one merged token, which is exactly the sub-token regime the paper is about, now
seen from *inside* the model. There is no token to amplify. **Cropping CREATES tokens on the target;
attention can only reweight tokens that already exist.** When the evidence is smaller than one token,
no redistribution of attention over the existing tokens can recover information that was never
encoded.

### What this settles

- **It proves the mechanism causally, by intervention.** The allocation effect is about the
  *information content of visual tokens*, not about attention routing. (This phrasing was briefly qualified by §12A on the
  basis of an apparent read-out gap; **§12D retracts that** — the gap was a double-normalisation bug
  in our own lens. With correct logits the final layer equals the best intermediate layer exactly, so
  the claim stands **unqualified** and is now supported by direct measurement as well as intervention.) An entire class of
  alternative explanations ("the model just isn't looking in the right place") is ruled out: we
  *made* it look in exactly the right place and got 16% of the benefit.
- **It retires attention steering for this problem, with evidence** rather than by omission —
  together with §10A's null on sink suppression, both the destructive and constructive halves of
  attention-space allocation are now measured and closed.
- **It explains why every method in this literature crops.** They are not being unimaginative;
  pixel-space allocation is the only operation that adds information, and the internal alternative
  was worth testing precisely because it would have been free.
- **It sharpens the method's remaining job.** Pixel allocation is *necessary*, and pixel allocation
  costs forward passes (§9A) — so the contribution is making it *cheap*, which is exactly what the
  free pass-1 coverage predictor does (§8A).

---

## §10C  ✗ NEGATIVE (with a real positive inside it): residual-stream steering (Phase 52)

§10B predicted residual steering would fail for the same reason attention steering did. The
prediction was **half wrong**, and the half that was wrong is the interesting part.

**Design.** For each item, run the identical prompt twice at 300 tokens — once on the uniform image,
once on the oracle crop (93.2%) — and take the last token's residual displacement
`dᵢ(L) = h_oracle(L) − h_uniform(L)`. Then inject it into the uniform pass. All steered arms are
**one pass at the same realized tokens**, so this is paired and free.

- `v_item` — the item's OWN displacement. A **reachability test**, not a method.
- `v_mean` — the out-of-fold mean displacement. **The method.**
- `v_rand` — a norm-matched random direction. The control.

### A shared direction exists, geometrically

| layer | mean \|dᵢ\| | \|mean d\| | ratio | mean cos(dᵢ, mean) |
|---|---|---|---|---|
| L8 | 5.28 | 4.10 | **0.78** | 0.75 |
| L14 | 17.76 | 13.30 | **0.75** | 0.71 |
| L20 | 221.39 | 150.33 | 0.68 | 0.63 |
| L26 | 495.05 | 286.82 | 0.58 | 0.53 |

The displacements do **not** cancel — there is a large common "allocate to the evidence" component,
strongest at early-middle layers. That contradicts the §10B-based prediction and is worth reporting.

### But it carries nothing transferable

n = 191. Δ vs baseline (56.5%); oracle crop +36.6pp. * = CI excludes zero.

| injection layer | v_item | v_mean | v_rand |
|---|---|---|---|
| L8 | −1.2 | +0.0 | +0.0 |
| L14 | −0.6 | −1.2 | −0.6 |
| **L20** | **+17.3\*** | +0.0 | −0.6 |
| L26 *(tautological — see below)* | +36.6\* | +0.0 | +1.6 |

**`v_mean` is flat zero at every layer and every strength**, and `v_mean − v_rand = −1.6pp`
CI[−5.2,+2.1], n.s. **Residual steering does not work as a method.**

### ⚠ The arm that must not be misread

`v_item` at **L26 recovers +36.6pp — the entire oracle gap — and this is near-tautological.** The
model has 28 layers; adding `h_oracle − h_uniform` at L26 sets the residual *equal to* `h_oracle`
two layers from the head, so reproducing oracle logits there demonstrates only that the readout
depends on the final residual. It is labelled as such in the analyzer and excluded from the headline.

The informative number is **L20: +17.3pp, i.e. 47% of the crop's benefit survives injection seven
layers before the head.** At L8 and L14 it is zero.

### What this actually shows

1. **The crop's benefit IS partially reachable by a residual shift** — but only late (L20), not
   early (L8/L14). The model cannot use an injected late-stage representation early; the benefit is
   manufactured by the vision encoder and early processing, and cannot be pasted in before then.
2. **The shared direction is real but answer-irrelevant.** Cosine 0.53–0.75 across items, yet zero
   transfer. The common component carries the *statistics* of a cropped image — sharper, tighter,
   different scale — not the evidence that makes the target legible. **The usable part of the
   displacement is item-specific.**
3. **Together with §10A and §10B this closes the internal-intervention direction with evidence.**
   Attention suppression (null), attention amplification (16% of the crop, nothing above random with
   a real localizer), and residual steering (no transferable direction) all fail for one reason: when
   the target occupies less than one merged token, the information is **absent from the
   representation**, not merely mis-weighted within it. Only adding pixels adds information.

This is the strongest support the paper's mechanism has: three different internal interventions, each
of which would have been free had it worked, each failing exactly where the sub-token account says it
must.

---

## §9B  ★★ PRIOR ART at 4K: the negative replicates with significance; OUR METHOD DOES NOT WIN (Phase 53)

HR-Bench 4k, **n = 800 rows / 200 instances**, 4032×4032 images, CircularEval, Qwen3-VL-2B. Bar =
uniform arms measured **in-run on these items** (293tok→52.8%, 584→59.9%, 1215→64.5%). Every policy
charged for every image token across every pass; per-instance proposal cost charged to each of its 4
rows (the deployment view, applied identically to all search-based methods). Adaptive uses the
**V\*Bench-transferred** 40% firing rate, not refit. No oracle arm — HR-Bench ships no boxes.

| policy | passes | tokens | acc | bar | **margin** | 95% CI |
|---|---|---|---|---|---|---|
| grounding_crop (Chain-of-Spot · Visual CoT · DualFocus) | 2.00 | 590 | 61.3% | 59.9% | **+1.3** | [−2.1,+4.7] n.s. |
| **ours, peak-gated adaptive** | **1.40** | **411** | 56.0% | 56.2% | **−0.2** | [−3.6,+3.3] n.s. |
| **zoom_eye (tree search)** | **9.00** | **2636** | 54.1% | 69.4% | **−15.3** | **[−18.6,−11.8] SIG** |
| ours, fixed policy | 2.00 | 587 | 42.6% | 59.9% | **−17.3** | [−20.7,−13.8] SIG |
| rand@0.15 (floor) | 1.00 | 294 | 35.0% | 52.8% | −17.8 | [−21.1,−14.4] SIG |

**CircularEval** (the benchmark's own metric): uniform@1200 **52.0%** · grounding 47.0% ·
ours_adaptive 45.5% · uniform@600 46.0% · zoom_eye 42.0% · uniform@300 37.5% · ours_fixed 26.5%.

### What replicates, now with significance

**Tree search loses decisively at matched budget, in the venue it was designed for.** Zoom Eye is
**−15.3pp [−18.6,−11.8]** against its own bar, and **−10.4pp [−14.1,−6.6] against uniform@1200 while
spending 2.2× the tokens**. Per token it is the worst arm in the table (0.21 vs uniform@300's 1.80).
The Phase 47 finding replicates at 4× the power, at 4K, with significance.

### ⚠ What does NOT replicate: our own headline

Phase 47's "ours is the only policy with a positive margin" **does not hold at 4K**:

- `ours_adaptive` is **−0.2pp** — exactly break-even, not a win.
- `grounding_crop` is the only positive margin (+1.3pp, n.s.) and **beats us by 5.2pp
  [−8.5,−2.1], significant**.
- `ours_attn@0.15` (the fixed policy) is **−17.3pp**, as bad as random placement's −17.8pp: a
  W=0.15 crop of a 4032² image discards 97.75% of the scene and is catastrophic.

**The claim "ours is the only policy that beats the budget axis" is therefore retracted as a
cross-benchmark claim.** It holds on V\*Bench (+2.8pp) and not on HR-Bench (−0.2pp). What survives
across both scales is weaker and must be stated as such: *the adaptive policy is the only arm that
stays at or above break-even at both scales, and it does so at the lowest cost in the table* (411
tokens; cost-normalised 1.36, second only to uniform@300's 1.80).

### The sign flip in the grounding family, and why

Grounding goes from **−6.2pp on V\*Bench to +1.3pp on HR-Bench**. §6D predicts this: grounding must
localise from a 300-token view, and on 1500–2000px V\*Bench images the target is sub-token there
(median IoU 0.029). On 4032² images the same targets are far larger in absolute pixels, so the
grounding pass can actually see them. The coverage account predicts the direction of a competitor's
sign flip across scales — further support for it, and a warning that **our own method's advantage is
scale-dependent in the opposite direction**.

### By category

| | uniform@300 | uniform@1200 | zoom_eye | ours_adaptive |
|---|---|---|---|---|
| cross (multi-region) | 51.7% | 57.0% | 41.5% | 50.5% |
| single (one region) | 53.8% | 72.0% | 66.8% | 61.5% |

The budget axis is far more productive on `single` (+18.2pp from 1× to 4×) than on `cross` (+5.3pp),
and every crop-based method suffers most on `cross` — consistent with §6D's coverage account.

### Honest consequence for the paper

The **negative result is now the strongest thing in the paper** and it is statistically significant:
at 4K, with n=800 and CircularEval, no crop/zoom policy beats simply spending the tokens, and the
best-known tree search loses by 15pp. The **method contribution is correspondingly weaker**: it wins
on V\*Bench, breaks even at 4K, and is beaten there by a grounding baseline. Any framing that leads
with the method must carry this table, not hide it.

---

## §9C  ✗ FOUR ATTEMPTS TO WIN AT 4K, ALL NEGATIVE — the scale boundary is real (Phases 53–56)

Phase 53 established that our crop-only policy loses by **−17.3pp** at 4K. Four distinct fixes were
built and measured, each on the full n=800 / 200 instances with CircularEval. **None wins.**

| attempt | idea | best margin at 4K |
|---|---|---|
| **peak-gated adaptive** (§8A) | pay the second pass only when `peak` clears τ | **−0.2pp** [−3.6,+3.3] |
| **scene+crop composite** (Phase 54) | never discard the scene; two images, one pass | −3.5 to −7.2pp |
| **proposer-agnostic gate** (Phase 55) | wrap any proposer in the free gate | −0.2pp (ours), −0.8pp (grounding) |
| **coarse-to-fine localiser** (Phase 56) | localise twice, 2.9× finer | **−0.2pp** gated; ungated −19.8pp |

### The coarse-to-fine result is the decisive one

The second localisation genuinely refines: cell size **237px → 83px** (2.9×), and the peak **moves a
median of 0.121 of image width (~489px)**. Yet:

**`c2f_fine@0.15` − `coarse_fine@0.15` = +0.0pp, CI [−3.0,+2.9].** Same window size, same final
budget, differing only in whether the centre came from one localisation or two. **Exactly zero.**

The fine peak is no more on-target than the coarse one. Refinement is not the missing piece — the
attention map at 4K does not carry localisation that survives being resolved more finely. (It does
carry *something*: attn − rand = +13.6pp, §6. But the coarse pass already extracts all of it.)

### The honest conclusion, and it is a finding

**The method's advantage is scale-bounded**, and the boundary is sharp:

| | V\*Bench (~1500–2000px) | HR-Bench 4k (4032px) |
|---|---|---|
| gated allocation vs its bar | **+5.4pp** | **−0.2pp** |
| crop-only vs its bar | −2.5pp | **−17.3pp** |
| uniform 1×→4× gain | +15.2pp | **+11.7pp, and cheaper per pp** |

Two things move together as image size grows: the budget axis becomes more productive (so the bar
climbs faster), and the localiser's grid cell grows in absolute pixels (237px at 4K), so the same
attention map buys a less precise proposal. Allocation loses the race on both counts at once.

**This is reported as a scale boundary, not as a failed method.** The paper's own coverage account
predicts it — at 4K a W=0.15 window keeps 2.25% of the scene, so a miss is catastrophic, and §6D
showed coverage governs the sign. What the paper can claim is bounded and true:

- Gated allocation **beats the budget axis on V\*Bench (+5.4pp) at 30% fewer tokens than the ungated
  policy**, and is the only policy in the prior-art table with a positive margin there.
- At 4K it reaches **break-even (−0.2pp)** — no worse than spending the tokens, at lower cost — while
  every published policy tested either loses significantly (Zoom Eye −15.3pp) or is statistically
  indistinguishable from the bar (grounding +1.3pp, n.s.).
- Gating is a **large, free improvement to attention-based allocation at both scales** (+7.9pp
  V\*Bench, **+17.0pp** at 4K) and cuts Zoom Eye's cost by 53% while recovering 5.7pp.

**What is NOT claimed:** that the method beats the budget axis at 4K. It does not, and four
independent attempts confirm it.

---

## §9D  ⚠⚠ THE SCALE SWEEP REFUTES §9C's MECHANISM — and the coverage account absorbs it (Phase 57)

**MMBench was checked first and rejected**, before any GPU was spent: its images are capped at
**512px** (median 512, 0% above 800px), so `uniform@300` already resolves them and there is no
allocation headroom. A null there would have been a benchmark-selection artifact.

Instead: **downsample HR-Bench's 4032² images**. Same 800 rows, same questions, same answers, at
1008 / 2016 / 4032 px — scale isolated with everything else held fixed, which no cross-benchmark
comparison can do. The bar is measured at each scale from that scale's own uniform arms.

### The prediction, and its refutation

§9C claimed the boundary was about **image scale**: as scale grows, (a) the budget axis becomes more
productive and (b) the localizer's cell grows in absolute pixels, so the proposal degrades. That
predicts the margin should **rise as scale falls**.

| scale | localizer cell | budget-axis gain (1×→4×) | gated margin |
|---|---|---|---|
| 1008px | **59px** (4× finer) | +9.0pp | **−3.6** [−7.1,−0.1] |
| 2016px | 119px | +9.6pp | +0.1 [−3.4,+3.6] |
| 4032px | 237px | +11.8pp | −0.2 [−3.6,+3.3] |

**It does the opposite, and it is not monotone in scale.** At 4× finer localizer cells the margin is
*worse*. **Proposal precision is not the binding constraint**, and the budget-axis term is nearly
flat (+9.0/+9.6/+11.8). **§9C's mechanism is wrong and is retracted.**

A confound in the design, stated plainly: **downsampling is not the same as a natively-smaller
image.** At 1008px a W=0.15 crop is 151px and there is no detail left to reveal — the downsample
destroyed it. So this sweep varies *available detail* as well as scale, and cannot cleanly test a
pure scale hypothesis. It can and does refute the precision half of it.

### Stratifying by category dissolves the boundary into §6D

HR-Bench is **50% `cross`** (multi-region) — and §6D says a single crop cannot cover a dispersed
evidence set, at any scale.

| scale | **cross** (multi-region) | **single** (one region) |
|---|---|---|
| 1008px | −4.4 | −2.4 |
| 2016px | −1.6 | **+1.6** |
| 4032px | −4.0 | **+3.5** [−1.2,+8.3] |

`cross` is negative at every scale and roughly flat; `single` rises with scale and turns positive.
**The pooled −0.2pp at 4032 is simply the average of +3.5 and −4.0.** No separate "scale boundary"
mechanism is needed — it is the coverage account, showing through a benchmark that is half
multi-region.

**Unified statement (replacing §9C):** allocation pays when **both** conditions hold — (1) the
evidence is confined to **one region**, so a single window can cover it; and (2) there is **detail to
reveal**, i.e. the evidence is sub-token at the base budget in the *native* pixels. V\*Bench satisfies
both. HR-Bench `cross` fails (1) at every scale. Downsampled HR-Bench fails (2). HR-Bench `single` at
native satisfies both, and there allocation wins.

### The significance test, and it still does not clear

Pooling the pre-specified qualifying regime (single-region, native resolution):

| stratum | n | acc | bar | margin | 95% CI |
|---|---|---|---|---|---|
| V\*Bench direct_attributes | 115 | 64.3% | 57.4% | +7.0 | [−1.7,+15.7] |
| HR-Bench single @4032 | 400 | 61.5% | 58.4% | +3.1 | [−1.7,+7.8] |
| **POOLED** | **515** | — | — | **+3.9** | **[−0.3,+8.2]** |

**Not significant** — the lower bound is −0.3. Reported alongside the all-items results (V\*Bench
+5.4pp [−1.4,+12.2]; HR-Bench −0.2pp [−3.6,+3.3]), none of which clears zero either.

⚠ **An error caught before it reached the paper.** A first version of this pooled test computed the
bar from **all** items while evaluating on the `single` subset. Single-region items are easier, so
the all-items bar understates what uniform achieves on them, and the margin came out **+5.2pp
[+1.1,+9.4] — "SIGNIFICANT"**. With the bar computed on the same subset it is +3.9pp [−0.3,+8.2] and
it is not. The bar must always be measured on the items being scored.

**Standing conclusion: the method's own positive result is not statistically established at any n we
have reached (515 in its best regime, 800 overall). The paper's significant results remain the
negative and mechanistic ones.**

---

## §11A  ★★★ THE METHOD RESULT: multi-crop in ONE pass (Phase 58)

### The idea, and why it was available all along

The binding constraint was never packaging or detection — it was **proposal quality**. The GT cell
ranks in the **top 3.4%** of the attention map, so the ranking contains far better windows than the
top-1 rule uses: oracle-among-top-5 reaches **61.5%** coverage against the argmax's 40.5%. Every
earlier attempt to exploit that paid a forward pass per candidate, and §9A showed the budget axis
outruns that.

**It does not have to cost passes.** The processor accepts several images in ONE forward pass, so
**k windows at B₀/k tokens each is one pass at the same total budget**. No selection is needed — the
model sees all k and answers from whichever contains the evidence. Resolution is still gained: a
W=0.15 window is 2.25% of the image, so at 75 tokens it carries 33 tokens per % of area against
uniform@300's 3.

Measured effect on the mediating variable: **coverage rises 40.4% → 55.7%** by handing over four
windows instead of one.

### The decisive contrast — same passes, same tokens, 4 windows vs 1

n = 191 V\*Bench, Qwen3-VL-2B, bar measured in-run.

| arm | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| uniform@300 / @600 / @1200 | 1 | 296/598/1182 | 56.5/63.9/71.7% | — | — |
| top1@0.15 (previous method) | 2 | 590 | 60.7% | 63.7% | −3.0 |
| **multi4** | **2** | **601** | **68.6%** | 63.9% | **+4.7** [−2.2,+10.9] |
| multi4_600 | 2 | 896 | 71.7% | 68.5% | +3.2 |
| oracle (uses GT) | 1 | 299 | 93.2% | 56.7% | +36.5 |

> **multi4 − top1@0.15 = +7.9pp, CI [+1.0,+14.7], SIGNIFICANT** — at 601 vs 590 tokens and the same
> two passes. The proposal bottleneck **is** buyable, by handing the model the candidates rather
> than choosing among them.

### ★ In the regime the mechanism predicts, it beats the budget axis

§7 says allocation pays when the evidence is confined to one region. Bar computed **on the same
subset** (an all-items bar would understate what uniform achieves on these items — the error caught
in §9D):

| single-region (n=115) | tokens | acc | bar | **margin** |
|---|---|---|---|---|
| top1@0.15 | 590 | 63.5% | 62.4% | +1.1 n.s. |
| **multi4** | 600 | **72.2%** | 62.6% | **+9.5 [+0.8,+17.4] SIG** |
| **multi4_600** | 897 | **77.4%** | 66.7% | **+10.6 [+2.8,+18.5] SIG** |

| multi-region (n=76) | | | | |
|---|---|---|---|---|
| multi4 | 603 | 63.2% | 65.9% | −2.7 n.s. |
| top1@0.15 | 590 | 56.6% | 65.8% | −9.2 n.s. |

**It fails on multi-region exactly where §6D says it must** — four windows still cannot cover a
dispersed evidence set — and the failure is milder than single-window (−2.7 vs −9.2), as more
coverage predicts.

### Honest scope

- The **pooled** V\*Bench margin is +4.7pp [−2.2,+10.9] — **not** significant. The significant
  budget-axis win is in the **single-region subgroup (n=115)**, which is **pre-specified by §6D/§7's
  mechanism** and by V\*Bench's own category annotation, not chosen after the fact. Both numbers are
  reported.
- The CI lower bound is **+0.8pp** — marginal. n=115.
- The **gated** variant is *not* significant (+7.7pp [−1.0,+16.4]); the significant arm is **ungated
  multi4 at ~600 tokens / 2 passes**. Gating saves tokens but costs the significance here, the
  reverse of the single-window case.
- `multi3_scene` (scene + 3 windows) is **negative** (−3.2pp): spending half the budget on the scene
  defeats the point. The gain comes from *more candidates*, not from retaining context.
- **Not yet tested at 4K.** The natural validation is multi-crop on HR-Bench `single`.

---

## §11B  MULTI-CROP AT 4K: the innovation transfers, the method's win does not (Phase 59)

HR-Bench 4k **`single`** — the qualifying regime (single-region, native 4032px), n = 400 rows / 100
instances, CircularEval. Bar computed **on this subset**; uniform@600/@1200 joined from Phase 53 with
20 consistency checks passed.

| arm | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| uniform@300 / @600 / @1200 | 1 | 290/578/1223 | 53.8 / 64.5 / **72.0%** | — | — |
| top1@0.15 | 2 | 580 | 49.8% | 64.5% | **−14.8** [−19.8,−10.0] |
| multi3 | 2 | 590 | 57.8% | 64.7% | −7.0 [−12.0,−2.2] |
| **multi4** | 2 | 611 | 57.8% | 65.1% | **−7.3** [−12.3,−2.6] |
| multi4_600 | 2 | 867 | 61.5% | 68.6% | −7.1 [−11.8,−2.3] |
| gated multi4 | 1.4 | 418 | 63.5% | 59.5% | **+4.0** [−0.7,+8.8] |

### 1. ✅ The multi-crop innovation transfers, and is significant at 4K

> **multi4 − top1@0.15 = +8.0pp, CI [+3.0,+13.0]** (and multi3 − top1 = +8.0pp [+4.0,+12.2]) — same
> passes, same tokens, k windows instead of one.

This replicates V\*Bench's +7.9pp [+1.0,+14.7] almost exactly, at 2.7× the image scale and on a
different benchmark. **Handing the model k candidates instead of one is a real, scale-general
improvement to attention-based allocation.** The proposal bottleneck is buyable at both scales.

### 2. ❌ But allocation still loses to the budget axis at 4K

multi4 is **−7.3pp [−12.3,−2.6]** against its bar — significantly negative. The reason is visible in
the bar itself: on this subset uniform goes **53.8 → 64.5 → 72.0%**, i.e. **+10.7pp for 2× tokens and
+18.2pp for 4×**. That is a steeper return than any crop policy we have produced. CircularEval says
the same: uniform@1200 63.0% against multi4's 43.0%.

**So the V\*Bench win (+9.5pp on single-region) is scale-bound.** Even with the better proposal, and
even restricted to the regime the mechanism qualifies, allocation does not beat spending the tokens
at 4032px.

`gated multi4` reaches **+4.0pp [−0.7,+8.8]** — the best 4K margin obtained across all five method
variants tried (previous best −0.2pp) — but its lower bound is −0.7 and it is **not significant**.

### The honest two-part claim this supports

- **Method contribution (significant, both scales):** multi-crop improves attention-based allocation
  by **+8pp** over the single-window policy at equal passes and tokens — +7.9pp [+1.0,+14.7] on
  V\*Bench, +8.0pp [+3.0,+13.0] at 4K.
- **Scope (significant, both directions):** allocation beats the budget axis at ~1500–2000px in the
  single-region regime (**+9.5pp [+0.8,+17.4]**) and **loses to it at 4032px** (**−7.3pp
  [−12.3,−2.6]**), because the uniform return on tokens is far steeper there.

The boundary is a property of the **budget axis**, not only of the proposal: at 4K, uniform scaling
returns +18.2pp for 4× tokens, and no placement policy we or the literature has produced matches it.

---

## §12A  ⛔ RETRACTED (see §12D): apparent read-out gap was a normalisation bug

**Why this was missing.** §10A–C concluded the information is *absent* from the representation, but
all three are **interventions** — they change behaviour and infer what must be encoded. None *reads*
the representation. The logit lens does, and it can falsify §10 directly: if the correct option were
decodable at an intermediate layer under `uniform` but lost by the output, the deficit would be a
**read-out** failure, not an information failure.

Method: same prompt twice at 300 tokens (uniform image, oracle crop), last token's residual captured
at **every** layer, projected through the model's own final norm and unembedding, restricted to A/B/C/D.

### The answer forms abruptly at L22, and the crop's advantage arrives with it

| layer | uniform | oracle | oracle − uniform |
|---|---|---|---|
| L0–L20 | 33.8% (flat) | 33.8% (flat) | ≈ 0 |
| **L22** | 47.1% | **88.2%** | **+41.2** |
| L24 | **52.9%** | **92.6%** | +39.7 |
| L27 (final) | 47.8% | 89.0% | +41.2 |

*(sub-token stratum, n=136; chance 25%)*

Both curves sit at chance-plus through L20. **The oracle advantage first exceeds 10pp at L21** and is
≈0 before it. So the crop's information is not "present early and read out late" — it becomes
decodable exactly when the answer does.

### §10 survives in substance, but its phrasing was too strong

**Under `uniform`, no layer exceeds 52.9%, while the oracle reaches 92.6%.** The crop supplies
information that is not decodable anywhere in the uniform run — §10's claim, now confirmed by
**direct measurement** rather than inferred from intervention nulls, which is the stronger form.

**But a real read-out gap exists.** Layer chosen on training folds, evaluated out of fold
(**L24 selected in all 5 folds** — a stable property of the model, not fold noise):

| stratum | final layer | CV best layer (L24) | delta |
|---|---|---|---|
| all items (n=191) | 52.4% | 56.5% | +4.2pp [−1.0,+9.9] n.s. |
| **sub-token (n=136)** | 47.8% | **52.9%** | **+5.1pp [+0.7,+10.3] SIG** |

Layer *ensembling* does not help (mean of last 3: −0.7pp; last 5: +1.5pp, both n.s.) — it is
specifically L24.

**So §10B's phrase "absent from the representation, not merely mis-weighted" is qualified**: ~12% of
the oracle gap *is* recoverable by a better read-out. The remaining 88% is not, at any layer.

### Two consequences

1. **A free +5.1pp on sub-token items**, at zero extra compute — decode from L24 rather than L27.
   Orthogonal to allocation, so it should compose with multi-crop (§11A), though that combination is
   **not yet tested**.
2. **The last three layers destroy signal on sub-token items** (52.9% → 47.8%). Why late layers hurt
   here is not established and is stated as an open question, not an explanation.

**Methodological note:** this should have been run before §10's conclusion was written. An
intervention null shows that a manipulation does not help; it does not show that the information is
absent. Reading the representation is the direct test, and it changed the claim.

---

## §12B  ⛔ RETRACTED (see §12D): built on the same corrupted final-layer read-out

Diagnosing §12A's read-out gap. Four hypotheses, each with a distinct offline signature.

### The last layers help when vision is decisive and hurt when it is not

| condition | stratum | L24 | final | drop |
|---|---|---|---|---|
| uniform | **sub-token** | 52.9% | 47.8% | **+5.1** |
| uniform | resolvable | 65.5% | 63.6% | +1.8 |
| oracle | sub-token | 92.6% | 89.0% | +3.7 |
| **oracle** | **resolvable** | 94.5% | **96.4%** | **−1.8** |

**H2 (evidence-independent degradation) is refuted**: where the evidence is strongest — oracle crop
on resolvable targets — the final layers *improve* accuracy by 1.8pp. The damage is concentrated
exactly where visual evidence is weakest.

**H4 (noise) is refuted on the stratum that matters**: on sub-token items the churn is asymmetric,
9 right→wrong against 2 wrong→right (net −7); on resolvable items it is symmetric (9 vs 8, net −1).

**The flips concentrate on one option.** Where L24 is right and the final layer is wrong, the final
layer answers **'B' on 78% of sub-token flips against a 37% base rate** (2.1× enrichment). Flipped
items also had smaller L24 margins (median 0.651 vs 0.998 unflipped), so the override happens where
the model was least certain — consistent with a fallback rather than a recomputation.

### But it is NOT a static prior — the obvious fix fails

The natural intervention from that diagnosis is to estimate the option prior on training folds and
divide it out at inference: free, one vector, no truncation, no extra pass. It does not work.

| rule | sub-token acc | vs final | depth |
|---|---|---|---|
| final (baseline) | 47.8% | +0.0 | 100% |
| final + prior removal | 48.5% | +0.7 [−1.5,+3.7] n.s. | 100% |
| **L24 truncation** | **52.9%** | **+5.1 [+0.7,+10.3] SIG** | **89%** |
| L24 + prior removal | 52.9% | +5.1 [+0.7,+10.3] SIG | 89% |

A constant marginal correction recovers almost none of the gap, and adds nothing on top of L24. **So
the bias is input-dependent, not a fixed offset** — the late layers move toward a default answer in a
way that interacts with the item. H1 holds in its evidence-dependent form and is refuted in its
static-prior form.

### The method that survives: truncate at L24

**+5.1pp on sub-token items, out of fold, at 11% LESS compute** (3 of 28 layers skipped). L24 was
selected in all 5 folds — a stable property of the model, not fold noise. Other free depth rules were
tested and do not beat it: `max_conf` +4.4pp n.s. (81% depth), `max_conf_late` +4.4pp n.s.,
`stable_3` **−13.1pp** (significantly worse).

**Scope, stated:** the effect is significant on the **sub-token stratum** (n=136); over all items it
is +4.2pp [−1.0,+9.9], n.s. Per-item oracle layer selection would reach 85–87% (+37pp), so the
headroom here is enormous but not reachable by any label-free rule we found — the gap between
`oracle_layer` and every real rule is the honest measure of how little of it we can capture.

**Open question, not explained:** *why* late layers trade visual evidence for a default under weak
evidence. We measure it; we do not account for it.

---

## §12C  ⛔ RETRACTED (see §12D): DoLa/DeCo were scored against a corrupted baseline

A literature survey established that our logit-lens territory is occupied — **DoLa** (2309.03883,
ICLR 2024) contrasts a premature against the final layer *and applies it to multiple choice*, so our
MCQ setting is **not** a differentiator; **DeCo** (2410.11779, ICLR 2025) already reports that MLLMs
recognise objects in preceding layers and that later layers suppress it under language priors,
exploiting layers 20–28 of 32; **Confident Decoding** (2606.21906) already claims "deeper is not
always better" in text LLMs; **Anchored Answers** (2405.03205) already localised MCQ letter bias with
the logit lens. **"Truncation improves VLM accuracy" is therefore NOT a novel finding and is not
claimed.**

Both scoring rules were reimplemented on our per-layer read-outs, with **their hyper-parameters swept
and their best reported** (which favours them; ours is fixed).

### Sub-token stratum, n = 136, arm = uniform@300

| method | acc | vs final |
|---|---|---|
| final layer (baseline) | 47.8% | — |
| DoLa dynamic (best of 3 buckets) | 49.3% | +1.5 n.s. |
| DoLa-static (oracle-picked layer) | 48.5% | +0.7 n.s. |
| DeCo (best band + α) | 52.2% | +4.4 [−0.7,+9.6] n.s. |
| **ours: fixed L24** | **52.9%** | **+5.1 [+0.7,+10.3] SIG** |

DoLa gives essentially nothing here — consistent with **ICLA (2603.00437)**, which reports DoLa
*collapsing* on Qwen2.5-VL and DeCo degrading. **We do not claim to beat DeCo**: +5.1 vs +4.4 with
overlapping CIs and swept hyper-parameters on their side. Competitive with DeCo, clearly better than
DoLa, and the only arm whose CI excludes zero.

### ★ The result that matters: the decoding channel is capped at ~5pp

| | sub-token (n=136) |
|---|---|
| best layer-decoding method, any | **≈ +5pp** |
| **allocation (multi4)** | **+19.9pp** [+11.0,+28.7] |
| **allocation + L24 read-out** | **+25.7pp** [+16.9,+34.6] |

**Every layer-based decoding method operates on a single forward pass that never encoded the
information.** §10A–C established that ceiling by intervention (sink suppression null, attention
amplification 16% of the crop, residual steering +0.0pp); §12A measured it directly with the logit
lens (under uniform, *no* layer exceeds 52.9% while the oracle crop reaches 92.6%). **This phase
measures what the published methods extract from that channel: ~5pp, and no more.**

Allocation changes *what is encoded* and delivers **4× more**, and the two **compose** (§11A + §12A:
additive prediction +22.8pp, observed +24.3pp).

### Robustness of the read-out component, stated plainly

Across **20 CV seeds** on identical data, the L24 gain is **+4.0pp mean, range [+0.7,+5.1],
significant in 13/20 splits** — the variance is entirely whether a fold selects L21 instead of L24.
The earlier "+5.1pp SIG" was a favourable split. By contrast **allocation is +19.9pp with nothing
fitted and no fold split at all.** The read-out is a small, somewhat fragile bonus; allocation is the
method.

### What remains novel from the logit-lens work

Per the survey, only the **evidence-dependent sign flip**: the final layers **help** when visual
evidence is strong (oracle + resolvable, −1.8pp drop, i.e. final is *better*) and hurt when it is
weak. DeCo and DAMO treat late-layer suppression as a uniform pathology; the oracle-crop
counterfactual is the instrument that exposes the reversal, and no surveyed paper reports it. The
static-prior-fails-while-truncation-works ablation (§12B) is also unclaimed. The abrupt L22
answer-formation with ≈0 oracle advantage before it is novel as measured but weak on its own.

---

## §12D  ⛔ RETRACTION: §12A–C were a double-normalisation bug in our own logit lens (Phase 64)

### The bug

Our lens computed every layer's read-out as `lm_head(final_norm(hidden_states[i]))`. For
**intermediate** layers that is correct. For the **last** one it is not: HuggingFace returns
`hidden_states[-1]` with the final norm **already applied**, so we normalised it twice.

Verified directly: `max |lm(h_last) − true_logits| = 0.06` versus
`max |lm(norm(h_last)) − true_logits| = 23.47`.

**Only the final layer was corrupted**, which is exactly why the error hid so well — Phase 60 and
Phase 64 agree on the L24 read-out for **100%** of items and on the final layer for only **82.2%**.

### What it produced, and what is now retracted

| retracted claim | what it actually was |
|---|---|
| §12A "+5.1pp free read-out gain at L24" | the final layer degraded by double-normalisation |
| §12B "the last layers inject an evidence-dependent answer bias" | L24 compared against that degraded layer |
| §12B the 'B'-concentration and prior-removal ablation | same corrupted comparison |
| §12C "our L24 beats DoLa, matches DeCo" | all scored against the corrupted baseline |
| §12C "the decoding channel yields ~5pp" | it yields **nothing** |

### The corrected measurements

With the model's own logits as the final layer:

| | uniform | oracle | best intermediate layer | final vs best |
|---|---|---|---|---|
| all items (n=191) | **56.5%** | 93.2% | L24 = 56.5% | **+0.0pp** [−2.1,+2.1] |
| sub-token (n=136) | **52.9%** | 92.6% | L24 = 52.9% | **+0.0pp** [−2.9,+2.9] |

**The final layer is already optimal** — even against the best intermediate layer chosen *in sample*,
an optimistic upper bound. There is no read-out gap.

Corrected prior-art comparison (sub-token; DoLa/DeCo hyper-parameters still swept in their favour):
DoLa dynamic **+0.0pp**, DeCo best-of-8 **−0.7pp**, read-at-L24 **+0.0pp**. **No layer-decoding rule
beats the true final layer.** Component ablation agrees independently: zeroing attention or MLP in
L25/26/27, singly or together, moves accuracy by ≤2.4pp and never significantly (Phase 64).

### What survives, and what it means

**Survives** (intermediate layers were never affected): the answer forms **abruptly at L22** — under
uniform 39.8% → 51.3% from L20 to L22, under oracle 41.9% → 90.1% — and the **oracle advantage is
≈0 before L21** (+0.0 to +6.6pp) and **+38.7pp at L22**. Visual evidence becomes decodable only at
answer-formation time, not gradually.

**§10 is strengthened, not qualified.** The earlier qualification is reverted. The information the
crop supplies is not recoverable at *any* layer, by *any* decoding rule we or the literature has
tried — now established by direct measurement, by component ablation, and by intervention nulls, all
agreeing.

**Unaffected:** every allocation result (Phases 47, 53, 58, 59) reads the model's own `logits` and
never touched the lens. Multi-crop's +7.9pp (V\*Bench) and +8.0pp (4K) over single-crop stand.
Phase 62's composition claim is **withdrawn** — its read-out component was the artifact.

### Process note

This is the **fourth** analysis bug caught in this project and the most consequential: it survived
four phases and produced a coherent, mechanistically plausible, publishable-looking story
(late-layer bias, evidence dependence, a letter prior, a free method beating DoLa). What exposed it
was not review but **an independent reimplementation** — Phase 64 computed the same quantity a
different way, and `both@late` ≡ `trunc_L24` served as an internal consistency check. **A derived
quantity that only one code path ever computes is not verified.**

---

## §13  ★★★ THE FOUNDING QUESTION ANSWERED: localisation is EMPTY, not mis-routed (Phase 66)

This project began from a dissociation: the ring-masked attention map ranks the GT-centre cell at
**gt_pct 0.034** — the top 3.4% of cells against an exact chance of 0.500 — yet uniform@300 answers
only 56.5% correctly. **The model looks in roughly the right place and is still wrong.** Until the
logit lens we could only characterise that from the outside.

*(Intermediate layers from the Phase 60 lens; the FINAL layer from the model's own logits, because
the lens double-normalised it — §12D.)*

### The dissociation, as a 2×2 (n=191)

| | answered RIGHT | answered WRONG |
|---|---|---|
| **localised** (window covers GT) | 59 | **36** |
| window missed | 49 | 47 |

### Two explanations, and the lens separates them

| cell | n | correct option is argmax at *some* layer |
|---|---|---|
| localised & right | 59 | 100.0% |
| **localised & WRONG** | **36** | **72.2%** |
| missed & WRONG | 47 | 68.1% |
| missed & right | 49 | 100.0% |

**Localising buys almost nothing in internal decodability**: 72.2% vs 68.1% for items where the
window missed entirely.

An apparent counter-signal had to be ruled out. On localised-but-wrong items the correct option is
argmax on **41.7% at L20**, collapsing to **0% from L22** — which looks like propagation failure.
It is not:

| layer | accuracy, all items | accuracy on localised & wrong | difference |
|---|---|---|---|
| L18 | 37.2% | 38.9% | **+1.7** |
| L20 | 39.8% | 41.7% | **+1.9** |
| L22 | 51.3% | 5.6% | −45.8 |

At L18–L20 these items score **exactly what an uninformative layer produces by chance** — those
layers are near-chance overall, so their argmax is near-random and will look "correct" on ~40% of any
wrong-answered subset. **There is no layer at which the answer is present and subsequently lost.**

### The decisive contrast

On the localised-but-wrong cell, the **oracle crop — same question, same 300-token budget, target
actually resolved — is 94.4% correct** [86.1,100.0]. Median tokens on target for this cell: **0.14**.
For the missed-and-wrong cell the oracle reaches 95.7% at 0.18 tokens on target — statistically the
same. **Localisation quality does not determine whether the item is fixable; resolution does.**

### The answer

> **The model localises but answers wrong because the localisation is EMPTY.** At B₀ = 300 the target
> occupies a median of **0.14 merged tokens**, so attending to the correct cell returns a cell that
> does not contain the answer. The correct option is decodable at **no layer** — the model is not
> mis-routing information it has; it never had it. Cropping to that same region supplies the missing
> pixels and fixes **94.4%** of these items.

"Localises well" never implied "has the answer". The two are different capacities, and the gap
between them is **resolution**, not routing.

### Why this unifies the whole project

- **§10** — three internal interventions (sink suppression, attention amplification, residual
  steering) all fail: there is nothing inside to redirect.
- **§12D** — no read-out gap at any layer, by any decoding rule including DoLa and DeCo: there is
  nothing inside to decode.
- **§6D** — coverage governs the sign of the allocation effect: what matters is whether the window
  *supplies* the evidence, not whether attention *points at* it.
- **§11A** — the method works by adding pixels to candidate regions, which is the only operation that
  changes what is encoded.

Every negative result in this project is the same fact seen from a different side, and §13 states it
directly: **attention localises; tokens carry the answer; below one token there is no answer to
carry.**

---

## §13B  ★★★ THE CLIFF: localisation goes empty below 0.15 merged tokens (Phase 66)

§13 says localisation is empty *because* the target is sub-token. That predicts a **threshold** —
and locating it turns the project's central quantity from a regime label into a number.

### A confound first

A naive sweep over all localised items is non-monotone: the oracle gain is +63.9 below 0.25 tokens,
falls to ~+7, then **jumps back to +53.8 at [1.0, 4.0)**. That is **composition, not a reversal**:

| bin | n | direct_attributes | relative_position |
|---|---|---|---|
| [0, 0.25) | 36 | 33 | 3 |
| [1.0, 4.0) | 13 | 6 | **7** |
| ≥4.0 | 17 | **0** | **17** |

For multi-region questions the "target" box is the **union of two objects**, so tokens-on-target does
not mean what it means for single-region items. The curve must be read on single-region items only.

### The cliff, on single-region localised items (n=63)

Sliding threshold *t*; uniform accuracy below vs at-or-above:

| t (merged tokens) | n below | acc below | n above | acc above | step |
|---|---|---|---|---|---|
| 0.10 | 16 | 18.8% | 47 | 72.3% | +53.6 |
| **0.15** | **21** | **19.0%** | **42** | **78.6%** | **+59.5** |
| 0.25 | 33 | 33.3% | 30 | 86.7% | +53.3 |
| 0.50 | 46 | 47.8% | 17 | 88.2% | +40.4 |
| 1.00 | 57 | 56.1% | 6 | 83.3% | +27.2 |

**Sharpest step at t = 0.15 merged tokens: 19.0% → 78.6%, CIs [4.8,38.1] and [66.7,90.5] —
disjoint.**

### The control that makes it a statement about encoding, not difficulty

**The oracle crop is FLAT across the same split: 100.0% below t, 97.6% above.** The answer is fully
available at every target size once the region is resolved. **Only the uniform arm collapses.** So
the step is not task difficulty and not annotation noise — it is entirely whether the uniform
encoding captures the target.

### What 0.15 merged tokens means

A merged token at B₀ = 300 covers 1/300 of the image area, i.e. ~5.8% of a side. **t = 0.15 tokens is
a target spanning ~2.2% of the image side** — about 33 px on a 1500 px image.

**Below the cliff the model scores 19.0%, which is *below* the 25% chance level.** It is not merely
uninformed: with the evidence unresolved it is systematically misled, presumably toward whichever
option the language prior and the unresolved scene favour.

### The statement this supports

> A VLM's visual question answering has a **sharp encoding threshold**. When the queried evidence
> occupies more than ~0.15 merged tokens it answers correctly ~79% of the time; below that it drops
> to ~19%, worse than chance — while the identical question with the identical budget, applied to the
> resolved region, is answered ~100% correctly at **every** target size. Attention still localises
> correctly across the cliff (§13); what changes is whether anything is there to read.

---

## §13C  ★★★ THE CLIFF IS ARCHITECTURE-INVARIANT (two models, same threshold)

§13B located the encoding cliff at ~0.15 merged tokens on Qwen3-VL-2B's *localised* single-region
items (n=63). Running the identical sweep on **all** single-region items, and on a second
architecture:

| model | n | cliff | uniform below | uniform above | CIs | **oracle below / above** |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 115 | **0.25 tok** | 32.4% [21,44] | 77.3% [64,89] | **disjoint** | 95.8% / 100.0% |
| Qwen2-VL-7B | 115 | **0.25 tok** | 36.6% [25,48] | 63.6% [50,77] | **disjoint** | 95.8% / 100.0% |

**The same threshold, on both.** Different model generation, different vision stack, **3.5× the
parameters** — and the step lands in the same place, with disjoint confidence intervals on each.

**The control is what makes it a claim about encoding.** The oracle crop is **flat across the split
on both models, at identical values (95.8% below, 100.0% above)**. The questions below the cliff are
not harder: given the same 300-token budget spent on the resolved region, both models answer them
essentially perfectly. Only the uniform arm steps, and it steps at the same target size in both.

### The consolidated statement

> A VLM's visual question answering has a **sharp, architecture-invariant encoding threshold** at
> roughly **0.15–0.25 merged tokens** of target extent (~2–3% of the image side at B₀ = 300). Above
> it, accuracy is 64–79%; below it, 32–37% overall and **19% on the cleanly-localised subset — below
> the 25% chance level**. The identical question at the identical budget, applied to the resolved
> region, is answered 96–100% correctly **at every target size**. Attention localises correctly on
> both sides of the cliff (§13); what changes is whether anything is encoded to read.

The exact threshold depends on the subset — 0.15 on localised single-region items (n=63, the
cleanest cut, where the step is 19.0% → 78.6%), 0.25 on all single-region items (n=115). Both are
consistent, and the sweep is reported in full rather than a single chosen value.

### Why this is the paper's core finding

It is the thing every other result is downstream of. The exchange rate (§2) is large because
allocation moves targets across this cliff. Coverage governs the sign (§6D) because a window that
misses leaves the target below it. Internal interventions fail (§10) and no read-out recovers
anything (§12D) because below the cliff there is nothing encoded. The method works (§11A) because
adding pixels is the only operation that moves a target across it. And it is not a quirk of one
checkpoint: **two architectures, one threshold.**

---

## §12E  ✅ PHASES 61 AND 62 RE-RUN CORRECTLY (61b, 62b)

The retracted phases redone with the final layer taken from the model's own logits instead of
`norm(hidden_states[-1])`. Intermediate layers come from the original lens runs (they were never
corrupted); the final layer is joined from runs that used `model.logits`.

**Join checks, asserted before anything is reported:**
- intermediate layers agree across the two lens runs: **99.5%** ✅
- phase-62's own final layer vs `model.logits`: **81.2%** — the disagreement *is* the bug, and its
  absence would have meant the join pointed at the wrong quantity ✅

### 61b — there is no read-out gap, and layer selection actively HURTS

| stratum | n | final layer | CV-chosen layer | delta |
|---|---|---|---|---|
| all items | 191 | 56.5% | 56.0% | −0.5 [−2.6,+1.0] |
| sub-token | 136 | 52.9% | 51.5% | −1.5 [−3.7,+0.0] |
| **resolvable** | 55 | **65.5%** | 58.2% | **−7.3 [−14.5,−1.8] SIG** |

Stronger than "no gap": on resolvable items, choosing a layer out of fold is **significantly worse**
than simply reading the final layer. The folds pick L21 there, which looks good in-fold and
generalises badly.

**All four Phase-61 hypotheses collapse**, because there is no gap to explain:

| test | original (corrupted) | corrected |
|---|---|---|
| L24 vs final, every condition × stratum | +5.1 / +1.8 / +3.7 / −1.8 | **+0.0 in all four cells** |
| flips where L24 right, final wrong | 9 sub-token, 9 resolvable | **2 and 0** |
| letter concentration | **'B' 78% vs 37% base** | n_flip=2; nothing to concentrate |
| churn symmetry | 9 right→wrong vs 2 wrong→right | **2 vs 2, net 0** |

No free depth rule beats the final layer: `fixed_cv` −0.5/−1.5, `max_conf` −1.0/−0.7,
`max_conf_late` −1.0/−0.7. The `oracle_layer` ceiling is +30.4pp, so the headroom is real but
unreachable by any label-free rule.

### 62b — nothing to compose with; the allocation effect stands

| effect vs uniform/final | all items | sub-token |
|---|---|---|
| **allocation alone (multi4)** | **+12.0 [+4.2,+19.9] SIG** | **+19.9 [+11.0,+28.7] SIG** |
| read-out alone (CV layer) | −1.0 [−2.6,+0.0] | −1.5 [−3.7,+0.0] |
| both | +12.0 | +22.1 |

The composition claim is **void** — the read-out term is slightly negative, so the combination is
just allocation.

**The allocation measurement is unchanged** (+19.9pp on sub-token, identical to the original run).
Both arms had been corrupted **identically**, so their difference survived the bug untouched. The
measurement was never wrong; only the interpretation built on the read-out half was.

### What this settles

Combined with §12D (no gap), §12C-corrected (DoLa +0.0, DeCo −0.7) and Phase 64 (component ablation
null), the layer-decoding direction is closed on this task by four independent measurements. And
§14 explains *why*: answer formation is a step, not an accumulation, so there is no premature layer
carrying partial signal for any layer-based rule to exploit.

## §14A  ✗ NEGATIVE: the cliff does NOT yield a budget-allocation rule (re-analysis of Phase 27)

Scoping test for the one remaining method family — a **per-item budget controller**: no crop, no
pixel preprocessing, just set `image_grid_thw` per item. Motivated by §13B's sufficient statistic
`tokens_on_target = area_fraction x total_tokens` with a threshold at t* ~ 0.15-0.25. If that is the
quantity that decides encodability, then `budget >= t*/area_fraction` should be the rule that tells
you how many tokens an item needs. **It is not.** Measured on phase 27's 7-rung uniform ladder,
n = 191, V\*Bench, ground-truth boxes used so the size estimate is PERFECT.

### 1. The ceiling of any budget controller is the uniform plateau, not above it

| | accuracy | mean tokens |
|---|---|---|
| uniform@8000 (plateau) | 84.8% | 7990 |
| **per-item best over the ladder (naive)** | **90.1%** | 1457 |
| **per-item best, MONOTONE-STABLE** | **84.8%** | **2418** |
| oracle crop@300 (for reference) | 93.7% | 300 |

The naive +5.2pp over the plateau is **noise harvesting**. 43/191 items (22.5%) are correct at some
rung and wrong at a *higher* one; picking whichever rung happened to be right is item-level test-set
selection on a 4-way MCQ. Requiring the item to be correct from some rung *onward* — the only
pattern a real controller could exploit — collapses the ceiling to **exactly uniform@8000's 84.8%**.

> **A budget controller cannot be an accuracy method.** Its ceiling is the plateau. Any headroom is
> in *tokens*, not points. Stated before running an estimator, not discovered after.

### 2. The efficiency headroom is real: 3.3x

The oracle monotone controller reaches the plateau's 84.8% at **2418 mean tokens vs 7990** — same
accuracy, **3.3x fewer tokens**, median spend 600. So an adaptive-budget method has something to win.

### 3. But target size does not unlock it — even with a perfect size oracle

Budget rule `b = t*/area_fraction`, t* swept, each arm scored against the uniform rung at its **own
mean budget** (the matched-bar discipline from §8's retraction):

| t* | all (n=191) | direct_attributes (n=115) | relative_position (n=76) |
|---|---|---|---|
| 0.20 | **−5.2pp** | −8.7 | −7.9 |
| 0.40 | **−5.2pp** | −6.1 | −9.2 |
| 0.80 | **−3.1pp** | +0.9 | −2.6 |
| 1.60 | +0.5pp | −1.7 | −1.3 |

**Eleven of twelve cells are negative or null.** A perfect-size budget rule loses to flat uniform.

### 4. Why, and it is not the ladder's coarseness

The oracle controller wins 3.3x **on the same 7 rungs**, so rung granularity is not the binding
constraint. The binding constraint is that the budget an item actually needs is **nearly flat in
target size**, across a 20x range of area fraction:

| area fraction | n | oracle mean spend |
|---|---|---|
| < 0.0005 | 58 | 3212 |
| 0.0005–0.002 | 56 | 2146 |
| 0.002–0.01 | 44 | 2318 |
| > 0.01 | 33 | 1615 |

### 5. What this costs us, stated plainly

§13B's synthesis paragraph reads that the cliff "is the thing every other result is downstream of."
That over-reaches by one step. **`tokens_on_target` predicts whether an item is ENCODABLE; it does
not predict the budget at which it becomes ANSWERABLE.** Those are different claims. The cliff
result stands as stated on its own subset (localised single-region, n=63, 19.0% -> 78.6%); what does
not follow — and is now refuted at n=191 with a perfect size oracle — is that inverting it gives an
allocation policy. The cliff is **necessary, not sufficient**.

### 6. Consequence for scope

This was the last untested member of the "internal, plug-and-play, no crop" method family. With
§10A/§10B/§10C (sink, attention, residual all null) and §12D (no read-out recovers anything), the
family is now closed **on measured evidence at its own ceiling**, not by assumption.


## §14B  ✗ NEGATIVE: no free pass-1 signal predicts the required budget (Phase 69)

§14A closed the *size*-based budget rule but left the family alive: it measured **3.3× of real
headroom** (oracle controller 84.8% @ 2418 tokens vs the plateau's 84.8% @ 7990) and showed the
failure was the *signal*, not the idea — required budget is nearly flat in target size. Phase 69
asks whether anything else free at pass 1 predicts it. **Nothing does.**

No GPU: 17 features per item, all read from artifacts already on disk — attention geometry from
`phase30c` (peak, top-5/top-20 mass, entropy, gini, spatial spread, peak/mean, 1-vs-5 ratio), answer
state from `phase60`'s logit lens (confidence, margin, entropy, cross-layer agreement, settling
depth), and free metadata (log pixels, aspect, category). Join verified by requiring `gt_area_frac`
to agree to 1e-9, since the three files spell question ids differently.

**Label** = the smallest rung from which the item is correct at that rung *and every rung above*.
Monotone stability is required because §14A found 22.5% of items flip correct→wrong going up the
ladder; regressing on "cheapest rung that happened to be right" fits that noise.

### The label is barely learnable, and not usably so

| | |
|---|---|
| best univariate ρ (answer entropy) | 0.340 |
| area fraction, for reference | 0.223 |
| **out-of-fold ρ, all 17 features** | **0.254** |
| OOF mean absolute error | **1.98 rungs** — wrong by ~4× in budget |
| exact-rung accuracy | **9.4%**, against a **35.6%** majority-class baseline |

The predictor is *worse than always guessing the modal rung*. Correlation exists; an exploitable
margin does not.

### The controller tracks the uniform ladder at every operating point

Predicted rung + a swept global offset, each point scored against the ladder interpolated at the
controller's **own** mean spend (n=191, out-of-fold, 5-fold × 20 repeats):

| offset | internals | size-only (§14A's refuted rule) | **shuffled-label CONTROL** |
|---|---|---|---|
| −1 | −5.7pp | −1.8 | −2.3 |
| +0 | −3.7pp | −1.9 | −1.4 |
| +1 | −1.6pp | +1.1 | −2.0 |
| +2 | **+1.0pp** | **+1.8** | +0.1 |
| +3 | −0.9pp | +1.0 | +0.0 |
| +4 | +0.7pp | +1.0 | +0.2 |

**The internals never beat size-only**, which §14A already refuted, and neither clears the shuffled
control by a margin worth reporting. Even a two-tier controller with its threshold swept *on the
test items themselves* — a cheating upper bound — reaches only **+1.0pp**.

### Why the headroom is unreachable

The required-rung distribution is bimodal: **68/191 items need only 150 tokens** and **37 need more
than the ladder has** (never stably correct). The middle is thin. The oracle's 3.3× comes from
knowing which of those two piles an item is in, and that is exactly the call no free signal makes.

> **The adaptive-budget family is now closed at both levels.** §14A closed it with a *perfect size
> oracle*; §14B closes it for *every free pass-1 statistic we can compute*. Together with §10A–C
> (sink, attention, residual all null) and §12D (no read-out recovers anything), the "internal,
> plug-and-play, no new pixels" method space is exhausted on measured evidence rather than assumed.


## §14C  ★★★ POSITIVE: a learned re-ranking head recovers 65% of the proposer's lost ceiling (Phase 70/70b)

The first positive method result that is **internal, plug-and-play, costs no extra forward pass, and
does not crop**. It is a fix to a *read-out*, not a new source of information.

### The gap it attacks

§11A measured that the proposer's ranking already contains the evidence while its **top-1 rule throws
it away**: the GT cell sits in the top 3.4% of cells (gt_pct 0.037), yet the argmax covers the
evidence only 39.3% of the time, against 60.2% for oracle selection among that same map's top-5.

### Result (n=191, out-of-fold, GroupKFold — an item's 295 cells never split across folds)

| arm | covers evidence | vs incumbent |
|---|---|---|
| deployed argmax (incumbent) | 39.3% | — |
| **learned head (attention + geometry)** | **52.9%** | **+13.6pp** |
| head, attention features only (no geometry) | 50.8% | +11.5 |
| *geometry only — centre-prior CONTROL* | **3.7%** | −35.6 |
| *shuffled labels — CONTROL* | **2.6%** | −36.6 |
| oracle among top-5 / top-10 / top-20 (restricted references) | 60.2 / 64.4 / 72.8% | |
| **TRUE ceiling: does ANY ring-masked cell cover?** | **88.5%** | |

The head captures **27.6% of the available headroom** (13.6 of 49.2pp to the 88.5% true ceiling).
A secondary reference: 65% of the *top-5* oracle gap — but that is a **restricted** comparison, not
a bound, since the head reorders all ~295 cells rather than only the deployed map's top 5, and can
in principle exceed it. **Both controls sit at chance**
(~3%, which is what a random cell scores given V\*Bench's tiny targets), so this is not a position
prior and the harness is not leaking. Attention features alone deliver +11.5pp of the +13.6.

### §14C(b) Why it works — and a correction to our own §5 (Phase 70b)

Feature-group ablation against two predictions the paper already makes:

| features | covers | Δ |
|---|---|---|
| deployed map only (value + 3×3 + position) | 45.0% | +5.8 |
| + sink indicators (last column / last row) | 44.5% | +5.2 |
| **+ DEPTH PROFILE (28 layers)** | **51.8%** | **+12.6** |
| + within-layer ranks [= full head] | 52.9% | +13.6 |
| full head **minus** sink indicators | **52.9%** | **+13.6** |

**P1 — "the block-16-26 mean destroys the depth profile" — CONFIRMED, +7.3pp.** This is the single
largest component. The deployed read-out averages 28 layers into one map and then takes a max; the
per-layer profile discriminates evidence from distractors and the mean erases it.

**P2 — "the serialization sink corrupts the argmax" — NOT SUPPORTED, +0.0pp.** Telling the head
exactly where the sink is buys *nothing*; removing those indicators costs *nothing*.

> ⚠ **Correction.** §5's sink is real as a description of attention mass (2.0–4.5× enrichment, four
> architectures, present at L0, every CI excludes 1.0). But it is **not the reason the proposer's
> top-1 fails**, and the §5→§6 link in PAPER_FLOW asserted that connection without testing it.
> Consistent with §10A, where suppressing the sink at inference also bought nothing. The sink is a
> serialization *artifact* we can characterise; it is not the *defect*. The defect is depth
> averaging.

### The trivial explanation is ruled out

A training-free blur of the same map does not reproduce the gain — it destroys it: argmax of a 3×3
blurred map scores **17.8%**, 5×5 **2.1%**, 7×7 **0.5%**, against the raw argmax's 39.3%. The head's
neighbourhood contribution is learned local *contrast*, not smoothing.

### Boundary, stated here rather than in review

This improves **proposal quality**, not task accuracy. A better proposal only converts if something
downstream acts on it, and the end-task gain is **unmeasured**. §6D's coverage→accuracy relation
(0% coverage −15.6pp, 100% +37.1pp, crossing zero near 25%) predicts 39.3%→52.9% should matter, but
that is a prediction, not a result. And it is **not** near its ceiling: 52.9% against a true
ceiling of 88.5% leaves 35.6pp of headroom unclaimed.


## §14D  ★★★ THE HEAD CONVERTS: allocation beats the budget axis on single-region questions (Phase 71)

The end-task measurement §14C said was missing. V\*Bench, n=191, Qwen3-VL-2B, bf16, every arm's
tokens **measured** from `image_grid_thw` (all within 2.0% of target, no arm voided). Proposals were
frozen out-of-fold in phase71a; **no fitting happens in the GPU script.**

### The pre-registered prediction hit

§6D's coverage strata (miss −15.6pp, full cover +37.1pp, a 52.7pp swing) and §14C's 39.3%→52.9%
coverage shift predicted **+7.2pp** for head over argmax. Observed: **+8.4pp [+2.6,+14.7]**. The
prediction was written into the script before the run and came from a phase that never saw this arm.

### Accuracy (n=191)

| arm | acc |
|---|---|
| uniform@300 | 56.5% |
| **uniform@600 (compute-matched bar)** | **63.9%** |
| argmax@0.15 (deployed allocator) | 60.2% |
| **head@0.15 (the method)** | **68.6%** |
| rand@0.15 (placement control) | 40.3% |
| oracle@0.15 (ceiling at this W) | 90.1% |

| contrast | Δ | CI |
|---|---|---|
| **head − argmax** | **+8.4pp** | **[+2.6,+14.7]** ✔ |
| head − uniform@600 (the bar) | +4.7pp | [−3.7,+13.1] ✗ |
| head − uniform@300 | +12.0pp | [+4.2,+19.9] ✔ |
| head − rand | +28.3pp | [+18.3,+37.7] ✔ |
| oracle − head (headroom left) | +21.5pp | [+14.7,+28.8] |

### ★ The claim the statistics carry: the regime split

| category | head | argmax | bar | **head − bar** |
|---|---|---|---|---|
| **direct_attributes** (single-region, n=115) | **75.7%** | 63.5% | 62.6% | **+13.0pp [+2.6,+23.5]** ✔ |
| relative_position (relational, n=76) | 57.9% | 55.3% | 65.8% | **−7.9pp [−21.1,+5.3]** ✗ |

> **On single-region questions a free internal re-ranking head makes allocation beat the
> compute-matched budget axis by +13.0pp, with the CI excluding zero. Nothing in §2 — not
> Zoom Eye, not grounding, not our own gated allocator — ever did that.**

**Pooled, the method does not clear its bar** (+4.7pp, lower bound −3.7). That is stated, not buried.
The boundary is exactly the one §3/§7 predict from coverage: relational evidence spans multiple
regions, one window cannot cover it, and the head's proposals there reach only 34.2% coverage
(from 19.7%) — barely past the ~25% zero-crossing.

### The internal control is exact

On the **79 items where head and argmax chose the same cell**, the two arms are the same computation
and split **+0.0pp, CI [+0.0,+0.0]**. On the 112 where they differ: **+14.3pp [+4.5,+24.1]**. The
effect lives entirely where the proposals actually diverge.

### Coverage is confirmed as the mediator, on an arm it never saw

| stratum | n | head | argmax | Δ |
|---|---|---|---|---|
| **head covers, argmax missed** | 31 | **90.3%** | 38.7% | **+51.6** |
| argmax covers, head missed | 5 | 80.0% | 80.0% | +0.0 |
| both cover | 70 | 90.0% | 91.4% | −1.4 |
| neither covers | 85 | 42.4% | 41.2% | +1.2 |

**All of the gain is in the coverage flips**; where coverage is unchanged the arms are
indistinguishable. This is §6D's account predicting a new arm's behaviour quantitatively.

### Limitations, stated here

- The head is trained on **V\*Bench's own GT boxes** with out-of-fold folds. Legitimate, but
  **cross-benchmark transfer is untested** and is the main exposure.
- Pooled margin vs the bar is **not significant**; only the single-region stratum is.
- **21.5pp of headroom remains** to oracle placement at the same window size — the head captures
  27.6% of the proposal ceiling (§14C), not most of it.
- One model, one benchmark, W fixed at 0.15 from earlier CV.


### §14D(b) Where the head pays: below the cliff, not above it

Splitting Phase 71's head−argmax contrast by `tokens_on_target` (§13B's cliff sits at 0.15–0.25):

| stratum | n | head | argmax | Δ |
|---|---|---|---|---|
| **below cliff** (<0.15) | 58 | 67.2% | 53.4% | **+13.8pp [+1.7,+25.9]** |
| **cliff zone** (0.15–0.25) | 25 | 64.0% | 48.0% | **+16.0pp [+4.0,+32.0]** |
| above cliff (≥0.25) | 108 | 70.4% | 66.7% | +3.7pp [−3.7,+11.1] n.s. |

**The head pays exactly where the encoding deficit is, and not elsewhere.** Above the cliff the
target is already resolved in the uniform image, so placement barely matters; below it the crop is
the only operation that resolves the target, so *where* it lands is decisive. This is the cleanest
statement of what the method does: it is a fix for sub-token evidence, not a general-purpose
re-ranker, and §13B predicts its own operating regime.

It also **refutes a tempting mechanism** for §14E's transfer failure. "Below the cliff there is
nothing encoded to find, so re-ranking cannot help" is false — below the cliff is precisely where
re-ranking helps *most*. Any account of the HR-Bench failure must survive this table.

## §14F  ★★ THE MECHANISM, RESOLVED: layers point in different directions and the mean cancels them (Phase 73)

§14C(b) showed the depth profile carries the head's gain (+7.3pp) but left two things open: whether
the head merely picks a better *layer*, and why explicit sink indicators bought nothing. Both are
now answered from attention maps already on disk. No GPU.

### It is not layer selection — it is layer *contrast*

| proposer | top-1 coverage |
|---|---|
| deployed block-16-26 mean | 39.3% |
| **best single layer (L17)** | **41.9%** |
| **linear learned reweighting, OOF** | **45.5%** |
| full head, 65 features (§14C) | 52.9% |

The best single layer beats the deployed mean by only 2.6pp, so **the block mean is not merely the
wrong layers**. A learned *linear* reweighting — `Σ wᵢ·Lᵢ` replacing `mean(L16..L26)`, 28 numbers,
a one-line change in any VLM's read-out — recovers **46% of the head's gain (+6.3 of +13.6pp)**.
The remaining half needs the head's non-linearity and neighbourhood features.

### Why the mean destroys it: the weights have BOTH SIGNS

| | layers | sink enrichment | gt_pct |
|---|---|---|---|
| most **positive** weight | L17, L19, L24, L16 | 2.3–4.2× | 0.32–0.42 |
| most **negative** weight | L27, L15, L26, L11 | 3.2–7.5× | 0.43–**0.53** |

**L27's gt_pct is 0.529 — worse than the 0.500 chance level.** The final layer is *anti*-correlated
with the target, and the deployed read-out adds it in with weight +1 like every other layer. The
learned read-out **subtracts** it. A mean can only add; that is the whole defect in one line.

### This reconciles §14C(b)'s sink puzzle

| correlation across the 28 layers | ρ |
|---|---|
| learned weight vs **gt_pct** (how well the layer ranks the target) | **−0.686** |
| sink mass vs gt_pct (sink-contaminated layers are worse layers) | **+0.522** |
| learned weight vs sink mass | −0.306 |

The sink **does** matter — it degrades layer quality (ρ = +0.522) — but it acts *through* which
layers are trustworthy, not through which cells are. That is exactly why adding a per-cell
last-column indicator bought **+0.0pp** (§14C(b)): the layer weighting already absorbs it.
**§14C(b)'s negative was not a refutation of the sink, it was a statement about where the sink
enters.** Both results stand and now agree.

### Layer structure

Sink enrichment decays with depth (early **10.49×**, mid 6.10×, late 5.75×), consistent with §5
finding it present at L0. Individual layers are weak everywhere (gt_pct 0.456 / 0.413 / 0.409
against a 0.500 chance) — **no single layer is a good localiser**; the signal exists only in the
contrast between them. Adjacent layers correlate at 0.806 and early-vs-late at 0.654, so the
disagreement the head exploits is a minority of the variance.

### ✗ NEGATIVE, run alongside: cross-layer disagreement is not a reliability signal

The spread of per-layer argmax positions predicts whether the proposal covers at **AUROC 0.572**,
against `peak`'s **0.745** on the identical target. Disagreement is **substantially worse** than the
signal we already had. It is not a free coverage detector.

> ✅ **Discrepancy resolved (our error, not §6's).** An earlier draft reported `peak` at 0.576 and
> flagged §6's 0.788 as suspect. §6 is correct: it predicts whether the argmax proposal covers **at
> all (>0)**, and that reproduces at exactly **0.788**. Our 0.576 came from computing `peak` as the
> max of the **unmasked** map, so it was dominated by the serialization-sink border cells; every
> other use of `peak` in this project ring-masks first. Ring-masked values: argmax-covers-at-all
> **0.788**, argmax-covers-≥0.5 **0.839**, head-covers-≥0.5 **0.745**.


## §14E  ★★ TRANSFER: the re-ranker generalises zero-shot; the allocator still loses at 4K (Phase 72)

The head trained on V\*Bench (191 items, GT-box coverage labels), applied to **HR-Bench 4k** with
**nothing refitted** — not W, not the layer block, not the ring mask, not B₀. HR-Bench ships no
boxes, so the head **cannot** have been fitted here even in principle. n = 800 rows / 200 instances,
balanced 400 `cross` / 400 `single`, all arms within 4.0% of their token target.

### Two separate claims, and they point opposite ways

| | per-row | CircularEval |
|---|---|---|
| uniform@300 | 52.8% | 37.5% |
| **uniform@600 (compute-matched bar)** | **59.9%** | **46.0%** |
| argmax@0.15 (incumbent) | 42.6% | 26.5% |
| **head@0.15** | **47.5%** | **33.0%** |
| rand@0.15 (control) | 38.9% | 20.0% |

| contrast | Δ | CI | |
|---|---|---|---|
| **head − argmax (P1)** | **+4.9pp** | **[+1.8,+8.1]** | ✔ **the component transfers** |
| head − rand | **+8.6pp** | [+4.6,+12.8] | ✔ not random placement |
| head − uniform@600 | −12.4pp | [−16.2,−8.4] | ✗ **the allocator still loses** |

**① The re-ranking transfers.** Across benchmark, resolution (~1500px → 4032px), and question
distribution, with zero refitting: **+4.9pp** per-row and **+6.5pp** on HR-Bench's own strict
CircularEval metric (26.5% → 33.0%). This is much stronger evidence for §14C than V\*Bench alone.

**② Allocation still loses to the budget axis at 4K.** −12.4pp against uniform@600. §9B is
unchanged: at 4032px, *spending* the budget beats *allocating* it, and a better proposal narrows the
gap without closing it.

### The regime boundary replicates exactly

| category | head | argmax | bar | head − argmax | head − bar |
|---|---|---|---|---|---|
| **`single`** (n=400) | 58.2% | 49.8% | 64.5% | **+8.5pp [+4.0,+13.0]** ✔ | −6.2pp ✗ |
| `cross` (n=400) | 36.8% | 35.5% | 55.2% | +1.3pp [−3.2,+5.8] ✗ | −18.5pp ✗ |

P1 ✔, P2 ✗, **P3 ✔** — the head helps only where one window can cover the evidence, which is the
§3/§7 coverage prediction, replicated on a benchmark with a different category vocabulary.

### The internal control is exact, again

On the **160 rows where head and argmax chose the same cell**: **+0.0pp, CI [+0.0,+0.0]**. On the
640 where they differ: **+6.1pp [+2.2,+10.0]**.

### ⚠ Two retracted readings of our own interim data

1. **"It does not transfer" — WRONG.** Called at n=244 from a prefix that was 65% `cross` (HR-Bench
   is category-ordered). `cross` is exactly where the head does nothing, so the prefix showed
   nothing. The category-ordering risk was flagged and the direction was asserted anyway.
2. **Two mechanisms proposed for a failure that was not real.** "The map is flat at 4K" — refuted by
   peak concentration (HR-Bench 0.0228 vs V\*Bench 0.0269 median, only 1.18×, both ~7× a uniform
   map). "Below the cliff there is nothing to re-rank" — refuted by §14D(b), where the head's gain is
   *largest* below the cliff. Neither was checked before being asserted.


## §14G  ★★★ "WHERE TO LOOK" DOES NOT EXIST BEFORE THE LM RUNS (Phase 76)

Gate for single-pass foveation, and a strong negative that stands on its own. The vision tower runs
before the language model, so if *its* attention localised the target, resolution could be allocated
at encode time for **2 vision encodes + 1 LM forward** instead of crop-and-re-encode's
**2 vision + 2 LM** — removing the second-pass cost that has sunk every allocation method in §2.

Qwen3-VL-2B's vision tower: 24 blocks, 16 heads, patch 16, merge 2. Per-patch salience = attention
received, averaged over heads and queries (what a class token approximates), pooled to the merged
grid the LM sees. n = 191, V\*Bench, same ring mask and same coverage definition as §14C.

| proposer | top-1 evidence coverage |
|---|---|
| **random ring-masked cell (chance, 200 draws/item)** | **2.3%** |
| best single **vision** layer (L17) | **1.6%** |
| learned signed combination over 24 vision layers, OOF | 1.0% |
| mean of all 24 vision layers | 0.5% |
| — | |
| deployed **LM** read-out (argmax) | **39.3%** |
| learned **LM** head (§14C) | **52.9%** |

**Every vision-tower arm is at or BELOW chance.** Not weak — absent.

> **The "where to look" signal is not a property of the image. It is CONSTRUCTED by the language
> model from the question.** The vision encoder cannot know what to prioritise because it never sees
> the question; there is no question-independent salience that suffices when the target is small and
> query-specific.

### Consequences

1. **Single-pass foveation steered by vision salience is dead at step one.** Any foveated encoder
   needs a question-conditioned signal, which requires an LM forward. A *cheap glance* (low-res LM
   pass) remains viable and is much cheaper than a full second pass — but it is not single-pass, and
   the honest cost model must say so.
2. **§14F's depth structure is LM-specific.** The vision tower's learned weights split 12 positive /
   12 negative with the last layer at −0.382 — the same *shape* — yet buy nothing, because there is
   no signal to combine. Depth contrast amplifies a signal; it does not create one.
3. **⚠ It qualifies a result in ram's paper.** His Vision-CLS selector *is* vision-tower attention,
   and he reports box-inside 0.808 on CUB. CUB birds are large and centred, so any salience finds
   them. On V\*Bench, where targets are small and the question selects among several plausible
   objects, the same selector is **at chance**. Both are correct; the scope differs, and a merged
   paper must state where vision-side selectors stop working.


## §14H  ✗ NEGATIVE as a method, ★★ POSITIVE as causal evidence: contrastive decoding (Phase 77)

A decoding-time fix: run normally, run again with the evidence region masked out of attention, push
the logits away from the masked version. Two LM passes at B₀ = 300 each, **no crop, no re-encoding,
no added pixels**. n = 190, V\*Bench. α swept.

### ★ The causal result, which is new and matters

| region masked | mean L1 logit shift | accuracy of the masked pass |
|---|---|---|
| **evidence (head)** | **3.348** | **45.8%** (−10.5pp vs baseline) |
| evidence (oracle GT box) | 2.906 | |
| evidence (argmax) | 3.213 | |
| **random region, same size** | **0.874** | |

**Masking the evidence region shifts the logits 3.32× more than masking a random region, and costs
10.5pp of accuracy.** This closes a gap §10A left open: §10A concluded image-token attention is
"not a scarce resource" because biasing away from *random / interior / sink* cells was inert — but it
never had a working localiser. With one, the evidence region is demonstrably load-bearing.

> **The model does read the evidence region. What is there is simply too impoverished to answer
> with.** That unifies §10 (interventions fail), §13B (the cliff — sub-token evidence is barely
> encoded) and §14D(b) (the head pays most below the cliff): the region is used, it just does not
> contain enough. Cropping helps because it puts *more* there, not because it redirects attention.

### ✗ As a method it fails

| arm (best α) | acc | vs baseline | CI |
|---|---|---|---|
| baseline | 56.3% | — | |
| cd_oracle (ceiling, GT region) | 58.9% | +2.6pp | [−0.5,+5.8] |
| cd_head (the method) | 58.4% | +2.1pp | [−1.1,+5.3] |
| cd_argmax (incumbent) | 57.9% | +1.6pp | [−1.6,+5.3] |
| cd_rand (control) | 56.3% | +0.0pp | [−1.6,+1.6] |
| **uniform@600 (compute-matched bar)** | **63.9%** | | |

**−5.5pp against its own bar.** It costs two LM passes and buys +2.1pp; spending those same two
passes on a bigger uniform image buys +7.4pp. Every CI against baseline includes zero, and even the
**ceiling** — contrast on the ground-truth region — reaches only +2.6pp [−0.5,+5.8].

The falsification test is directionally right but underpowered: the gain is larger where the head's
window covers (+3.0pp, n=101) than where it misses (+1.1pp, n=89), so the effect is evidence-linked
rather than generic logit sharpening — it is just far too small to matter.

### Why this was worth running anyway

It is the **seventh** internal intervention to fail, and the first to fail *with a working
localiser*. The previous six could always be dismissed as "you were steering to the wrong place."
This one steered to the ground-truth place, proved the place is causally live (3.32×, −10.5pp), and
**still** could not convert it into answers. That is a much stronger version of §10's claim:

> Allocation must add pixels. Not because the model looks in the wrong place — it looks in the right
> place and the right place is causally load-bearing — but because **what is encoded there is not
> enough, and no decoding-time operation can add information that was never encoded.**


## §14I  ★★★ WHY DCR WORKS: it restores answer formation at L21 (Phase 79)

The causal mechanism, read through a correctly-implemented logit lens. n = 190, V\*Bench, four arms
at identical realized budget — only the pixels differ. Intermediate layers via
`lm_head(final_norm(h_i))`, **final layer from `model.logits`**; the naive path is computed
alongside and its disagreement asserted, so §12D's double-normalisation bug cannot return silently.

### The arms are identical until L21, then split

head@0.15 − uniform@300, per layer:

| L14–L20 | **L21** | L22 | L23 | L24 | L25 | L26 | L27 |
|---|---|---|---|---|---|---|---|
| ≈ 0.0pp | **+15.3** | +16.8 | +17.4 | +12.1 | +17.9 | +14.7 | +12.1 |

**Zero separation for twenty layers, then a step.** This is not a model that reasons better with a
crop — it is a model that receives information a specific depth converts.

### Where each arm takes its biggest jump

| arm | final | biggest single-layer jump | at |
|---|---|---|---|
| uniform@300 | 56.3% | +23.2pp | **L2** |
| argmax@0.15 | 60.0% | +23.7pp | **L2** |
| **head@0.15** | **68.4%** | **+28.4pp** | **L21** |
| oracle@0.15 | 90.0% | +44.2pp | **L21** |

The two arms that fail jump **early** (L2 — the answer prior forming, not evidence) and are flat
thereafter: uniform's max over all 28 layers is 56.3%, exactly its final. The two arms that work
jump **late, at L21**, and the size of that step scales with how much evidence was supplied
(+28.4 head, +44.2 oracle).

### ★ The control that makes this causal, not correlational

Split the **head arm by whether its own window covers** — same intervention, same arm, differing
only in whether the crop actually delivered the evidence:

| stratum | head | uniform | oracle | head − uniform |
|---|---|---|---|---|
| **window COVERS** (n≈101) | **90.1%** | 60.4% | 98.0% | **+29.7pp [+19.8,+39.6]** |
| **window MISSES** (n≈89) | 43.8% | 51.7% | 80.9% | **−7.9pp [−18.0,+2.2]** |

**A 37.6pp swing between strata.** Where the window covers, DCR nearly reaches the oracle
(90.1 vs 98.0). Where it misses, it *hurts*. The late step appears only when the evidence is
actually there.

> **DCR does not make the model reason better. It moves items across a threshold — out of the regime
> where the answer exists at no layer, into the regime where L21 has something to convert.** The
> +12.0pp of §14D is 101 items gaining ~30pp minus 89 items losing ~8pp, and nothing else.

### Ever-argmax, read against its reference

uniform 86.8% · argmax 93.2% · head 93.2% · oracle 97.4%. Uniform's 86.8% is chance inflation from
reading a 4-way question at 28 depths (§13 established this); the ordering, not the level, is the
signal.

### What this closes

It unifies the whole causal chain. §13B: below the cliff nothing is encoded. §14H: the evidence
region is causally live (3.32×, −10.5pp) but too impoverished to answer with. §10/§14H: seven
internal interventions fail because **no operation on the residual stream can supply information the
encoder never wrote**. §14I: supplying it restores a step that was always there, waiting at L21.


## §14K  ⚠ REPLICATION: the averaging defect generalises; the *contrast mechanism* does not (Phase 74)

Qwen2-VL-7B, identical pipeline, n=191, block rescaled to the same fraction of the stack (L15–26 of
28). This is the test that decides whether §14F is a claim about VLM read-outs or about one
checkpoint. **It is partly the latter, and §2 of PAPER_FLOW must be split accordingly.**

| | Qwen3-VL-2B | **Qwen2-VL-7B** |
|---|---|---|
| deployed block mean | 39.3% | **35.1%** |
| best single layer, **out-of-fold** | **36.1%** (folds disagree) | **43.5%** (all 5 folds → L21) |
| best single layer, in-sample | 41.9% | 43.5% |
| learned linear signed combination, OOF | **45.5%** | 43.5% |
| mean of ALL layers | 36.6% | **21.5%** |
| final-layer gt_pct (0.500 = chance) | **0.529** | 0.416 |

### ✅ What replicates — and more strongly

**The deployed block-mean read-out is suboptimal on both models**, by **+6.2pp** on Qwen3-VL and
**+8.4pp** on Qwen2-VL. Averaging *all* layers is catastrophic on both (36.6% / **21.5%**). The
general claim — *averaging across depth dilutes localisation* — holds on two architectures.

### ✗ What does NOT replicate

- **Signed contrast is not the universal mechanism.** On Qwen2-VL the learned combination reaches
  **exactly** the best single layer's 43.5% — contrast buys nothing there.
- **The anti-correlated final layer is Qwen3-specific** (0.529 vs 0.416). This was §14F's sharpest
  and most surprising prediction and it does not transfer.

### The remedy is architecture-dependent, and that is the honest finding

| model | what fixes the read-out | gain over block mean |
|---|---|---|
| Qwen2-VL-7B | **pick L21** — one integer, zero learning, stable in all 5 folds | **+8.4pp** |
| Qwen3-VL-2B | **learned signed combination** — no single layer is stable OOF | **+6.2pp** |

> **Claim split.** *General (2 architectures):* the block-mean attention read-out is a measurable
> defect worth 6–8pp of localisation. *Specific (Qwen3-VL-2B):* the loss is signed contrast, and the
> final layer is anti-correlated. PAPER_FLOW §2 currently asserts the second as if it were the
> first.

### ⚠ A selection-bias trap, caught here

The in-sample "best single layer" is **not** a usable number: on Qwen3-VL it reads 41.9% and
collapses to **36.1%** out-of-fold — *below the block mean it was supposed to beat* — because it is
the best of 28 candidates on 191 items. On Qwen2-VL it survives (all folds pick L21 independently),
which is what makes that model's result trustworthy. **Any "just use layer k" recommendation must be
fold-validated**; the first version of this analysis used in-sample values and would have reported
the Qwen3 remedy backwards.

### Open

LLaVA-OneVision extracted **0 items**: it uses `image_newline` separators so its image tokens do not
form a clean grid, and the extractor skips rather than guesses. Phase 41 already solves this by
matching the model's own `image_newline` parameter against the merged embeddings; porting that is
the path to a third architecture and a two-family claim.


## §14L  ★★★ THE FINDING LEAVES OUR PROBLEM: pruning at the wrong depth costs 22pp (Phase 75)

The first result that lives entirely outside this paper's own task. Different problem (visual-token
pruning, not crop placement), different metric, an established published baseline, and a large active
literature. V\*Bench, n=191, Qwen3-VL-2B. Every arm prunes the **same number of tokens**, so cost is
matched by construction; pruning is a large negative attention bias on the dropped columns from
layer K onward, which is functionally what FastV does.

| keep | random | **layer-2 (FastV default)** | block-mean L16–26 | learned signed |
|---|---|---|---|---|
| **10%** | 38.2% | **34.6%** | **56.0%** | **58.6%** |
| 25% | 41.4% | 40.3% | 58.1% | 57.1% |
| 50% | 47.1% | 52.9% | 55.5% | 56.5% |

**No pruning: 56.5%.**

### ★ Two headline numbers

**1. 90% of visual tokens are deletable at zero cost — if you rank by late layers.**
block-mean **−0.5pp [−5.8,+4.7]**, learned **+2.1pp [−2.6,+6.8]** against no pruning at all.

**2. Ranking at layer 2 costs 22 points and is worse than random.**
layer-2 vs no-pruning: **−22.0pp [−29.8,−14.7]**. vs random selection: **−3.7pp** — *below chance
selection*. Margin of a late read-out over it: **+24.1pp [+16.2,+31.9]** at 10% keep,
**+16.8pp [+8.9,+24.6]** at 25%.

### ✗ And the signed combination contributes nothing here

linear vs block-mean: **+2.6 / −1.0 / +1.0pp, all CIs spanning zero.** A plain average of layers
16–26 matches the learned head. **The claim is about WHICH LAYERS are read, not about learning
weights over them** — and the recommendation is a one-line change requiring no training.

> ⚠ This is the **second independent failure** of signed contrast outside Qwen3-VL crop placement
> (§14K was the first). Under the replication standard it is finished as a general claim.

### The mechanism, and why it unifies three sections

§14G: the vision tower carries **no** localisation signal (at or below the 2.3% chance rate) because
it never sees the question. §14I: the answer forms **abruptly at L21**. §14L: pruning by layer-2
attention is worse than random.

> **Question-conditioned localisation is constructed late in the language model, and every decision
> the field makes earlier than that is made blind.** FastV prunes at layer 2. The vision encoder has
> nothing. Our own crop proposer averaged a fixed block without checking it was the right one. Three
> independent tasks, one cause.

### Scope and honesty

- **Reimplementation, not released code.** `layerK` ranks by attention at FastV's default K=2 on a
  common backbone; the original method is not run from its own checkpoint.
- **⏳ ONE MODEL.** Under the standard adopted 2026-09-16 this is **provisional** until it replicates
  on Qwen2-VL. Given §14K's finding that Qwen2-VL's early layers sit at gt_pct **0.620** — far worse
  than Qwen3's 0.456 — the prediction is that the effect is *larger* there, which is a sharp,
  falsifiable test.
- At keep=50% the margin over layer-2 is **+3.7pp n.s.** — the effect is specific to aggressive
  pruning, which is where the literature operates.


## §14M  ✗ NEGATIVE: rank-only multi-crop fails, and it is not a prompting artifact (Phases 81, 81b)

The method the allocation ledger pointed to. §14A/B/§78/§10B showed every attempt to **regress a
continuous geometric quantity** from internals fails, while **ranking cells** works on two
architectures. So: rank only. Take the head's top-4 separated cells, split B₀ evenly, show all four
in one pass. No size prediction, no budget prediction.

**Priced offline first** — union coverage of the top-4 *and* above the §13B cliff at 75 tok each is
**63.4%** against single-crop's 52.9%, i.e. **+10.5pp of answerable items at identical total
budget**, turning over at k=5. At §6D's exchange rate that predicted **+5.5pp**.

| arm | acc | tokens |
|---|---|---|
| uniform@300 | 56.5% | 294 |
| uniform@600 (bar) | 63.9% | 600 |
| **dcr_single@300** | **68.6%** | 294 |
| dcr_multi_k4 | 65.4% | 280 |
| argmax_multi_k4 | 66.5% | 280 |
| rand_multi_k4 | 37.7% | 280 |

**Measured −3.1pp [−9.4,+3.1] against a predicted +5.5pp.** The prediction was not imprecise, it had
the wrong sign — so the model behind it is wrong.

**Why the coverage model failed.** §6D's exchange rate was measured on **single** crops, where
coverage decides whether the evidence is visible *at all*. With four crops, three are distractors
and the model must both find the evidence **and** ignore three irrelevant views. **Nothing in the
coverage account charges for distractor cost.** Ranking still works inside the format
(dcr_multi − rand_multi = **+27.7pp [+17.8,+37.2]**); the format costs more than the coverage buys.

### The confound was checked, not assumed away

This contradicted phase 58, which measured multi-crop at **+7.9pp**. The clearest difference was the
text between images — phase 58 used *"Here is another zoomed-in crop from the same image"*, phase 81
used a bare newline, handing the model four pictures with no indication of what they were. Phase 81b
re-ran with the descriptive connector, paired on the same items:

| arm | newline | connector | Δ |
|---|---|---|---|
| dcr_single | 75.0% | 75.0% | +0.0 |
| dcr_multi_k4 | 75.0% | 75.0% | +0.0 |
| argmax_multi_k4 | 62.5% | 62.5% | +0.0 |

n=24 paired. The connector **does** change the model's output — **0/25 probability vectors are
byte-identical** — but it shifts confidences without flipping decisions. **The deficit is not a
prompting artifact.** (Run stopped at n=27 to free a shared GPU for the §14L replication, which
matters more; recorded as partial.)

> **Consequence: the withdrawn claim stays withdrawn.** DCR does not beat spending the same budget
> uniformly (+1.6pp [−6.3,+9.4] for multi, +4.7pp [−3.7,+13.1] for single), and the one principled
> idea for fixing that has been tested and failed.


## §14L(b)  ★★★ REPLICATED: pruning at the wrong depth costs 11–21pp on two architectures (Phase 83)

§14L was the paper's strongest claim and single-model. Qwen2-VL-7B, identical pipeline, block
rescaled to the same fraction of the stack (L15–26 of 28), n=191.

| keep | random | **layer-2 (FastV default)** | **block-mean (late)** | no pruning |
|---|---|---|---|---|
| **10%** | 44.5% | **39.8%** | **51.3%** | **52.9%** |
| 25% | 47.1% | 40.3% | 50.8% | |
| 50% | 45.5% | 43.5% | 50.3% | |

### ✅ The claim survives at two models

| | Qwen3-VL-2B | **Qwen2-VL-7B** |
|---|---|---|
| late read-out − layer-2, 10% keep | **+21.4pp** | **+11.5pp [+4.2,+18.8]** ✔ |
| …25% keep | +16.8pp | **+10.5pp [+3.1,+17.8]** ✔ |
| …50% keep | +3.7pp n.s. | **+6.8pp [+0.5,+13.1]** ✔ |
| **layer-2 vs random selection** | **−3.7pp** | **−4.7pp** |
| late read-out vs no pruning (10%) | −0.5pp | **−1.6pp [−6.3,+3.1]** |

**Ranking visual tokens by layer-2 attention is worse than ranking them at random, on both
architectures.** And on both, a late read-out discards 90% of visual tokens at a cost
indistinguishable from zero.

The effect is **about half the size** on Qwen2-VL — the same direction-preserved,
magnitude-halved pattern §80 found for DCR.

### ⚠ The mechanism prediction FAILED, and is reported separately from the verdict

Pre-registered: because Qwen2-VL's early layers rank the target *worse* (gt_pct **0.620** vs
Qwen3's 0.456, §14K), the layer-2 penalty should be **larger**. It is **smaller** —
**−13.1pp** against Qwen3's **−21.9pp**.

> **Layer ranking quality does not predict pruning damage.** "The question-conditioned signal does
> not exist yet at layer 2" survives as a *description* of where the signal is, but it is **not a
> sufficient account of how much pruning there costs**. §7 of PAPER_FLOW must state the claim
> (replicated) without the causal story (unsupported).

Candidate explanations, none tested: model scale (7B may carry more redundancy across visual
tokens), depth-normalised position of layer 2, or the interaction between pruning depth K and where
the signal forms. **Not asserted** — two mechanisms were already proposed and refuted today, and a
third guess is not worth more than the honest gap.

### ⚠ The `linear` arm is NOT the replication

It loads Qwen3-VL's learned weights and applies them to Qwen2-VL attention — a cross-model weight
transfer (50.3% at 10% keep, *below* this model's own block-mean 51.3%). The replication contrast
is block-mean vs layer-2, both computed from the model's own attention. This distinction was written
into the analyzer before the run so the stronger-looking number could not be quoted by mistake.


## §14N  ★★ FOUR ARCHITECTURES, TWO FAMILIES: the read-out defect holds on 3 of 4 (Phase 82)

The strongest multi-architecture evidence in the project. Identical pipeline, block rescaled to the
same fraction of each stack, learned read-out fitted **out-of-fold** and grouped by item, n=191 each.

| model | deployed block mean | learned read-out | Δ | CI | chance |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 39.3% | 45.5% | **+6.3pp** | [+2.6,+10.5] ✔ | 2.2% |
| Qwen2-VL-7B | 35.1% | 43.5% | **+8.4pp** | [+3.1,+13.6] ✔ | 2.2% |
| **LLaVA-OneVision-7B** | 25.1% | 26.2% | **+1.0pp** | **[−2.6,+5.2] ✗** | 2.1% |
| LLaVA-NeXT-7B | 12.6% | 18.8% | **+6.3pp** | [+2.1,+10.5] ✔ | 2.2% |

**3 of 4, spanning two model families and two tokenization schemes** (implicit raster boundaries in
Qwen; explicit `image_newline` separators in LLaVA). Averaging *all* layers is worse than the block
on all four (36.6 / 21.5 / 20.4 / 9.9%).

### The exception is reported as an exception

**LLaVA-OneVision is a null** (+1.0pp, CI spans zero). It is also the only model where the best
single layer *equals* the block mean (25.1% = 25.1%) — its layers apparently agree with one another,
so there is no disagreement for a combination to exploit. Internally consistent, but **one data
point: not built into a story.**

### Absolute localisation varies enormously — a finding in its own right

39.3% (Qwen3-VL) → 35.1% → 25.1% → **12.6%** (LLaVA-NeXT), against a ~2.2% chance rate. **All four
localise far above chance** (unlike the vision tower, §14G, which sits *at* chance) but the LLaVA
models are 2–3× worse in absolute terms. Anyone building crop-placement methods on LLaVA starts from
a much weaker proposal signal than the Qwen numbers in this paper suggest.

### ⚠ Methodological note carried from §14K

The "best single layer" column is **in-sample** (best of N on 191 items) and is reported for
description only. On Qwen3-VL the in-sample value reads 41.9% and collapses to **36.1%**
out-of-fold — *below* the block mean it appeared to beat. **Only the learned read-out column is
fold-validated**, and only it is claimed.

### The final-layer claim stays rejected

Final-layer gt_pct: **0.529** (Q3) · 0.416 (Q2) · 0.451 (OV) · 0.488 (NX). Only Qwen3-VL is above
the 0.500 chance level. **1 of 4** — §14K's rejection is confirmed at four models, not two.


## §14I(b)  ★★★ REPLICATED: cropping restores answer formation, on two architectures (Phase 84)

§14I was Part III's centrepiece and single-model. Qwen2-VL-7B, identical pipeline, n=191. Lens
implemented as in §12D (final layer from `model.logits`, both paths asserted to disagree).

**Pre-registered:** the specific layer index was **not** expected to transfer. What had to replicate
was (a) the *shape* — near-zero separation across most of the stack, then a step that persists — and
(b) the *causal control*, that the step appears only where the window covers.

### ✅ The causal control replicates

| stratum | head | uniform | oracle | Δ (head − uniform) |
|---|---|---|---|---|
| **window covers** (n=84) | **86.9%** | 63.1% | 92.9% | **+23.8pp [+13.1,+34.5]** ✔ |
| **window misses** (n=107) | 41.1% | 41.1% | 89.7% | **+0.0pp [−11.2,+11.2]** |

**Swing between strata: 23.8pp** (Qwen3-VL: 37.6pp). Where the crop delivers, the gain nearly reaches
oracle; where it misses, it is *exactly* zero. On Qwen3-VL the miss stratum was mildly negative
(−7.9pp); here it is flat. Direction preserved, magnitude smaller — the same pattern as §80 and
§14L(b).

### ✅ The shape replicates, one layer later

head − uniform separation: **≈0 through L21**, then **+8 (L22), +16 (L23), +13, +12, +12, +10**. The
step is at L22–23 rather than L21 — within the tolerance fixed before the run.

### ✅ And the sharpest sub-claim holds identically

**uniform's maximum over all 28 layers equals its final answer** — 50.8% = 50.8% here, 56.3% = 56.3%
on Qwen3-VL. On neither model is there a depth at which the answer was available and lost. This is
what licenses "the information is absent, not mis-routed" as a measurement rather than an inference.

> **Ledger: 8 survived / 5 rejected / 3 provisional.** Part III no longer rests on a single model.

⚠ The "biggest single-layer jump" statistic is noisy here (head's largest single step lands at L4,
the answer prior) and should not be quoted; the **separation profile** is the stable measure and is
what the figure shows.


## §14O  ★★★ THE TWO MECHANISMS ARE DISSOCIATED — the review was right (Phase 88)

External review objected that the paper conflates **(a)** where localisation signal lives in a
per-layer attention map with **(b)** when the answer becomes decodable from the residual stream,
calling them "conceptually distinct mechanisms (attention quality vs. computation/answer-formation
timing)". Both quantities exist per layer on the same items, so this is testable rather than
arguable. **It is correct.**

| quantity | peak layer | value |
|---|---|---|
| attention localisation, top-1 coverage | **L17** | 41.9% |
| attention localisation, gt_pct | **L19** | 0.324 |
| **answer formation** (largest oracle-arm jump) | **L21** | **+44.0pp** |

**Separation: 4 layers.** Rank correlation across layers: attention-top1 vs decodability
**ρ = +0.297**; gt_pct vs decodability **ρ = +0.100**. These are weakly related, not the same signal.

### The decisive number

| | attention top-1 |
|---|---|
| the 6 layers **before** the answer forms | **30.1%** |
| from the answer layer **onward** | **24.8%** |

**Attention quality peaks, declines, and only then does the answer form.** The two are *sequential*,
not simultaneous: the model localises at L16–19 and answers at L21.

### Why this is an improvement, not a concession

1. **Two clean claims replace one muddy one.** *"Read attention late"* is a claim about **maps** —
   it is what the pruning result (§14L) and the read-out result (§14N) rest on. *"The answer forms
   abruptly at a single late layer"* is a claim about the **residual stream** — §14I. They are
   related only in that both live late in the stack.
2. **The pruning finding no longer depends on the logit lens at all.** Given the lens cost us the
   §12D double-normalisation retraction, a headline result that is independent of that machinery is
   strictly more robust.
3. **It sharpens the mechanism.** "Localise, then answer" is a specific sequential claim with an
   observable signature — attention quality *falling* before the answer appears — that the
   single-mechanism story does not predict and would have obscured.

> **Reframe required:** the paper's one "wrong depth" narrative should become two: a claim about
> **where the field reads attention maps** (pruning, crop placement — the practical contribution) and
> a claim about **when the answer is computed** (the interpretability contribution). §14L does not
> cite §14I for support and must stop appearing to.


## §14P  ★★★ THE PRUNING COLLAPSE HOLDS ON GENERAL VQA — and is far larger there (Phase 93)

External review objected that V\*Bench is the benchmark most likely to flatter this result: a
small-object *search* task where the answer hinges on one tiny region, so question-conditioned
late-layer attention is exactly what should matter. Tested on two benchmarks the pruning literature
actually uses, both loaded from local cache. **The objection is refuted, and the effect is larger on
general VQA.**

| | none | random | **layer-2** | late |
|---|---|---|---|---|
| **POPE** (n=200, yes/no object presence) | 90.5% | 82.5% | **67.5%** | 84.0% |
| **MMBench** (n=200, 4-way general) | 85.5% | 78.5% | **53.0%** | 80.0% |

| layer-2 vs random selection | |
|---|---|
| V\*Bench | −1.6pp |
| **POPE** | **−15.0pp [−22.0,−7.5]** |
| **MMBench** | **−25.5pp [−33.0,−18.0]** |

vs no pruning: **−23.0pp** and **−32.5pp**. Late read-out beats layer-2 by **+16.5pp [+8.5,+24.5]**
and **+27.0pp [+20.5,+33.5]**. Three benchmarks, two of them general VQA.

### ⚠ RETRACTED: "delete 90% of visual tokens at zero cost" does NOT generalise

On V\*Bench a late read-out cost −1.0pp (free). On POPE it costs **−6.5pp [−10.5,−2.5]** and on
MMBench **−5.5pp [−10.0,−1.0]**, both significant. V\*Bench answers hinge on one small region so 90%
of tokens genuinely are irrelevant; general VQA uses more of the image. **The free-lunch claim was
V\*Bench-specific and is withdrawn.** What generalises is that a late read-out is *far better than
the alternative*, not that it is free.

⚠ MMBench was rejected earlier in this project for **allocation** work (512px images leave no
allocation headroom). That objection does not apply to pruning, which adds no pixels. Reusing a
previously-rejected benchmark requires the justification stated, not assumed.

---

## §14Q  ★★ EARLY ATTENTION IS NO BETTER THAN COIN-FLIPPING, EVEN FOR COARSE CULLING (Phase 94)

Reading late means paying for half the network before discarding anything. Hypothesis: early
attention is useless for choosing the best 10% but adequate for dropping the *worst 50%*. Two-stage:
cut to 50% at L2, re-rank survivors at L16, final 10%. n=191, all arms end at 10% keep.

| arm | acc | visual-token compute saved |
|---|---|---|
| no pruning | 56.5% | 0% |
| late only (L16) | 55.5% | 39% |
| **two-stage, attention-guided 50%** | **51.8%** | **64%** |
| **two-stage, RANDOM 50%** | **50.8%** | **64%** |
| two-stage, 25% early | 39.3% | 74% |
| layer-2 only (FastV) | 34.6% | 84% |

**The control decides it: attention-guided vs random first cut = +1.0pp [−4.7,+6.8], indistinguishable.**

> Early attention is not merely a poor *ranker* — it is no better than a coin-flip even for deciding
> which half of the tokens to discard. So the honest method drops the pretence: **cut blind early,
> rank late.** 64% of visual-token compute saved against pure-late's 39%, at −4.7pp [−9.9,+0.5], and
> **+17.3pp [+9.9,+25.1]** over FastV's single early cut. Cutting to 25% early collapses to 39.3%,
> so 50% is near the limit a blind cut tolerates.

⏳ V\*Bench only; needs POPE/MMBench.

---

## §14R  ★★★ THE MECHANISM: attention is QUESTION-BLIND until ~50% of depth (Phase 95)

Phase 87 measured a threshold but could not explain it. Hypothesis: early layers cannot rank tokens
by relevance because their attention does not yet depend on **what was asked**. Test with no labels
at all — run the same image with 4 *different* questions (borrowed from other items, not paraphrases)
and measure per-layer divergence between the resulting maps.

| layer | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| L0–L12 | 0.001–0.005 | 0.001–0.005 |
| L13 | — | 0.0130 |
| L14 | **0.0105** | **0.0442** |
| **L16** | **0.1374** | **0.1779** |
| L20 | 0.2144 | 0.3107 |

**Divergence is ~0.001 through the first half — asking about the scarf, the cart, or the shirt
produces effectively the same attention map — then jumps 13× at L14→L16.** Both models switch on at
**50–57% of depth**.

This is **exactly** the pruning threshold (§87: the +13.1pp step at L14→L16). Early-layer ranking
fails because it is ranking on something question-blind, which is why it performs like coin-flipping
(§14Q) and below random (§14P).

### Three stages, not one

**blind (L0–L13) → question-aware (L14–L16) → answered (L21).** §14O's dissociation now has a shape:
question-dependence and usable localisation switch on together; the answer forms five layers later.

### It is also a deployable tool

"Read at 57% of depth" is useless to someone with a different model — locating their threshold would
need ground-truth boxes. Question-divergence needs **none**: a few forward passes on one image.
**Prediction for Qwen2-VL, untested:** its pruning threshold should sit at **L13–L14**. Running the
depth sweep there confirms or kills the locator.

⚠ Our first detection rule (3× the early baseline) fired on noise at L5 — 0.0055 against a 0.0016
baseline, inside a flat region — and was briefly read as refuting the hypothesis. The real signal is
10–100×. Rule replaced with "first layer exceeding 10× baseline", which gives L16 and L14.



## §14S  ⚠ THE LOCATOR IS RIGHT ABOUT THE REGION, WRONG ABOUT THE LAYER (Phase 96)

Phase 95's question-divergence measurement is **label-free** — same image, four unrelated questions,
per-layer map divergence — and on Qwen2-VL it switched on at **L13–L14** (0.0046 → 0.0130 → 0.0442,
then 0.1779 at L16). That is a prediction about a quantity never measured on this model: pruning
damage should step there. Phase 96 sweeps read depth directly. n=191, budget 286–315 measured, all
arms prune the same count from the same layer.

| read depth | acc | vs L2 |
|---|---|---|
| no pruning | 52.9% | +13.6 [+5.8,+21.5] |
| L0 / L1 / **L2** / L4 / L6 | 38.7 / 39.3 / **39.3** / 40.3 / 40.8% | ~0 |
| L8 / L10 / L12 | 36.6 / 38.7 / 40.3% | ~0 |
| **L14** | **44.0%** | +4.7 **[−1.0,+10.5]** ✗ |
| **L16** | **50.3%** | **+11.0 [+3.7,+18.8]** ✔ |
| L20 / L24 / L26 | 49.2 / 50.3 / 48.7% | +9.9 / +11.0 / +9.4, all ✔ |
| late block (L15–26) | 50.8% | +11.5 [+4.2,+18.8] ✔ |
| **random selection** | **42.9%** | +3.7 [−3.1,+10.5] |

### ✅ The shape is confirmed
A flat floor for the first half of the stack, a single transition, then a plateau to the end.
L20−L16 is **−1.0pp [−6.3,+4.2]** — nothing is gained after L16. This is the depth-effect reading
(b) of phase 87, not the "layer 2 is an anomalous quirk" reading (a): **every** depth up to L12 is
equally bad, and L2 is not special.

### ⚠ The prediction is off by one to two layers — report it that way
Divergence switches on at **L13–L14**; damage recovers at **L14–L16**. L14 sits halfway between the
floor (~39%) and the plateau (~50%) and its gain over L2 does **not** clear zero. The decisive step is
**L16−L14 = +6.3pp [+0.0,+12.6]** and **L16−L12 = +9.9pp [+3.1,+17.3]**.

> **Claim as supported:** question-divergence locates the *transition region* without labels.
> **Not supported:** naming the single best read layer. Reading at the layer where divergence first
> rises still loses 6.3pp against reading two layers later.

### The random-selection comparison is consistent but underpowered
All eight early depths fall below random (mean **−3.7pp [−10.3,+2.7]**) and all four late depths sit
above it (mean **+6.7pp [−0.1,+13.7]**). The sign pattern is 12/12 in the predicted direction; no
individual contrast clears zero at n=191. **The "early is worse than random" claim rests on §14L(b)'s
paired design, not on this sweep** — recorded here so the weaker evidence is not double-counted.


## §14T  ★★★ THE METHOD CLEARS THE BAR ON SINGLE-OBJECT QUESTIONS, ON TWO MODELS (Phases 97, 101)

§14D's "0 of 2 against the equal-compute baseline" compared **Qwen3-VL at the held-out-validated
W=0.25** against **Qwen2-VL at W=0.15, a window that was never swept on it**. Phase 97 fixes that:
same items, same head, W ∈ {0.15,0.25,0.35}, W **transferred** from Qwen3-VL rather than selected.

### Pooled — still does not clear, on either model

| | bar (uniform@600) | head@0.25 | Δ |
|---|---|---|---|
| Qwen3-VL-2B | 63.9% | 71.7% | **+7.9pp [−0.5,+16.2]** ✗ |
| Qwen2-VL-7B | 58.1% | 64.9% | **+6.8pp [−1.0,+14.7]** ✗ |

The unfairness was real — Qwen2-VL's margin **more than doubles**, +3.1 → +6.8pp — and fixing it was
still not enough. **Two positive point estimates of similar size with lower bounds within a point of
zero is an underpowered contrast, not a refuted one**, and the ledger's wording is corrected
accordingly: NOT DEMONSTRATED, not REJECTED.

### ✅ Single-object questions — clears on both

| | n | bar | head@0.25 | Δ |
|---|---|---|---|---|
| **Qwen3-VL, `direct_attributes`** | 115 | 62.6% | 78.3% | **+15.7pp [+6.1,+25.2]** ✔ |
| **Qwen2-VL, `direct_attributes`** | 115 | 57.4% | 68.7% | **+11.3pp [+1.7,+20.9]** ✔ |
| Qwen3-VL, `relative_position` | 76 | 65.8% | 61.8% | −3.9pp [−18.4,+10.5] |
| Qwen2-VL, `relative_position` | 76 | 59.2% | 59.2% | +0.0pp [−13.2,+13.2] |

> **This is the first thing in the project to beat spending the same compute on a bigger image, on
> more than one architecture.** The scope limit is not a caveat bolted on afterwards — §6D/Ph 36
> predicted it before any of these runs: a relational question's evidence set is the **union** of the
> objects involved, **7.9× larger in area**, and one window cannot cover it.

⚠ **Declared honestly:** phase 71b's *formal* pre-registered decision rule was on the **pooled**
contrast, and pooled does not clear. The single-region stratum was a mechanism-derived expectation
named in §14D's own title before Qwen2-VL was ever run — it is not a slice found after the fact — but
it was not the registered primary, and both numbers belong in the paper.

### The window-width mechanism, replicated

| | W=0.15 | 0.25 | 0.35 | 0.50 | 0.70 |
|---|---|---|---|---|---|
| Qwen3-VL **oracle** placement | **90.1** | 88.0 | 83.8 | 79.6 | 65.4 |
| Qwen3-VL **head** placement | 68.6 | **71.7** | 69.1 | 65.4 | 63.4 |
| Qwen2-VL **oracle** placement | **91.1** | 87.4 | 80.6 | — | — |
| Qwen2-VL **head** placement | 61.3 | **64.9** | 64.4 | — | — |

**Oracle placement is monotone decreasing in W on both models; head placement is not.** Perfect aim
wants the tightest window; imperfect aim buys forgiveness with width. The optimum is interior *only*
for a proposer that misses, which is why W=0.25 and not W=0.15 — and it retroactively explains why
phase 39 had to retract an interior optimum measured on the weaker argmax proposer.

⚠ On Qwen2-VL the peak itself is weak: head@0.25 − head@0.15 is **+3.7pp [−2.1,+9.4]**, and 0.25 vs
0.35 is flat. Claim the oracle/head *contrast*, not a sharp peak location.

### ✗ And the sizer fails its own robustness check (Phase 99, 101)

A free per-item window sizer over the six pass-1 attention features looked like the missing point at
**+8.9pp [+0.5,+17.3]** on the 5-value grid. Restricted to the 3-value grid both models share, it
**inverts**: +6.8pp [−1.6,+14.7] against the bar and **−1.0pp** against the fixed constant. Found on
the model where it was discovered, before it reached the second one. A `top1_frac` routing gate
(+9.6pp [+2.0,+17.7]) is **not** substituted for it — `PREREG_DCR_SIZER.md` bars exactly that move.
The two-pass confidence rule that rescued the *old* proposer is also null here (−0.5pp [−4.7,+3.1]):
it was compensating for a bad proposer, and there is less to compensate for now.


## §14U  ✗ THE PRE-REGISTERED SIZER TEST FAILS, AS WRITTEN (Phases 97b, 101)

`PREREG_DCR_SIZER.md` froze one configuration before any Qwen2-VL data for it existed. Phase 97b
supplied the two wide windows phase 97 omitted, so the pre-registered 5-value grid could actually be
run rather than narrowed to what was on disk.

| Qwen2-VL-7B, n=191 | acc | vs uniform@600 | vs fixed W=0.25 |
|---|---|---|---|
| uniform@600 (bar) | 58.1% | — | — |
| fixed W=0.25 (incumbent) | 64.9% | +6.8pp [−1.0,+14.7] | — |
| **SIZER, OOF, 5-value grid (PRIMARY)** | **61.8%** | **+3.7pp [−3.1,+10.5]** ✗ | **−3.1pp** [−9.4,+3.1] |

**The decision rule was applied as written: the CI spans zero, so the claim stays rejected.** The
sizer is worse than the constant it was meant to improve, and it sends **43 of 191** items to W=0.7,
the worst window in the sweep. Combined with §14T's grid inversion on Qwen3-VL (+8.9pp → −1.0pp), the
sizer is **dead on two models and two grids**.

The `top1_frac` routing gate (+9.6pp [+2.0,+17.7] on Qwen3-VL) is **not** substituted. The
pre-registration bars that move and the ban is the reason the file exists.

> **What this leaves.** The method is the incumbent, unchanged: rank cells by the 28-layer attention
> profile, crop once at a **fixed W=0.25**. Single-object questions **+15.7pp [+6.1,+25.2]** and
> **+11.3pp [+1.7,+20.9]** over equal compute on two models; pooled **+7.9pp / +6.8pp**, neither
> clearing. Phase 78's +11.0pp sizer headroom is closed as unreachable from free pass-1 signals.

⚠ Note the sizer's category split on Qwen2-VL: `relative_position` falls to **−7.9pp**, worse than
the incumbent's +0.0pp. Choosing a window per item actively hurts on questions no single window can
cover, which is the coverage account applied to the sizer itself.


## §14V  ★★★ MAX, NOT MEAN: an LLM prefill paper's aggregation rule beats our read-out (Phase 105)

**CLAA** (McDanel, Li & Khaitan, arXiv 2602.16054, Feb 2026) diagnoses our depth problem on the LLM
side, independently and with a different oracle. They build an *Answer-Informed Oracle* — true token
importance measured as attention from the **generated answer** back to the prompt — score ranking
heuristics layer by layer against it, and find **"layer-wise ranking instability": rankings "degrade
sharply at specific layers, a failure mode invisible to end-to-end benchmarks"**, with early layers
(0–4) consistently worst. Their fix: *do not trust any single layer* — aggregate over a window of
consecutive layers, **by MAX rather than mean**, "to preserve tokens deemed important by any recent
layer while filtering layer-specific noise".

Our deployed read-out takes the **mean**. Applied to VLM localisation, on disk, W=0.25, n=191 each:

| aggregation rule | Qwen3-VL | Qwen2-VL |
|---|---|---|
| **mean over block (the incumbent)** | 46.6% | 39.3% |
| mean over all layers | 42.4% (−4.2) ✗ | 22.5% (−16.8) ✗ |
| median over block | 41.9% (−4.7) ✗ | 26.2% (−13.1) ✗ |
| **max over block** | **53.9% (+7.3 [+2.6,+12.6])** ✔ | **46.1% (+6.8 [+3.1,+11.0])** ✔ |
| max over all layers | 52.4% (+5.8) ✔ | 44.5% (+5.2) ✔ |
| **max over a 4-layer window (CLAA as specified, window OOF)** | **55.0% (+8.4 [+3.1,+13.6])** ✔ | **51.3% (+12.0 [+6.8,+17.3])** ✔ |

> **Replacing the mean with a max is a one-line, training-free change worth +8.4 / +12.0pp of
> evidence coverage on two architectures.** The deployed convention is not merely suboptimal in its
> *range* (§14F) — it is suboptimal in its *rule*, and the better rule was published for LLM prefill
> acceleration in a literature this field does not cite.

### ⚠ And it takes most of the learned head's margin

| contrast | Qwen3-VL | Qwen2-VL |
|---|---|---|
| learned head − deployed argmax | +16.8pp [+11.0,+23.0] ✔ | +15.2pp [+9.4,+21.5] ✔ |
| learned head − max over block | +9.4pp [+4.2,+15.2] ✔ | +8.4pp [+4.2,+13.1] ✔ |
| **learned head − max_win4 (CLAA)** | **+8.4pp [+3.1,+13.6]** ✔ | **+3.1pp [−1.0,+7.3]** ✗ |

**"The learned head beats a properly aggregated training-free baseline" is 1 of 2, and therefore
REJECTED** under the standing rule. The head survives against max-over-the-deployed-block on both
models, but against CLAA's OOF window-max its margin halves and loses significance on Qwen2-VL —
the same direction-preserved, magnitude-halved pattern §14K and §80 found twice before.

> **What this means for the method.** Our reported gains have been measured against a baseline that
> is weak for a reason nobody had named. A one-line rule recovers **50% (Qwen3) to 79% (Qwen2)** of
> what a 65-feature learned head buys, at zero training cost, with an independent justification from
> the LLM literature. The honest paper reports both, and may well *prefer the simpler proposer*.

### What has not been run yet
Every end-task number in this project uses the head's proposals. Whether max_win4 converts as well
as the head does is an open GPU run — and if it does, the method becomes: **crop at the argmax of a
max-aggregated attention map, W=0.25, no training at all.**


## §14W  ★★ THE MAX RULE PROPOSES BETTER BUT DOES NOT CONVERT (Phase 106)

§14V showed CLAA's max-over-a-4-layer-window beats our mean-over-a-block read-out by +8.4 / +12.0pp
of evidence coverage, training-free. This runs it through the answer. Qwen3-VL, n=191, W=0.25.

**Pipeline control passed exactly:** `head@0.25` was RE-RUN rather than copied — stored 71.7%,
re-run 71.7%, agreeing on **100.0%** of items. The join to the stored uniform arms is sound.

| arm | acc | vs uniform@600 |
|---|---|---|
| uniform@300 (pass 1) | 56.5% | — |
| **uniform@600 (the bar)** | **63.9%** | — |
| **maxwin4@0.25 (training-free)** | **68.6%** | **+4.7pp [−3.7,+13.1]** ✗ |
| **head@0.25 (learned)** | **71.7%** | +7.9pp [−0.5,+15.7] |

| single-object questions (n=115) | vs the bar |
|---|---|
| **learned head** | **+15.7pp [+6.1,+25.2]** ✔ |
| max_win4 | +9.6pp [+0.0,+19.1] ✗ |
| head − max_win4 | +6.1pp [+0.0,+13.0] |

> **The learned head earns its keep.** A one-line training-free rule recovers much of the read-out
> gap but converts 3.1pp worse end-task and does not clear the equal-compute bar on either the pooled
> set or the single-object stratum. Only the head clears.

The arithmetic is consistent rather than surprising: max_win4's proposal coverage is 55.0% against the
head's 63.4%, an 8.4pp gap, and at this project's measured conversion rate (~0.62pp of accuracy per pp
of coverage) that predicts 5.2pp. Observed 3.1pp.

### Why this strengthens the paper rather than weakening it

Our gains were previously measured against the **deployed block-mean argmax**, which §14V showed is
weak for a nameable reason. They now stand against a **principled, training-free, independently
motivated baseline** — and the margin over that baseline, while not decisive (+6.1pp [+0.0,+13.0] on
single-object), is in the right direction with the bar cleared only by the head.

⚠ Qwen2-VL is queued. On coverage the head − max_win4 contrast was **1 of 2** (§14V), so the end-task
replication is the one that decides whether the head can be claimed over the simple rule at all.

### Incidental
Both out-of-fold folds selected window **L16–19 or L17–20** — the max rule independently rediscovers
the band the deployed block was hand-set to, and the band phase 95's label-free locator points at.


## §14X  ✗ NEGATIVE: the attention depth profile does NOT detect encoding failure (Phase 107)

Pre-registered geometry-free — no GT box, no target extent, nothing from the label. Features: per
layer peak / entropy / top-1 share / top-5 share / peak-over-median, plus cross-layer statistics
(max-minus-mean gap, argmax instability across depth, consecutive-layer divergence and where it
first rises, early-vs-late map agreement). Out-of-fold, 5 folds.

**Targets.** `err` = pass-1 wrong. **`encfail` = pass-1 wrong AND the oracle crop right** — a
*certified* encoding failure, a label no other work can construct because it needs the crop control.

| AUROC, target `encfail` | Qwen3-VL | Qwen2-VL |
|---|---|---|
| max-softmax (baseline) | 0.588 | **0.651** |
| predictive entropy (baseline) | 0.625 | **0.679** |
| `peak` alone (§8A) | 0.506 | 0.484 |
| **depth profile, GBT** | 0.630 | **0.579** |
| depth + confidence | 0.652 | 0.580 |

**1 of 2, with the ordering reversed on the second model — rejected.** On Qwen3-VL the depth profile
edges the baselines; on Qwen2-VL it is worse than both. Recorded next to §14B: no free pass-1 signal
predicts the required *budget*, and no attention-depth signal predicts *encoding failure*.

### ★ Two things survive the negative

**1. The base rates are the paper's headline.** 37.7% (Qwen3) and 41.4% (Qwen2) of *all* V\*Bench
items are certified encoding failures, i.e. **86.7% / 84.0%** of every error is fixed by spending the
same 300 tokens on the evidence region (94.0% / 89.4% at W=0.15). **Nine in ten errors are encoding
failures, not reasoning failures, on two architectures.**

**2. `peak` is at chance for correctness (0.484–0.506) while predicting coverage at 0.788 (§8A).**
The free signal that says whether a crop will land says nothing about whether the model is wrong.
Another instance of the §14O dissociation: quantities about *where attention is* and quantities about
*whether the answer forms* are different quantities.

⚠ This does **not** test Orgad et al. (ICLR'25), whose central claim is that **token selection** is
what makes probing work — probe the *exact answer tokens*, not a pooled summary. We probed attention
statistics, which is not their method. Phase 109 tests theirs.


## §14W(b)  ✅ REPLICATED: only the learned head clears the bar, on both models (Phase 106, Qwen2-VL)

Pipeline control passed exactly a second time: `head@0.25` stored 64.9%, re-run 64.9%, agreeing on
**100.0%** of items.

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| uniform@600 (the bar) | 63.9% | 58.1% |
| max_win4 @0.25 (training-free) | 68.6% | 60.7% |
| **learned head @0.25** | **71.7%** | **64.9%** |
| max_win4 − bar (pooled) | +4.7 [−3.7,+13.1] ✗ | +2.6 [−5.8,+11.0] ✗ |
| head − bar (pooled) | +7.9 [−0.5,+15.7] ✗ | +6.8 [−1.0,+14.7] ✗ |
| **head − bar, single-object** | **+15.7 [+6.1,+25.2]** ✔ | **+11.3 [+1.7,+20.9]** ✔ |
| max_win4 − bar, single-object | +9.6 [+0.0,+19.1] ✗ | +6.1 [−3.5,+15.7] ✗ |
| head − max_win4, single-object | +6.1 [+0.0,+13.0] | +5.2 [+0.0,+10.4] |

> **Only the learned head clears the equal-compute bar, and it clears it on both architectures.**
> §14V's "head − max_win4 is 1 of 2" was a contrast on *coverage*; on the end task the head wins on
> both models (+6.1 / +5.2pp, both lower bounds at zero — consistent though marginal), and the
> training-free rule never clears the bar on either.

**The two results are not in tension and both belong in the paper.** As a *read-out*, max over a
window beats the deployed mean by +8.4 / +12.0pp of coverage, training-free — that is a real and
useful finding about how the field aggregates across depth. As a *proposer*, it converts worse
(−3.1 / −4.2pp) and does not earn its second forward pass. Coverage is not accuracy, and this is the
cleanest demonstration of that gap the project has.

Windows selected out-of-fold: L16–19/L17–20 on Qwen3-VL, **L18–21/L19–22 on Qwen2-VL** — both land on
the band the deployed block was hand-set to, and Qwen2-VL's includes **L21**, the layer §14K found
was its best single localiser and §14I(b) found is where its answer forms.


## §14Y  ⚠ CORRECTED — the head DOES transfer to LLaVA-NeXT at the deployed window (Phase 110, 110b)

Phase 82's stored per-layer attention made this a no-GPU test. Phase 70's machinery verbatim — same
features, same out-of-fold GroupKFold grouped by item, same negative subsampling, same ring mask —
pointed at a different family. Block rescaled to the same fraction of each stack. W=0.25.

| | deployed argmax | learned head | Δ |
|---|---|---|---|
| Qwen3-VL-2B | 46.6% | 63.4% | **+16.8 [+11.0,+23.0]** ✔ |
| Qwen2-VL-7B | 39.3% | 54.5% | **+15.2 [+9.4,+21.5]** ✔ |
| **LLaVA-NeXT-7B** (32 layers, 552 tokens) | 23.0% | 28.3% | **+5.2 [−1.6,+12.0]** ✗ |
| **LLaVA-OneVision-7B** (28 layers, 540 tokens) | 35.6% | 37.2% | **+1.6 [−4.2,+7.9]** ✗ |

**4 of 4 positive in direction; 2 of 4 significant, and both are the same vendor.** The magnitude
collapses by 3–10× across the family boundary.

### The failure is predicted by §14N, which is why it is reported rather than explained away

§14N found LLaVA-OneVision is the **one model where the best single layer equals the block mean**
(25.1% = 25.1%) — its layers agree with one another, so there is no disagreement for a re-ranker to
exploit. The head's whole mechanism is exploiting layer disagreement. And LLaVA-NeXT has the
disagreement but a far weaker base signal (23.0% against Qwen3-VL's 46.6%), leaving less to re-rank.
Both ceilings are intact (97.4% / 96.9% of items have some covering cell), so this is not a data
problem.

### The end task was deliberately NOT run

At this project's measured conversion rate (**~0.62pp of accuracy per pp of coverage**, §14D),
+5.2pp of coverage predicts ≈+3pp end-task — inside noise at n=191, against a bar the Qwen models
clear by 11–16pp. An hour of GPU for a null we can compute in advance, which would not change the
scoping decision.

### ⚠ CORRECTION (phase 110b): the above is a W=0.25 result, not a general one

The table above uses **W=0.25**. The deployed window is **W=0.15**, and at that window the head
clears on LLaVA-NeXT:

| W=0.15 | block-mean | max over block | **learned head** |
|---|---|---|---|
| Qwen3-VL-2B | 39.3% | +1.6 ✗ | **+14.1 [+7.9,+20.4]** ✔ |
| Qwen2-VL-7B | 35.1% | +5.2 ✔ | **+8.9 [+3.1,+14.7]** ✔ |
| **LLaVA-NeXT-7B** | 12.6% | +0.0 ✗ | **+7.3 [+2.1,+12.6]** ✔ |
| LLaVA-OneVision-7B | 25.1% | −3.7 ✗ *worse* | −1.0 ✗ |

**The head is 3 of 4 across two families**, matching §14N's independent +6.3pp on LLaVA-NeXT — three
measurements of the same quantity at +6.3, +7.3 and +5.2, the last being the W=0.25 arm whose CI
spans zero. The single failure is LLaVA-OneVision, and §14N already explains it: that is the one
model whose **best single layer equals its block mean**, so its layers agree with one another and
there is no disagreement for any re-ranker to exploit.

> **And the comparison that matters for the paper's framing: the learned head transfers across
> families (3 of 4) while the training-free max rule does not (1 of 4 at W=0.15, and −3.7pp on
> LLaVA-OneVision).** The hand-designed corrections that beat a simple read-out on Qwen are
> Qwen-specific; the learned read-out is not. Any move to replace the head with something more
> elegant must clear this bar, not just the Qwen bar.

⚠ My error, recorded: I ran LLaVA at W=0.25 only and wrote the scope conclusion from it. The window
interacts with the model, and a single-window scope claim was not supported.


## §14Z  ✗ NEGATIVE: the fixes that rescue a SIMPLE read-out add nothing to a LEARNED one (Phase 112)

Norm weighting (Kobayashi et al., EMNLP'20) and max aggregation (CLAA, 2602.16054) are each worth a
lot against the deployed **block-mean argmax**: +4.7pp and +7.3pp alone, **+10.5pp** together (§14V,
phase 104b). The head has never been given either — it reads raw per-layer attention and raw
within-layer ranks, and a gradient-boosted tree cannot construct a max across 28 columns, nor
`alpha·||v||` at all, since the value norms were never passed to it.

Qwen3-VL, n=191, W=0.25, identical out-of-fold folds:

| arm | features | coverage | vs incumbent |
|---|---|---|---|
| **raw65 (the deployed head)** | 65 | 62.3% | — |
| + max over block and over all layers | 68 | 62.3% | +0.0 [−1.6,+1.6] |
| + 28 norm-weighted layers and their ranks | 122 | 61.8% | −0.5 [−4.2,+2.6] |
| + both | 128 | 63.9% | +1.6 [−1.0,+4.2] |
| built entirely from norm-weighted attention | 65 | 63.4% | +1.0 [−1.6,+4.2] |

**Nothing clears.** Given the raw depth profile and ranks, the head already recovers whatever those
hand-designed corrections encode.

> **The useful statement is conditional.** *If you train nothing*: take a max across layers and weight
> by the value norm — **+10.5pp**, one line, two independent justifications from the LLM literature.
> *If you train a head on the raw depth profile*: both are redundant. This also explains §14W(b) —
> why the head beats max_win4 end-task on both models despite max_win4 being the better
> hand-designed read-out.

### ⚠ A noise floor, measured by accident, that applies retroactively

`raw65` scores **62.3%** here against the **63.4%** quoted throughout, because this run uses phase
104b's re-extracted attention (a separate forward pass under eager attention) rather than phase 30c's.
Same quantity, same model, same items, two extractions: **~1pp apart.**

**Our run-to-run noise floor on top-1 coverage is therefore about 1pp**, and the +1.6pp arm above sits
inside it. Any coverage difference of that size anywhere in this project is noise and must not be
read as signal.


## §15A  ⚠ REJECTED ON REPLICATION — the token-level cliff is single-model (Phase 109)

Orgad et al., *LLMs Know More Than They Show* (ICLR'25), claim truthfulness information is
**concentrated in the exact answer tokens**, and that probing the last position or a pooled mean
misses it. The VLM analogue of an exact answer token is the **exact evidence token** — visual tokens
whose patch overlaps the annotated region. Qwen3-VL-2B, n=191, hidden states at 5 positions × 28
layers, out-of-fold logistic probes.

| position | best AUROC, `err` | `encfail` |
|---|---|---|
| **`last`** (final prompt position) | **0.714** | **0.733** |
| `evid` (mean over evidence tokens) | 0.587 | 0.648 |
| `evid_max` | 0.594 | 0.674 |
| **`rand`** (size-matched random region, same image, same pass) | 0.503 | 0.489 |
| `imgmean` | 0.541 | 0.563 |

### ✗ Their headline does not reproduce
`evid` − `last` = **−0.127** and **−0.085**. In text, the exact answer tokens beat the last position;
in vision the evidence tokens are **worse** than it. The control passes — `evid` − `rand` = +0.084 /
+0.158, so the evidence tokens carry more than mere region identity — but they carry *less than the
final position does*. (Phase 24's retraction is why that control is mandatory: without it a probe
separates "annotated object" from "random rectangle" and can read AUROC 1.000.)

### ✅ And the deeper version of their claim is confirmed, sharply

| best AUROC for `err` | below the cliff (n=83) | above the cliff (n=108) |
|---|---|---|
| `last` | 0.725 | 0.693 |
| **`evid`** | **0.513 — chance** | **0.656** |

> **The visual tokens covering the target predict the model's correctness only when the target is
> above the encoding cliff. Below it they are at chance.**

This is a **token-level measurement of the cliff**, obtained with no crop, no oracle arm and no
accuracy contrast — an independent route to §13/§66's conclusion. `last` is unaffected by the split
because the final position encodes the model's own uncertainty, which exists whether or not it saw
anything. The stratum contrast is within-position, so the optimistic best-layer selection biases both
sides equally.

### Incidental: the evidence token's content is an EARLY-layer quantity
`evid` peaks at **L1** (0.587 / 0.648) and decays to ~0.47–0.51 by L27. A visual token is most
informative about its own content immediately after the projector, and mixing washes it out — the
same phenomenon that made attention rollout collapse to 12.6% (§14V/phase 104).

### ✗ REPLICATION FAILED (Qwen2-VL, phase 109 q2)

| `evid` probe, best AUROC for `err` | below cliff | above cliff |
|---|---|---|
| Qwen3-VL-2B | **0.513 — chance** | **0.656** |
| **Qwen2-VL-7B** | **0.623** | **0.633 — no gap** |

On Qwen2-VL the evidence tokens are **equally informative on both sides of the cliff**, which is the
opposite of the claim. **1 of 2 — REJECTED.** And the mandatory control fails there too: on the
`encfail` target `evid` − `rand` = **+0.018**, i.e. not separable from region identity (Qwen3 gave
+0.084 / +0.158).

**What DOES replicate: Orgad et al.'s headline fails in vision on both models.** `evid` − `last` =
−0.127 / −0.085 (Qwen3) and −0.031 / −0.077 (Qwen2). Probing evidence tokens is consistently *worse*
than probing the final position — 2 of 2.

> **Consequence.** The "intervention-free proof of the cliff" is withdrawn. The cliff rests on the
> intervention evidence alone: 87% / 84% of errors fixed by re-spending the same budget, with the
> oracle arm flat across the threshold, both on two models. PAPER_FLOW §3's second pillar is removed.

⚠ **Process note.** I described this as "the paper's best opening" and "the strongest single-model
result in the project" before it had a second model. That was the fourth single-model result to fail
under scrutiny in one session, after the window sizer (§14U), the LLaVA scope claim (§14Y) and CoRe's
selection leakage (§15C). **Single-model results in this project fail replication often enough that
none should be characterised as strong before the second model lands.**


## §15B  ✗✗ HARD SCOPE BOUNDARY: crop-based allocation is INVALID for existence questions (Phase 111)

POPE, RePOPE-clean (482 items relabelled, 812 dropped as ambiguous), joined to COCO boxes,
**stratified** to over-represent the below-cliff band (overall numbers are therefore NOT comparable to
published POPE). Qwen3-VL-2B, n=688, budget drift **0.1%**.

| stratum | n | uniform@300 | uniform@600 | **crop@0.25** | random crop | oracle crop |
|---|---|---|---|---|---|---|
| below cliff | 90 | 76.7% | 70.0% | **13.3%** | 6.7% | 93.3% |
| cliff zone | 198 | 84.8% | 89.4% | **12.1%** | 16.7% | 97.0% |
| above cliff | 200 | 100.0% | 98.5% | **35.5%** | 44.5% | 89.5% |
| negatives | 200 | 88.5% | 89.5% | 93.5% | 94.0% | — |

crop − bar: **−56.7 [−66.7,−46.7]**, **−77.3 [−83.3,−71.2]**, **−63.0 [−69.5,−56.0]**.

### The cause is structural, not a localisation failure

The crop arm answers **"no" to almost everything** — 13% on present-object items, 93.5% on absent ones.
POPE asks *"Is there an X **in the image**?"*. A window showing 6% of the area is genuine evidence of
**absence** for that question: the model answers the crop correctly and the image incorrectly.

> **V\*Bench questions presuppose the target exists** ("what colour is the X"), so discarding scene
> context is safe. **POPE asks whether it exists**, and cropping destroys the evidence base a negative
> answer requires. Crop-based allocation is **invalid** for existence questions, not merely unhelpful.

### ⚠ And POPE cannot supply a ceiling: the oracle arm is CIRCULAR here

The oracle crop scores 93.3 / 97.0 / 89.5% — but on POPE, **knowing the GT box is knowing the object
is present**. The oracle leaks the label. Every other venue in this project uses the oracle arm as the
control that makes the contrast interpretable (§13, §14T); on POPE it cannot. That rules POPE out as a
venue rather than placing the method behind on it.

### The pre-registered prediction about negatives was WRONG, in an informative direction

Pre-registered: cropping might **raise** false positives, since there is no target to crop to. It
**lowered** them — 6.5% vs the bar's 10.5%, **−4.0pp [−9.0,+1.0]**. Same cause: the crop biases the
model toward "no". It looks like a benefit and is the identical defect, measured on the side where it
happens to help. Recorded as a failed prediction, per §14M and §14L(b).

### What it establishes

The **second hard boundary** of the method, both found by our own controls:
1. **Relational questions** (§14T) — one window cannot cover an evidence set 7.9× larger in area.
2. **Existence questions** (here) — removing scene context removes the basis for a negative answer.

It also converts this project's early move from POPE to V\*Bench from a pragmatic choice into a
measured one, and it means **MMBench is not worth running**: 512px images leave no budget axis, and
its general-VQA questions carry the same presupposition problem in milder form.


## §15C  ⚠ HEAD SELECTION WORKS; CoRe's CONTRASTIVE CRITERION DOES NOT REPLICATE (Phases 108, 108b, 108c)

Lu et al. (2510.10285) select heads by the **absolute** share of their attention on visual tokens.
CoRe (2510.02219) prove that criterion is flawed in text — it cannot penalise heads that also flood
irrelevant content, and they measure top-8 such heads dropping **below** the all-heads baseline — and
replace it with a head-level **contrastive** score. Both ported here. Qwen3-VL, n=191, 16 heads,
W=0.25.

| rule | needs labels? | coverage | vs all-heads mean (43.5%) |
|---|---|---|---|
| **`S_v` top-10% + max** (Lu et al., absolute) | **no** | **55.0%** | **+11.5 [+5.8,+17.8]** ✔ |
| `S_v` top-25% + max | no | 51.3% | +7.9 ✔ |
| CoRe top-10% + max, **IN-SAMPLE (leaky)** | — | 60.7% | +17.3 — **not a result** |
| **CoRe top-10% + max, OUT-OF-FOLD** | yes | **58.1%** | **+14.7 [+8.9,+20.4]** ✔ |
| CoRe top-25% + max, out-of-fold | yes | 54.5% | +11.0 ✔ |

### ✅ Head selection is real, and label-free
Keeping **1–2 heads of 16** and taking a max across layers is worth **+11.5pp** with no annotations
anywhere. Combined with §14V (the layer rule) this is the second axis of the same idea: the deployed
read-out averages indiscriminately over both heads and layers, and both averages destroy signal.

### ✗ CoRe's specific claim does not replicate
**CoRe − `S_v` = +3.1pp [−1.0,+7.3]** — direction consistent, significance absent. In text the
absolute criterion *fails*; here it works fine. And the practical verdict is sharper than the
statistical one: `S_v` needs **no labels at all** while CoRe needs boxes on the selection set, so a
non-significant +3.1pp does not justify the annotation cost.

### ⚠ The leakage was worth 2.6pp, and this is the third such correction today
Phase 108's first implementation scored heads on **each item's own GT box**: 60.7%. Fold-honest
selection, CoRe's actual protocol: **58.1%**. Reporting the first would have claimed near-parity with
the learned head (63.4%) on a number that was partly label leakage. Alongside the sizer (+8.9 → −1.0
across grids, §14U) and the LLaVA scope claim (which inverted at the other window, §14Y), that is
three first-pass numbers shrinking under a proper protocol in one session.

### The training-free ladder, and what it does to the method's margin

| locator | coverage |
|---|---|
| deployed block-mean argmax | 43.5% |
| `S_v` top-10% heads + max over layers | 55.0% |
| norm-weighted × max | 56.5% |
| CoRe OOF heads + max *(needs boxes)* | 58.1% |
| **learned head** | **63.4%** |

> **The head's lead over the best training-free locator is 5.3pp, not the 17pp it holds over the
> deployed baseline.** Every number in this project was measured against a baseline that averages
> indiscriminately over heads and layers. The paper must report the strong baseline, not the deployed
> one — this is the reviewer's first question and it should be answered before it is asked.


## §15D  ⛔ CORRECTION: phase 72b's HR-Bench negative used a CRIPPLED LOCALISER (Phases 116, 117)

Phase 72b has been this project's evidence that "allocation loses at 4K" since §5A, cited in every
scope statement including today's. Re-running the same arm on the same rows exposed a defect in it.

**The two runs are identical except for one line.** Joining phase 72b and phase 116 on HR-Bench
`index`, n=150 rows present in both:

| arm | phase 72b | phase 116 | predictions agree |
|---|---|---|---|
| uniform@300 | 55.3% | 55.3% | **100.0%** |
| uniform@600 | 62.0% | 62.0% | **100.0%** |
| **argmax@0.15** | **42.7%** | **62.0%** | 76.7% |

Labels agree 100%. Both uniform arms agree on **every prediction**, so images, prompt assembly,
budget, fit() and scoring are identical. Only the crop arm differs — and the **argmax cell is
identical on just 44.7% of items**, median displacement 0.118 of the image.

### The cause
Phase 72b localises with `rows[0]["question"]` — the **bare question, options stripped**. Every other
phase passes the full prompt (phase 71 uses `ex["text"]`, options included). Attention is
question-conditioned (§14R) and the read-out depends on which query token is read (phase 48), so
removing the options removes most of the text the localiser conditions on.

**Including the options is worth +19.3pp on the crop arm**, on identical rows.

### What it invalidates

| HR-Bench single-region, argmax@0.15 vs uniform@600 | |
|---|---|
| phase 72b, bare-question localiser, n=400 | **−14.8pp [−20.5,−8.8]** |
| phase 116, full-prompt localiser, n=83 | **+9.6pp [−1.2,+21.7]** |

> **"Allocation loses at 4K" is withdrawn pending re-measurement.** It was measured with a localiser
> we do not use anywhere else. Phase 72c re-runs the full n=800 with the corrected prompt; until it
> lands, neither sign may be claimed — the corrected number does not clear zero at n=83, and the
> unambiguous part is only the 21pp swing in the crop arm from a prompt change.

⚠ This also puts §14T/§15B's scope statements under review: the claim that V\*Bench is the only venue
where the method can work rested partly on 72b. Two diagnostics I proposed today to explain the 4K
failure — the budget-axis slope and resolution matching — were each refuted by measurement (§115,
§116); it now appears the failure they were trying to explain may not have been real.


## §15E  ★★★ THE 4K NEGATIVE WAS AN ARTEFACT — the method wins at 4K with the correct localiser (Phase 72c)

HR-Bench 4k, **n=800**, same rows as phase 72b. Both uniform arms agree with 72b on **100.0%** of
predictions, so the only change is the localiser prompt (§15D: 72b stripped the answer options).

| stratum | bar (uniform@600) | arm | 72b (bare question) | **72c (full prompt)** | 72c vs bar |
|---|---|---|---|---|---|
| **single** (n=400) | 64.5% | **head@0.15** | 58.2% | **72.0%** | **+7.5pp [+2.2,+12.5]** ✔ |
| single | 64.5% | argmax@0.15 | 49.8% | 67.0% | +2.5 [−2.8,+7.8] ✗ |
| cross (n=400) | 55.2% | head@0.15 | 36.8% | 41.5% | −13.8 [−19.5,−7.8] |
| ALL (n=800) | 59.9% | head@0.15 | 47.5% | 56.8% | −3.1 [−7.0,+0.9] |

CircularEval (all 4 cycles): uniform@600 46.0% · argmax 40.0% · head 44.0%.

> **The V\*Bench pattern reproduces exactly on a second benchmark**: the learned head clears the
> equal-compute bar on single-object questions, the plain argmax does not, relational questions
> lose, pooled does not clear. And this is at **W=0.15, unswept** — the deployed window.

### What is overturned
"Allocation loses at 4K" (§5A, §9B, §14E, and today's scope statements) was carried for six weeks on
a localiser that stripped the answer options. It is withdrawn. Two diagnostics proposed today to
explain that failure — budget-axis slope (§115) and resolution matching (§116) — were refuted by
measurement; they were explaining an artefact.

### Status
The method is now **2 benchmarks on Qwen3-VL** and **2 models on V\*Bench**. The crossing — Qwen2-VL
head on HR-Bench — is queued (phase 72a/72c-qwen2). Until it lands the method is not 2 × 2.


## §15F  ⛔ VOID: phase 93b (Qwen2-VL pruning on POPE/MMBench) patched the wrong modeling module

All four arms — none / layer-2 / late / random — returned **identical predictions on 400/400 items**,
CI [+0.0,+0.0]. That is the phase-42 signature of an intervention that never fired. Cause: the script
was a sed copy of phase 93 and kept `import ...qwen3_vl.modeling_qwen3_vl as QM`; the attention patch
was installed on Qwen3-VL's module while Qwen2-VL ran unpatched, so the prune bias was set but never
read. Phase 83 (which did prune Qwen2-VL correctly) imports the qwen2_vl module. Fixed and re-queued
behind the 72c-qwen2 run. The void file is kept as `phase93b_VOID_wrong_module.jsonl`.

**Rule reinforced:** an arm contrast of exactly +0.0 [+0.0,+0.0] is a pipeline fault, never a null.


## §15G  ✗ THE W-BY-CATEGORY INTERACTION IS 1 OF 2 — coverage crossing fails as specified (Phase 116b)

Qwen2-VL-7B, HR-Bench 4k, n=150, same rows and arms as phase 116. Void check passed (29/150 items
with all arms identical; a void run gives 150/150).

| pre-registered: W0.12 − W0.25 | Qwen3-VL (116) | **Qwen2-VL (116b)** |
|---|---|---|
| single (n=83) | +2.4 [−3.6,+8.4] | −4.8 [−13.3,+3.6] |
| **cross (n=67)** | **−11.9 [−23.9,−1.5]** ✔ | **−3.0 [−11.9,+6.0]** ✗ |

Direction consistent on `cross`, significance absent. **1 of 2 — the interaction is rejected as a
cross-model, cross-benchmark claim.** What *does* replicate is the coarser boundary: relational
questions lose under cropping on both models (argmax@0.15 vs bar on `cross`: −10.5 / **−32.8pp**),
which with V\*Bench's null on relational makes "cropping does not beat the bar on relational
questions" a **2 model × 2 benchmark** boundary — the first thing in this project to survive both
axes, and it is a *limit*, not a gain.

### Secondary, replicated: HR-Bench's budget axis is NOT exceptional
slope 300→600: Qwen3 **+6.7 [+1.3,+12.7]**, Qwen2 **+8.7 [+3.3,+14.7]**; V\*Bench +7.3 [+0.7,+14.0].
The premise that 4K is special because "spending tokens obviously helps more there" is not supported
on either model at the doubling the method is charged for.

### Also on the record
argmax@0.15 on single vs bar: Qwen3 +2.5 [−2.8,+7.8], Qwen2 **−6.0 [−18.1,+4.8]**. The plain argmax
clears on neither model at 4K; only the learned head did (§15E, Qwen3). Whether the head clears on
Qwen2 at 4K is phase 72c-qwen2, queued.


## §15H  ★★★ BOTH CROSSINGS PASS — the first claims to survive models × benchmarks (Phases 72c-q2, 93b)

### The method: 2 models × 2 benchmarks on single-object questions
Qwen2-VL-7B head (trained on V\*Bench, phase 72a-q2), applied zero-shot to HR-Bench 4k, n=800,
full-prompt localiser, W=0.15 unswept. Budget 586 vs 584 tokens. Internal control passed: on the 84
rows where head and argmax chose the same cell the contrast is exactly +0.0.

| single-object, head vs equal compute | |
|---|---|
| Qwen3-VL, V\*Bench (n=115) | +15.7 [+6.1,+25.2] ✔ |
| Qwen2-VL, V\*Bench (n=115) | +11.3 [+1.7,+20.9] ✔ |
| Qwen3-VL, HR-Bench 4k (n=400) | +7.5 [+2.2,+12.5] ✔ |
| **Qwen2-VL, HR-Bench 4k (n=400)** | **+8.5 [+3.8,+13.2]** ✔ |

And the rest of the pattern replicates in every cell: the plain argmax does **not** clear (Qwen2 4K:
−5.8 [−10.8,−0.8]), relational loses (−8.8), pooled is null (−0.1 [−3.6,+3.4]). head − argmax on
single at 4K: **+14.2 [+9.8,+19.0]**. CircularEval: bar 44.0%, argmax 28.0%, head 45.0%.

### The pruning collapse: 2 models × 3 benchmarks
Qwen2-VL-7B, 10% keep from layer 2, block L15–26. Void check passed (arms differ; the first run's
283/400 identical rows are expected at 10% keep on near-ceiling benchmarks — cf. the void run's 400/400).

| | POPE | MMBench | V\*Bench (§14L(b)) |
|---|---|---|---|
| layer-2 − random | **−11.0 [−17.0,−5.0]** | **−5.5 [−10.5,−1.0]** | −4.7 |
| late − layer-2 | **+12.0 [+5.0,+19.0]** | **+8.0 [+3.5,+12.5]** | +11.5 |
| late − none | −4.0 [−8.5,+0.0] | −2.5 [−6.5,+1.5] | −1.6 |

**Ranking visual tokens by layer-2 attention is worse than ranking them at random on two
architectures and three benchmarks.** "Late pruning is free" does *not* generalise (−4.0 / −2.5 off
V\*Bench), consistent with §14P's earlier retraction of that half.

### What this settles
Before today no claim in the project had been replicated across both axes. Now two have:
- **layer-2 pruning below random** — 2 × 3
- **the method beats equal compute on single-object questions** — 2 × 2
plus one boundary (relational questions lose under cropping, 2 × 2, §15G). Both findings survive;
neither had to be demoted to make room for the other.


## §15I  ⛔ SYSTEMIC CORRECTION: every HR-Bench phase before 72c localised on the BARE QUESTION

Audit of every script that calls `localize()` / `propose()`:

| localiser text | phases | benchmark |
|---|---|---|
| full prompt (`ex["text"]`, options included) | 31, 32, 42, 45, 47, 58, 71, 80b, 111, 116, **72c**, 72c-q2 | V\*Bench, POPE, HR-Bench (72c only) |
| **bare question** (`grp["question"].iloc[0]` / `rows[0]["question"]`) | **33, 46, 53, 54, 56, 57, 59, 72b** | **every HR-Bench phase except 72c** |

Phase 33 even annotates it: *"question text only; options irrelevant"*. §15D measured what that
costs on identical rows: **+19.3pp on the crop arm** when the options are restored, with both uniform
arms agreeing on 100% of predictions.

### What this invalidates (all HR-Bench-based)
- **§5A/§5B** (phase 33): the 4K transfer negative and the +13.6pp attn−rand control — the *sign* of
  the control likely survives (random placement is worse still), its magnitude does not.
- **§8A** (phase 46): gating at 4K.
- **§9B** (phase 53): *our* arm's −0.2pp break-even. The prior-art arms (Zoom Eye, grounding) run
  their own localisation and are **unaffected** — their losses at matched budget stand.
- **§9C** (phases 53–56): "four attempts to win at 4K, all negative" — all four crippled.
- **§9D** (phase 57): the scale sweep and the "coverage absorbs the boundary" account.
- **§11B** (phase 59): multi-crop at 4K, "the win does not transfer".
- **§14E** (72b): already superseded by §15E.

### What replaces them
Phase 72c / 72c-qwen2 with the full-prompt localiser: the head clears the equal-compute bar on
single-object at 4K on **both** models (+7.5 / +8.5). The "scale boundary" this project spent five
phases explaining (§9C, §9D, §115, §116) **was a prompt bug**. The V\*Bench half of the project is
untouched — every V\*Bench phase used the full text.

### Process
The bug survived six weeks and eight phases because each new HR-Bench script was cloned from phase
33. It was found only when the multi-benchmark standard was applied and a fresh script (116) was
written from the V\*Bench template instead. **Rule: never clone a localiser; import one.**


## §16B  ✗ ARCHITECTURE SEARCH, ROUND FOUR: a spatial CNN does not beat the tree (Phase 130, Track 2)

Spatial 3×3 CNN over the 28-channel map (+ ranks + geometry), MSE and pairwise losses, flip
augmentation, seed ensembles, GBT+CNN rank ensembles — all four models, folds identical to phase 70.
**Nothing clears on both Qwen models**; the CNN is significantly worse on Qwen2-VL at both windows
(−8.9 / −9.4). With phases 45, 102 and 112 this is the fourth independent failure of a more expressive
head: **the limit is 191 boxed items, not the model class.** Full table in `REPORT_track2.md`.


## §16A  ★★ READ ATTENTION AT THE ANSWER POSITION — the 72b effect was the instruction line, not the options (Phase 121, Track 1)

Localisation-prompt sweep, 5 variants × 191 items, both models, prior art checked first (ViCrop's
locate-first prefix tested as V1; LookWise's noun queries not repeated — phase 48 already found noun
attention at chance).

| localisation prompt | Q3 argmax | Q3 head OOF | Q2 argmax | Q2 head OOF |
|---|---|---|---|---|
| **V0 question + options + answer-instruction (current)** | **46.1%** | **61.8%** | **39.8%** | **56.0%** |
| V1 ViCrop locate-first prefix + V0 | 43.5% | 61.8% (+0.0) | 42.4% | 57.6% (+1.6) |
| **V2 question + options, NO instruction** | **3.7%** | 45.5% (−16.2) | **0.5%** | 42.9% (−13.1) |
| V3 question + options + "which region…?" | 4.2% | 40.8% (−20.9) | 0.5% | 36.6% (−19.4) |
| V4 bare question (the 72b bug) | 9.4% | 44.5% (−17.3) | 4.2% | 40.8% (−15.2) |

**No variant beats V0.** But V2 re-attributes §15D/§15I: with the options present and the
instruction line absent, the argmax collapses to **3.7% / 0.5%** — *below* the ~2% chance of hitting
the target cell, and worse than the bare question. **What matters is that the prompt ends at the
answer-emission point.** The read-out is the final token's attention; when that token is the tail of
option (D) instead of the answer slot, its attention is about that text.

> **Scope condition for every attention-guided localiser:** read attention at the answer position.
> Generalises phase 48 (final token best of seven query tokens). Plausibly explains published
> "random beats attention" grounding nulls whose prompts end elsewhere (e.g. ACL Findings'25 on
> RefCOCO). 2 models, V\*Bench; HR-Bench prompts in 72c already end with the instruction.

⚠ §15D/§15I are corrected accordingly: 72b's `rows[0]["question"]` lacked *both* options and
instruction; the sweep shows the instruction is the operative half. One variant not yet run —
question + instruction, no options — would fully isolate it.

Track 1's other levers — encfail up-weighting (118), alternative targets (122), top-k verification
(120, argued) — are all negative on both models; the head is at a data-limited optimum on every axis
(architecture, features, weighting, target, prompt). **Noise floor revised to 1–2.5pp** (the incumbent
re-reads 63.9/57.1 and 61.8/56.0 across seedings vs 63.4/54.5). Full report: `REPORT_track1.md`.


## §16C  ⚠ AN INTUITIVE HEAD: a 63-parameter log-linear loses to the tree on Qwen but is the most family-general (Phase 131)

Three readable replacements, phase-70 folds, four models, two windows. Δ vs the GBT:

| W=0.15 | GBT | A log-linear (Σ w_l·log A_l + Σ v_l·rank_l + geo, Lasso) | B additive GAM | C rank fusion |
|---|---|---|---|---|
| Qwen3-VL | 53.4 | 50.8 (−2.6 [−7.9,+2.6]) | 45.0 (**−8.4**) | 26.2 (**−27.2**) |
| Qwen2-VL | 44.0 | 41.9 (−2.1 [−6.8,+2.6]) | 44.0 (+0.0) | 18.8 (**−25.1**) |
| LLaVA-NeXT | 19.9 | 20.9 (+1.0) | 18.3 (−1.6) | 15.2 (−4.7) |
| **LLaVA-OneVision** | 24.1 | **30.9 (+6.8 [+2.1,+11.5])** ✔ | 28.3 (+4.2) | 13.1 (−11.0) |

W=0.25: log-linear −3.1 / **−6.8 [−11.5,−2.1]** on Qwen; +5.2 / +0.5 on LLaVA.

- **No arm reaches non-inferiority (≥ GBT − 1.5pp) on both Qwen models.** The tree stays.
- **Rank fusion is catastrophic (−25pp)**: ordinal information alone is insufficient — the magnitudes
  carry the signal, consistent with norm weighting helping and rollout failing.
- **The log-linear is the only head that works on LLaVA-OneVision** (+6.8 over the tree at W=0.15,
  CI clear), the model where the tree, max-over-layers and everything else fail. Provisional: one
  model, one window (W=0.25 is +0.5 n.s.). If it holds, the readable head is the more *portable* one
  and the tree the more *accurate* one — a trade-off worth one paragraph, not a replacement.

### §16C addendum — the log-linear clears the DEPLOYED argmax on all four architectures (W=0.15)
| vs deployed | Qwen3-VL | Qwen2-VL | LLaVA-NeXT | LLaVA-OneVision |
|---|---|---|---|---|
| tree | +14.1 ✔ | +8.9 ✔ | +7.3 ✔ | −1.0 ✗ |
| **log-linear** | +11.5 ✔ | +6.8 ✔ | **+8.4 [+3.1,+13.6]** ✔ | **+5.8 [+0.5,+11.5]** ✔ |

**4 of 4 for the readable head vs 3 of 4 for the tree**, at a 2–3pp cost on Qwen. At W=0.25 OneVision
is +2.1 [−4.2,+7.9] (n.s.) and NeXT +10.5 ✔. The portability claim is W=0.15-only until replicated.

### §16C addendum 2 — what the readable head reads (standardised Lasso weights on log-attention, W=0.15)
| | largest + | largest − | geometry |
|---|---|---|---|
| Qwen3-VL | **L19** +.110, L5, L17, L8, L24 | **L26** −.078, L10, L3, L22, L23, **L27** | 3×3 nb **+.116**; sink flags ≈0 |
| Qwen2-VL | **L19** +.064, **L21** +.051, L16 | L7 −.069, L11, L17, L23, L25 | 3×3 nb **+.137**; sink flags ≈0 |

Read directly: both models weight **L19** most; Qwen2 adds **L21**, its answer-formation layer
(§14I(b)); Qwen3's last layers (L26, L27) carry *negative* weight — the anti-correlated-final-layer
observation of §14F, Qwen3-specific as §14K found; the neighbourhood term dominates geometry and the
sink indicators are ≈0, matching the ablation (§14C(b): profile +7.3, sink +0.0). This is the tree's
mechanism stated in 63 numbers. Queued: phase 121b (question + instruction, no options) to finish
isolating the answer-position effect.


## §16D  ★★ THE NO-TRAINING COUNTERPART: max over the question-conditioned layers (Phases 140–141, Track 3)

All rules fixed-constant, label-free, no OOF selection. W=0.25, ring-masked top-1 coverage.

| rule | needs | Qwen3-VL | Qwen2-VL |
|---|---|---|---|
| deployed: mean over block | — | 46.1 | 39.8 |
| max over block (CLAA) | — | 53.4 | 46.6 |
| norm-weighted max | ‖v‖ | 56.5 | 48.7 |
| heads + ‖v‖ + max (composite) | per-head | 56.5 | 49.7 |
| **max over the divergence-gated layer set** (raw maps) | phase-95 curve | 56.0 | **52.4** |
| composite, divergence-gated | all | **59.7** | 51.8 |
| *learned head* | boxes | *63.4* | *54.5* |

**The three sink fixes are not additive** — norm weighting, head selection and max each beat the
deployed rule, but stacked they land on the same ~56 / ~49 (composite − nw-max: +0.0 / +1.0). **The
layer SET is the lever that was left.** Replacing the hand-set block with the layers where attention
is question-conditioned (phase 95, label-free) is the only component that adds on top: +3.1 / +3.1
over nw-max, and on raw maps alone **+3.7 [+1.0,+6.8]** on Qwen2. The gate is threshold-robust
(0.3/0.5/0.7 select the same layers: **L17–20** on Qwen3, **L19–22** on Qwen2 — independently
re-finding Qwen2's L21). Gap to the learned head: **3.7 / 2.7pp**, down from 17 / 15.

**Do not ship:** ReAttn entropy rescaling (−19 to −27pp on norm-weighted maps — early layers are
diffuse, any entropy reweighting promotes them); VEA denoising (−1 to −3pp on Qwen — isolated
high-value cells on V\*Bench are usually the *target*, the opposite of document VQA); phase-30d
background normalisation (null-to-negative once max/heads are present).

### End-task (phase 141): no training-free rule clears the bar on both models
`head@0.25` re-run matches the stored value on **100%** of items on both models.

| single-object vs uniform@600 | Qwen3-VL | Qwen2-VL |
|---|---|---|
| learned head | **+15.7 [+6.1,+25.2]** ✔ | **+11.3 [+1.7,+20.9]** ✔ |
| **max over divergence-gated layers, raw, label-free** | **+10.4 [+0.9,+20.9]** ✔ | +7.0 [−2.6,+16.5] ✗ |
| composite, divergence-gated (pre-registered primary) | +9.6 [−0.9,+20.0] ✗ | +7.8 [−1.7,+17.4] ✗ |
| CLAA max_win4 (§14W) | +9.6 [+0.0,+19.1] ✗ | +6.1 [−3.5,+15.7] ✗ |

**1 of 2 for the best label-free rule — rejected under the standing rule.** The learned head remains
the only arm clearing on both. The training-free counterpart is stated as: *a one-line, label-free
rule — max over the layers where attention is question-conditioned — recovering +10.4 / +7.0pp over
equal compute on single-object questions, closing the coverage gap to 3.7 / 2.7pp, and not
significantly clearing the bar on the second model.* Its advantage over CLAA's max_win4: the layer
set comes from a label-free measurement, not out-of-fold selection on boxes.

Cross-family: no training-free rule helps LLaVA; max hurts OneVision (−4.2). The divergence gate on
LLaVA (phase 142) is the one open test. Full report: `REPORT_track3.md`.

### §16C addendum 3 — per-layer neighbourhoods bring the readable head to parity at W=0.15 (Phase 132)
Three readable additions to the log-linear. Δ vs the GBT, phase-70 folds:

| W=0.15 | GBT | A+ (per-layer quadratic) | A++ (6 layer×nb products) | **A+nb (per-layer 3×3 neighbourhood, ~91 params)** |
|---|---|---|---|---|
| Qwen3-VL | 53.4 | 52.9 (−0.5) | 50.3 (−3.1) | **56.0 (+2.6 [−2.1,+7.3])** |
| Qwen2-VL | 44.0 | 43.5 (−0.5) | 42.4 (−1.6) | **45.5 (+1.6 [−3.1,+6.3])** |
| LLaVA-NeXT | 19.9 | 22.5 (+2.6) | 23.6 (+3.7) | **24.6 (+4.7 [−0.5,+9.9])** |
| LLaVA-OneVision | 24.1 | **33.0 (+8.9 [+4.7,+13.6])** ✔ | 30.9 (+6.8) ✔ | **30.4 (+6.3 [+1.6,+11.5])** ✔ |

W=0.25: A+nb −2.6 [−7.3,+2.1] on Qwen3 (fails the −1.5 rule, inside the CI), +2.1 on Qwen2, +6.3 /
+4.2 on LLaVA; A+ collapses on Qwen2 (−7.9 ✗).

> **At the deployed window, a log-linear model over log-attention, ranks and per-layer local means —
> ~91 readable parameters — is at parity with the tree on both Qwen models (within the 1–2.5pp noise
> floor, CIs spanning zero) and ahead of it on both LLaVA models.** Not superior anywhere on Qwen;
> at W=0.25 it is 3 of 4. The tree stays as the *accuracy* head for the method's end-task numbers;
> A+nb is the head to *show* — every weight is "layer l, this much, smoothed over its neighbours".
> The per-layer neighbourhood is the ingredient: it turns the tree's one 3×3 feature (on the block
> mean) into 28, one per layer, which is the spatial structure phase 130's CNN tried to learn from
> scratch and could not at n=191.

### §16C addendum 4 — what the parity head reads (A+nb, W=0.15; per-layer weight = point + neighbourhood)
| | strongest + | strongest − | nb mass / point mass | sink flags |
|---|---|---|---|---|
| Qwen3-VL | **L19** +.112, L5, L17, L24, L8 | **L10** −.101, **L26** −.096, L3, L22, L23 | **2.30** | ≈0 |
| Qwen2-VL | **L19** +.079, **L21** +.071, L16 | L17 −.062, L7, L11, L20 | **1.46** | ≈0 |

The head reads mostly *locally-smoothed* log-attention (neighbourhood weights carry 1.5–2.3× the
mass of point weights), led by **L19 on both models** and **L21 on Qwen2** (its answer-formation
layer), with late layers L26/L22/L23 subtracted on Qwen3 and mid layers L7/L11/L17 on Qwen2. Sink
indicators ≈0 on both. That is the complete description of a head at parity with the tree.

### §16D addendum — the divergence gate on LLaVA (Phase 142): mechanism replicates, rescue does not
New label-free question-divergence curves for both LLaVA models. **The switch-on replicates**:
L14 of 32 (44% depth) on LLaVA-NeXT, L16 of 28 (57%) on LLaVA-OneVision — against L14→16 of 28 on
both Qwen models. **Attention is question-blind until roughly half depth on four architectures across
two families** (§14R is now 4 × 1). But max over the gated layers does not rescue LLaVA localisation
(+2.1 n.s. on NeXT, −4.2 on OneVision): every training-free rule is Qwen-only; the learned head
(3 of 4) and the readable head (4 of 4 vs deployed, §16C) are the only things that cross the family
line. Curves in `data/phase142_llava_divergence.json`.

### §16A addendum — fully isolated (Phase 121b): the options contribute nothing
| localisation prompt | Q3 argmax | Q3 head | Q2 argmax | Q2 head |
|---|---|---|---|---|
| V0 question + options + instruction | 46.1 | 61.8 | 39.8 | 56.0 |
| **V5 question + instruction, NO options** | **47.6** (+1.6 [−3.1,+6.3]) | 58.6 (−3.1 [−7.3,+1.0]) | **41.4** (+1.6 [−2.1,+5.2]) | 55.0 (−1.0 [−6.3,+4.2]) |
| V2 question + options, no instruction (121) | 3.7 | 45.5 | 0.5 | 42.9 |

Removing the options changes nothing; removing the instruction collapses the map. **The attention
read-out works when — and only when — the prompt ends at the answer-emission point.** 2 models,
both directions tested. §15D/§15I's "restored the options" is corrected to "restored the
answer-emission point" (72b's bare question lacked both; only the second mattered).


## §16E  ✗✗ MORE BOXED DATA DOES NOT HELP — the head's read-out is TASK-SPECIFIC (Phases 133–134)

3,000 TextVQA items with Visual-CoT evidence boxes (public), Qwen3-VL attention dumped with the
phase-30c extractor and a prompt ending at the answer-emission point. Pre-registered: a TextVQA-
trained head applied zero-shot to V\*Bench must beat the 191-item OOF head for "data was the limit".

| V\*Bench top-1 coverage | W=0.15 | W=0.25 |
|---|---|---|
| deployed argmax | 39.3 | 46.6 |
| **V\*Bench OOF GBT (incumbent, 191 items)** | **53.4** | **64.4** |
| TextVQA-only GBT → V\*Bench, zero-shot (3,000 items) | **38.2 (−15.2 [−21.5,−9.4])** | **37.2 (−27.2 [−34.0,−20.9])** |
| TextVQA-only readable head → V\*Bench | 24.1 (−29.3) | 35.1 (−29.3) |
| TextVQA **+** V\*Bench (OOF) GBT | 49.2 (−4.2 [−8.9,+0.5]) | **58.6 (−5.8 [−11.0,−1.0])** |

**Sanity passed on TextVQA itself** (OOF: GBT 46.5 / 62.9 vs argmax 38.3 / 54.7), so the head
learns — but what it learns does not carry to a different question type. Sixteen times more boxed
items from an OCR-style task produce a head *worse than no head*, and mixing them in degrades the
V\*Bench head significantly. **The data route is closed unless the data is same-task.**

### What this says, joined to what we already had
- Zero-shot transfer **within** a question type works: the V\*Bench head on HR-Bench (object/attribute
  questions) is +4.9 coverage and clears the bar at 4K on both models (§15E/§15H).
- Zero-shot transfer **across** question types fails: text-reading → object attributes, −15 to −27pp.
- Orgad et al. (ICLR'25) found truthfulness probes are "skill-specific" in LLMs; the attention→evidence
  read-out is skill-specific in VLMs the same way. **The head is not a universal attention decoder;
  it is a per-task one**, and the paper must say so.

⚠ One confound not yet separated: TextVQA prompts are open-ended (no options), V\*Bench are MCQ. The
attention profile could be prompt-format-conditioned rather than task-conditioned. Phase 121b showed
options change nothing *within* V\*Bench, which argues against it, but a V\*Bench-style MCQ source
with different content (Visual7W, gated) would settle it.

### §16E addendum — why it fails: the switch-on layer is task-dependent
Median gt_pct per layer (0.500 = chance, lower = the layer ranks the target higher), Qwen3-VL:

| | L11 | L12 | L13 | L14 | L15 | **L16** | L17 | L19 | L21 | L24 | L27 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V\*Bench (191) | .279 | .282 | .218 | .296 | .269 | **.068** | **.023** | .040 | .058 | .071 | .377 |
| TextVQA (3,000) | **.136** | **.133** | **.043** | **.043** | .053 | .024 | .023 | .017 | .017 | .017 | .230 |

Spearman between the two profiles 0.71; best-5 layers overlap (19, 20, 21). But **question-conditioned
localisation switches on at L11–13 for text reading and at L16 for small-object attributes.** A head
trained on TextVQA learns to trust the mid layers; on V\*Bench those layers are still question-blind
(§14R), so it ranks on noise — below the deployed argmax. This is Lu et al.'s (2510.10285) Obs. 5,
*task-dependent boundary bands*, measured with our label-free gt_pct on a second task: the band is
not a property of the model alone but of (model, task). It also says what same-task data would have
to be: same switch-on layer, not just same image domain.


## §17  REVIEWER-DRIVEN EXPERIMENTS (2026-09-17, evening)

### §17A  ★ W1 — routing by question type makes DPR clear the bar POOLED (Phase 150)
Composition at exactly matched tokens: routed items get localise@300 + crop@300, unrouted items get
uniform@600, scored against uniform@600 everywhere.

| model | bench | router | routed | pooled − bar |
|---|---|---|---|---|
| Qwen3-VL | V\*Bench | none (always DPR) | 100% | +7.9 [−0.5,+16.2] ✗ |
| Qwen3-VL | V\*Bench | **free keyword rule** | 60.2% | **+9.4 [+3.1,+15.2]** ✔ |
| Qwen2-VL | V\*Bench | **free keyword rule** | 60.2% | **+6.8 [+1.0,+12.6]** ✔ |
| Qwen3-VL | HR-Bench | free keyword rule | 84.5% | −1.6 [−5.5,+2.1] ✗ |
| Qwen3-VL | HR-Bench | oracle category | 50.0% | **+3.8 [+1.1,+6.5]** ✔ |
| Qwen2-VL | HR-Bench | oracle category | 50.0% | **+4.2 [+1.9,+6.8]** ✔ |

The keyword rule (left/right/above/below/next to/between/behind/…) matches V\*Bench's category
labels on **100%** of items and lifts the pooled contrast over the bar on both models. On HR-Bench its
agreement is 51.5% — that benchmark's relational questions read "relative position of X compared to
Y", "how many", "where is" — so pooled is null there with the keyword rule and clears with an oracle
router. **The composition works when the router works.** Phase 153 (queued) tests a router that is not
tuned on any evaluation set: the model itself, text-only, zero-shot.

### §17B  ★ W2 — fifty boxed items suffice (Phase 151)
Train on k items, evaluate on the rest, 10 random draws, both models, W=0.25:

| k | Qwen3-VL head − argmax | Qwen2-VL |
|---|---|---|
| 10 | +0.4 (sd 7.3) | +2.1 (sd 5.3) |
| 25 | +7.6 (sd 4.4) | +8.2 (sd 3.9) |
| **50** | **+11.9 (sd 2.3)** | **+11.4 (sd 2.8)** |
| 100 | +10.3 | +12.3 |
| 191 (OOF) | +16.8 | +16.2 |

Fifty boxed items recover ${\approx}70\%$ of the full gain with tight spread. Together with §15E/§15H
(zero-shot to HR-Bench) and §16E (task-specific), the supervision requirement is: **~50 boxes from
one benchmark of the target question type, once.**

### §17C  W3 — pooled single-object n
Across V\*Bench + HR-Bench single-object: Qwen3-VL **+9.3 [+4.7,+14.0]** (n=515), Qwen2-VL
**+9.1 [+4.9,+13.4]** (n=515); both models **+9.2 [+6.1,+12.4]** (n=1030). The per-cell lower bounds of
+1.7/+2.2 were the smallest cells; the pooled estimate is not thin.

### Queued
W4 latency (phase 152, both models) · W1 model-as-router (153) · W3/W6 HR-Bench 8K (155) · W6 Qwen3-VL-8B
and Qwen2.5-VL-7B full pipeline (154).

### §17D  W4 — wall-clock latency (Phase 152; batch 1, n=50, median ms/item, CUDA-synchronised)
| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| bar: uniform@600, one sdpa pass | 314 | 335 |
| DPR: localise@300 (eager + `output_attentions`) + crop@300 | 577 = **1.84×** | 676 = **2.02×** |
| DPR: localise@300 (sdpa + last-token hook at block layers) + crop@300 | 560 = **1.78×** | 467 = **1.40×** |

Tokens match; wall-clock does not: DPR is $1.4$--$1.8\times$ the bar with a deployable localiser, $1.8$--$2.0\times$ with
the naive one. The eager `output_attentions` path materialises every layer's full attention; a forward hook
that computes only the last query row at the block layers removes most of that overhead on the 7B model.

### §17E  ★ W1 — an UNTUNED router: the model itself, text-only, zero-shot (Phase 153)
"Does answering this question require comparing, counting, or locating two or more objects relative to
each other? Yes/No" — no image, ~60 tokens, no tuning on any evaluation set. Route to DPR iff No.

| model | bench | routed | agreement w/ category | pooled − bar |
|---|---|---|---|---|
| Qwen3-VL | V\*Bench | 63.9% | 96.3% | **+10.5 [+4.2,+16.8]** ✔ |
| Qwen2-VL | V\*Bench | 59.2% | 99.0% | **+6.8 [+1.0,+12.6]** ✔ |
| Qwen3-VL | HR-Bench | 75.0% | 74.0% | +0.4 [−3.1,+3.9] ✗ |
| Qwen2-VL | HR-Bench | 49.5% | 88.5% | **+2.8 [+0.4,+5.1]** ✔ |

**3 of 4 cells clear pooled with an untuned, free router.** The one miss is the 2B model on HR-Bench, where
it over-routes (75% vs 50% relational) — the weaker text classifier, not the method. With the oracle router
all four clear (+9.4, +6.8, +3.8, +4.2). The composition is built and evaluated; it is not an oracle.

### §17F  ★ W3/W6 — a THIRD benchmark: HR-Bench 8K (Phase 155, Qwen3-VL-2B head zero-shot, n=800)
single: **head +10.5 [+5.2,+16.0]** ✔ (62.3% vs bar 51.7%); argmax +3.5 [−1.5,+8.8] ✗; random −15.5.
cross: head −9.2; pooled +0.6 [−3.4,+4.6]. CircularEval: bar 33.5, argmax 33.0, head 36.5. Same pattern as
V\*Bench and 4K: the head clears on single-object, the argmax does not, relational loses, pooled is null.

### §17G  W6 — scale and generation (Phase 154, V\*Bench, W=0.25 transferred, block = same stack fraction)
| model | NL | OOF coverage head / argmax | single head−bar | single argmax−bar | pooled head−bar |
|---|---|---|---|---|---|
| **Qwen2.5-VL-7B** (new generation) | 28 | 60.7 / 44.0 | **+14.8 [+4.3,+25.2]** ✔ | +7.8 [−1.7,+17.4] | +6.3 [−1.6,+14.1] |
| **Qwen3-VL-8B** (4× scale) | 36 | 60.7 / **31.9** | +4.3 [−5.2,+13.9] ✗ | **−14.8 [−25.2,−4.3]** | −1.6 |

Qwen2.5-VL-7B is a **third checkpoint** on which DPR clears the bar on single-object questions (now Q3-2B,
Q2-7B, Q2.5-7B on V\*Bench; Q3-2B and Q2-7B on HR-Bench 4K; Q3-2B on 8K). Qwen3-VL-8B does **not** clear:
its deployed argmax is far worse (31.9% coverage — the read-out defect grows with scale) and the head
recovers +28.8pp of coverage and +19pp over the argmax end-task, but the 8B model at uniform@600 is strong
enough (65.2% single) that the crop's gain no longer clears the bar. Honest reading: **the read-out
defect is larger at scale; the allocation win is smaller**, because the bar rises faster than the crop.


## §18  ARCHITECTURE FIXES FOR THE TWO DEPLOYABILITY CRITICISMS

### §18A  ✗ Fix 1 — depth-aligned features do NOT make the head task-agnostic (Phase 160)
Re-index the profile relative to the task's switch-on layer $s$ (18 layers, $s{-}6\ldots s{+}11$) and
transfer TextVQA $\to$ V\*Bench zero-shot.

| TextVQA head → V\*Bench | W=0.15 (argmax 39.3) | W=0.25 (argmax 46.6) |
|---|---|---|
| unaligned (same absolute layers) | 27.2 (−12.0 [−17.8,−6.3]) | 26.7 (−19.9) |
| **aligned, $s$ from gt profile (16 / 12)** | **33.5 (−5.8 [−11.0,−0.5])** | **34.0 (−12.6)** |
| aligned, $s$ from a label-free max-share proxy (13 / 11) | 28.3 (−11.0) | 23.6 (−23.0) |

Alignment recovers **6–7pp** of the cross-task gap — the switch-on shift is a real part of the
mechanism — but the aligned head is still significantly *worse than no head*, and the label-free
proxy for $s$ (first layer whose ring-masked max share doubles the early-layer median) selects the
wrong layer. **The read-out remains task-specific under depth alignment.** Recorded as a negative
with a mechanism confirmation inside it; the head's supervision must come from the target question
type (§17B: ~50 boxes).

### §18B  ✗ (interim, Qwen3-VL) Fix 2 — the peaks-spanning window as specified fails (Phase 161)
Rule: window = bbox of the top-2 peaks of the OOF head score map if peak2 ≥ 0.5·peak1 (NMS 3 cells).
**k=2 fired on 90% of single-object items** — the GBT map is a smooth coverage regression and its
second peak is routinely comparable — so the span window costs **−11.3 [−20.0,−2.6]** on single-object
vs the fixed W=0.25 head, and does not help relational (−2.6 [−15.8,+9.2] vs the bar; head +0.0).
Median span area 0.100 vs 0.0625. The criterion is wrong, not necessarily the idea; a diagnostic of
which untuned map statistics separate the two question types follows. Qwen2-VL leg and the
threshold-free mass-containment variant (163) pending.

#### §18B diagnostic — which untuned map statistic sees question type?
Median per item, AUROC for relational (both models):

| map | statistic | single | relational | AUROC Q3 / Q2 |
|---|---|---|---|---|
| GBT score | peak2/peak1 | 0.755 / 0.694 | 0.825 / 0.813 | 0.62 / 0.65 |
| **raw gated-max attention** | **top-1 share** | **0.111 / 0.207** | **0.064 / 0.079** | **0.735 / 0.805** |
| raw gated-max attention | top-5 share | 0.298 / 0.422 | 0.206 / 0.243 | 0.70 / 0.80 |

**The raw question-conditioned map is concentrated for single-object questions and diffuse for
relational ones; the GBT map is not** (it regresses coverage and is smooth everywhere). Fix 2 must
therefore read dispersion of the *raw* gated map, not peaks of the score map. Phase 163 patched
accordingly before running: mass box grown on the raw gated map; concentrated maps fall back to the
head's W=0.25 window.

### §18B (final)  ✗ Fix 2 as specified — the peaks-spanning window is 0 of 2 (Phase 161)
| span − bar, relational | span − head, single |
|---|---|
| Qwen3-VL −2.6 [−15.8,+9.2] ✗ · Qwen2-VL −1.3 [−14.5,+11.8] ✗ | −11.3 [−20.0,−2.6] · −1.7 [−11.3,+7.8] |

k=2 fired on 90–97% of items on both models. Rejected; the diagnostic above locates the usable
signal in the raw gated map, which phase 163 tests.

### §18C  ✗ Fix 3 — the box-free pseudo-label head is 1 of 2 (Phase 162)
Pseudo-target = max-softmax gain under cropping at each of the top-6 gated-max candidates; no boxes,
no answer labels; OOF GBT; real coverage and end-task of the OOF pick.

| | OOF coverage vs deployed argmax | single-object vs bar | pooled vs bar |
|---|---|---|---|
| Qwen3-VL | **45.0 vs 46.6** ✗ | −5.2 [−15.7,+6.1] | −5.8 |
| Qwen2-VL | **47.1 vs 39.3** ✔ | +4.3 [−4.3,+13.9] ✗ | +5.8 [−2.1,+13.6] |

Consistent with §6E/§8A: the two-pass confidence signal is a strong coverage detector on Qwen2-VL
(AUROC 0.885) and weak on Qwen3-VL. As a label-free head it is rejected under the standing rule; a
pre-registered variant with a different label-free target (agreement with the model's own
600-token answer, i.e. self-distillation from the bar) is queued as 162b.


### §18D  ✗ Fix 4 — a map-only dispersion ROUTE does not carry the pooled claim (Phase 156)

Standing constraint 7 permits question-type adaptation **only from the attention map**. The §18B
diagnostic gives the map-side signal: top-1 mass share of the raw gated-max map separates
single-object from relational at AUROC 0.735 / 0.805. This tests it in the form the banned keyword
rule took — a hard route — before 163 tests it in the permitted form, a window that adapts
continuously. Composition, thresholds and scoring are phase 150's, so the rows are comparable line
for line; τ is chosen out-of-fold, GroupKFold(5) grouped by item (V\*Bench) / instance (HR-Bench).

| model | bench | router | routed | pooled − bar |
|---|---|---|---|---|
| Qwen3-VL | V\*Bench | R1 keyword (banned, context) | 60.2% | +9.4 [+3.1,+15.2] ✔ |
| Qwen3-VL | V\*Bench | **R3 top1_frac, map-only, OOF** | 71.2% | **+2.6 [−4.7,+9.4]** ✗ |
| Qwen2-VL | V\*Bench | **R3 top1_frac, map-only, OOF** | 69.6% | **+7.3 [+1.0,+13.6]** ✔ |
| Qwen3-VL | HR-Bench | R3 top1_frac (block-mean, phase 46) | 5.0% | −1.1 [−1.9,−0.5] ✗ |
| Qwen3-VL | HR-Bench | R2 oracle category (ceiling) | 50.0% | +3.8 [+1.1,+6.4] ✔ |

**The pre-registered primary fails: 1 of 2 on V\*Bench.** R3 − R1 is **−6.8 [−12.0,−1.6]** on Qwen3
and +0.5 [−4.2,+5.2] on Qwen2 — on Qwen3 the map-only route is significantly *worse* than the text
rule it would replace. The AUROC replicates exactly (0.735 / 0.805 gated-max; 0.745 / 0.839
block-mean) so the signal is real — and it reproduces §18B's diagnostic from independent code, which
is a genuine cross-check rather than a re-read. **Detection at AUROC 0.74 is not enough to route when
the alternative is a rule that is 100% concordant with the labels.** Agreement with the category
oracle: R3 63.9% / 69.6%, R1 100%.

⚠ Direction: `entropy_norm` predicts the *relational* class, so its printed AUROCs (0.315 / 0.296 /
0.209 / 0.170) are 0.685 / 0.704 / 0.791 / **0.830** as a single-object detector — on Qwen2 block-mean
the strongest of the four. `top1_frac` was pre-registered as primary and substitution was barred, so
this is recorded, not acted on.

On HR-Bench the statistic is at **chance** (AUROC 0.515), so the threshold has nothing to fit and
collapses to routing 5% of rows — **40 rows, 10 distinct instances** — i.e. "almost never crop". The
−1.1pp attached to that row is **not** a loss claim: 95% of rows are exact ties against the bar and
contribute no variance, and `ci()` bootstraps rows rather than instances, so the interval is
understated twice over. The conclusion rests on the AUROC alone. §17A's HR-Bench null is therefore
not a keyword-rule artefact: **no question-type signal of this kind is present in that benchmark's
block-mean maps at all.** The gated-max arm there is unrun (no per-layer HR-Bench maps on disk; one
localisation pass per instance, queued), as is Qwen2-VL HR-Bench.

> **What this leaves for fix 2b.** A hard route is the wrong shape for a map-only signal this noisy:
> it spends the whole AUROC on one binary decision. Phase 163's mass-containment window uses the same
> statistic *continuously* — a concentrated map yields a small window, a dispersed one a larger window
> — so a mid-confidence item degrades gracefully instead of being misrouted. 156 is the control that
> says the continuous form is the one worth running, not a redundant second router.

Free, CPU-only, 2.3 s, 237 MB peak: `scripts/phase156_dispersion_router.py`,
`data/phase156_dispersion_router.json`.

### §18D  ✗ (Qwen3-VL) Fix 2b — the mass-containment window fails for a legible reason (Phase 163)
Mass box grown on the raw gated map to q=0.6 of its mass, falling back to the head's W=0.25 window when
the box is concentrated. **The box is huge for everyone**: median area 0.286 (single) / 0.308
(relational) against 0.0625, and only 4% / 1% of items fell back to the tight crop. Single-object
**−19.1 [−27.8,−10.4]** vs the head window; relational +1.3 [−11.8,+14.5] vs the bar (null); pooled
−1.6. The raw map is *peaked* but its *mass* has a long tail (median top-1 share 0.11 even on
single-object), so 60% of the mass spans ~30% of the image regardless of question type. The dispersion
signal (§18B diagnostic, AUROC 0.74/0.81) is **relative**, not absolute, and a fixed mass fraction
cannot read it. This is the second geometric window to fail after the peaks rule; with the closed
per-item sizer (§14U) it is the third attempt to derive a window from the map. **Not tuning q.**
Qwen2-VL leg pending as the formal second cell.

### §18E  (analysis only — no method change) a map-concentration skip clears pooled on both models
Not run as a method; computed from stored outcomes to inform a decision. Apply DPR only when the raw
gated map's top-1 share is above the median of the training fold (OOF, label-free, no question text);
otherwise spend the 600 tokens on the whole image. Scored against uniform@600 everywhere.

| | routes | (single / relational) | always-DPR pooled − bar | map-skip pooled − bar |
|---|---|---|---|---|
| Qwen3-VL | 50% | 65% / 26% | +7.9 [−0.5,+16.2] ✗ | **+5.8 [+0.5,+11.0]** ✔ |
| Qwen2-VL | 52% | 71% / 22% | +6.8 [−1.0,+14.7] ✗ | **+5.8 [+0.5,+11.5]** ✔ |

The map's dispersion is enough to decide *whether to crop at all*, even though two attempts to turn
it into a *window size* failed (§18B, §18D). Whether a map-derived skip counts as a "router" under
constraint 7 is a framing decision left to the user; it uses no question text and no labels, and the
threshold is the fold median (no tuning).

### §18F  (analysis, on disk) the self-calibrated dispersion ladder — only the 2-rung version clears (Phase 165)
Dispersion $d$ = top-1 share of the raw gated map; per OOF fold, $d \to$ its quantile among training
items; the quantile selects an action whose constant is already validated. Pooled − bar:

| rule | Qwen3-VL | Qwen2-VL |
|---|---|---|
| always tight (head@0.25) | +7.9 [+0.0,+16.2] | +6.8 [−1.0,+14.7] |
| **2-rung: tight / no crop (median)** | **+5.8 [+0.5,+11.0]** ✔ | **+5.8 [+0.5,+11.0]** ✔ |
| 3-rung: tight / wide@0.5 / no crop (terciles) | +5.8 [+0.0,+12.0] | +4.7 [−1.1,+11.0] |
| 4-rung: + @0.35 (quartiles) | +6.3 [+0.0,+12.6] | +5.8 [−1.0,+12.6] |

Single-object median dispersion quantile 0.63 / 0.68; relational 0.32 / 0.31 — the map separates them
without seeing the question. **Intermediate window sizes add nothing** (wide crops on middling maps are
no better than no crop): the map can decide *whether* to crop, not *how large*. The mechanism that is
question-agnostic and clears pooled on both models is therefore the simplest one — *crop tight when
the model's own attention is concentrated, otherwise spend the budget on the whole image.* Whether
that is admissible under constraint 7 is the pending ruling (§18E).
