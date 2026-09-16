# Every paper this evidence can support

Each entry: the claim, the findings that carry it, how many models it holds on, what is missing, and
whether it is main-track, short, or not a paper. **A finding can only headline one of these** —
where two papers want the same result, the one that owns it is named.

Replication rule in force: a claim on one model is not a claim. Survey status matters as much as
n — Track A's spine was lost to prior art after five phases were built on it.

---

## P1 — "Vision-language models cannot answer what they never encoded" ★ strongest
**Main track. 2 models on every load-bearing number. Unsurveyed.**

Below ~0.25 merged tokens of target extent the evidence is absent from the representation, and no
operation on the residual stream can put it back.

| carries it | number | models |
|---|---|---|
| Ph 13 | the median confident-denial object is **0.4 merged tokens** | 1 |
| Ph 66 | on localise-but-wrong items the answer is decodable at **no layer** (72.2% vs 68.1% for window-missed); oracle crop fixes **94.4%** | 1 |
| §13C | the cliff at 0.15–0.25 tokens, **oracle flat across it at 95.8 / 100.0%** | **2** |
| Ph 51 | aim attention perfectly: **+5.8pp** vs the crop's **+36.2pp** — 16%. Oracle amplification set = **1 cell of 294** | 1 |
| Ph 77 | the region is load-bearing (**3.32×**, −10.5pp masked) and still unconvertible (ceiling +2.6pp) | 1 |
| Ph 16, 50, 52, 63, 64 | five more internal nulls, each well-controlled | 1 each |
| Ph 79 + 84 | supplying pixels restores answer formation at **L21**, only where the crop delivered (+29.7/−7.9 and +23.8/+0.0) | **2** |
| Ph 1 | not miscalibration — a perfect per-size-bin threshold buys almost nothing | **2** |

**Why it works as a paper:** seven independent interventions failing is not a list of failures, it is
a bound. Phase 51 and 77 are the two that make it airtight — one aims perfectly and gets a sixth of
the benefit, the other proves the target is causally live and still cannot convert it.

**Missing:** the interventions are single-model. At minimum 51 and 77 need a second architecture,
and that is a rerun of existing scripts. **And it has never been surveyed** — that comes first.

**Owns:** 13, 66, 51, 77, 52, 76, 79, 84, the cliff.

---

## P2 — "What do those tokens buy? A matched-budget audit" ★ least scoopable
**Main track or benchmark track. 4 published method families, 2 benchmarks, n=800 at 4K. 1 model.**

Charge every allocation method for the tokens it actually spends and the family's wins evaporate.

| carries it | number |
|---|---|
| Ph 47 | **Zoom Eye** has the highest raw accuracy in the table and loses to one uniform image of its own budget; grounding **−6.2pp** at median IoU **0.029** |
| Ph 53 | 4K, n=800: tree search **−15.3pp** against its own bar, **−10.4pp** vs uniform@1200 at **2.2×** the tokens — worst arm per token |
| Ph 65 | **AttWarp** scored against a bar its own paper never runs: **+0.6pp**, break-even |
| Ph 27 | the exchange rate is a **lower bound set by the architecture: ≥26×** — no uniform rung ever reaches the crop arm |
| Ph 31 | our own fixed policy fails the same bar (60.2% vs 66.0%) |
| Ph 80, 90 | and our learned one: **0 of 2**, best case **+7.7pp [−0.6,+16.0]** |
| Ph 28 | on LLaVA-NeXT there is **no axis to move along** — the audit is not runnable at all |
| Ph 55 | the positive deliverable: gating recovers **+5.7pp for Zoom Eye while cutting its tokens 53%** |

**Why it works:** it is a claim about *evaluation practice*, not about a mechanism someone else may
have found first — the one kind of claim here that cannot be scooped out from under us. It also
holds us to the same standard, in public, which is the part reviewers reward.

**Missing:** one model. A second architecture on phases 47/53 would make it much harder to dismiss;
that is the single highest-value rerun in the project. Also needs a survey for existing
matched-compute critiques.

**Owns:** 47, 53, 65, 27, 31, 55, 28.

---

## P3 — "The VLM attention sink is a serialisation artefact" ★ cleanest
**Short paper / findings track. 4 architectures, 2 families. Self-contained.**

| carries it | number | models |
|---|---|---|
| Ph 34 | **columnar, not cornered** — last column **5.1×**, first **2.9×**, rows null (0.9 / 0.8×) | 1 |
| Ph 34 | transpose test (14×21 vs 21×14): identical (row,col) positions, only 4/8 shared absolute indices → **not register tokens** | 1 |
| Ph 34 | replaying modal-grid sink indices into the interior gives **1.1% vs 3.3% chance** — below chance | 1 |
| Ph 41 | **the risky prediction**: models with a learned `image_newline` put the sink **on the separator** (2.0–2.3×); implicit row boundaries put it in the last column (3.4–4.5×) | **4** |
| Ph 43 | the correction that keeps it honest — LLaVA bottom rows *are* enriched 2.1–3.0×, so "rows carry no excess" is not universal | 2 |
| Ph 50 | and it does not matter: suppressing the sink is null, and biasing **any** 4.8% of image tokens moves ≤2pp while the same count of **text** tokens costs **17pp** | 1 |

**Why it works:** a separator embedding is not a corner, not a pixel and not image content. A 2-D
spatial account predicts nothing there; the serialisation account predicts exactly it. That is a
falsifiable prediction confirmed across two tokenisation schemes, with a causal null attached.

**Missing:** nothing structural. **Survey first** — attention sinks are heavily worked in LLMs
(StreamingLLM and successors) and there is VLM sink literature.

**Owns:** 34, 41, 43, 50.

---

## P4 — "Why attention-guided cropping helps half the time"
**Main track only if fused with P2. Standalone it is thin. 2 models (partial).**

| carries it | number |
|---|---|
| Ph 37 | dose–response in coverage crossing zero at **~25%**; **half of all windows miss**; missed windows cost **−15.6pp**, full cover **+37.1pp** |
| Ph 38 | difficulty held fixed: **70.8pp** between covered and missed in one stratum; missed windows score **13.0%, below chance** |
| Ph 36 | a perfect crop **wins** on relational questions (+19.7pp) — kills the region-count story; the GT box is a **union**, 7.9× larger |
| Ph 40 | pre-registered subgroups: **+38.9pp** where a second window added coverage, **+50.0pp** where it rescued zero coverage, **−4.0pp** where it added nothing; **pooled effect zero** |
| Ph 42 | replicates on Qwen2-VL, gate AUROC **0.885** |
| Ph 57 | absorbs our own refuted scale-boundary mechanism |
| Ph 81 | and it gets one wrong, on the record: rank-only multi-crop predicted **+5.5pp**, measured **−3.1pp** — no term for distractor cost |

**Verdict:** this is the *mechanism section of P2*, not a paper. It explains why every audited method
breaks even: they place a window that misses half the time, and a miss costs almost as much as a hit
gains.

**Owns:** 36, 37, 38, 40, 42, 81 — as P2's §mechanism.

---

## P5 — "Where the answer forms, and where attention is best, are not the same place"
**Interpretability workshop, or a section of P1. 2 models on both headline numbers.**

| carries it | number | models |
|---|---|---|
| Ph 79 / 84 | the answer forms **abruptly at L21**; 20 flat layers then +15.3pp; the failing arms jump at **L2** and their 28-layer max **equals** their final answer | **2** |
| Ph 88 | attention quality peaks **L17–19**, the answer forms **L21**, ρ = **+0.297**; attention top-1 is *higher* before the answer forms (30.1% vs 24.8%) | 1 |
| Ph 95 | attention is **question-blind until ~50% of depth** — divergence 0.001 → 0.14 | **2** |
| Ph 60 / 64 | and the retraction that makes it credible: a double-normalisation bug manufactured the opposite conclusion, caught by our own ablation | 1 |

**Verdict:** the honest home is P1's mechanism section. It answers the reviewer objection that the
project had two entangled findings — and it confirmed the objection rather than deflecting it.

---

## P6 — "Read depth is not prune depth" †
**Workshop. 2–4 models. Spine anticipated — see `TRACK_A_PRIOR_ART.md`.**

What survives the survey: phases 75/83 pin the prune point at FastV's `K=2` and move **only the read
source**, which no surveyed paper does — **+21.4pp / +11.5pp** from the read alone. That reconciles
Wang et al. (CVPR'26) finding attention ≈ random, since they read *and* prune deep. Plus two things
that are ours outright: the **failed mechanism prediction** (Qwen2-VL ranks worse at layer 2 and is
hurt *less*, −13.1 vs −21.9) and the **selection-bias caution** (in-sample best-of-28 41.9% → 36.1%
out-of-fold, below the block mean it appeared to beat). Phase 96 adds the shape: flat floor, one
transition, plateau — and the locator finds the region but **not** the layer.

**This is what the current 19-page draft is built on.** It is a workshop paper, retitled.

---

## P7 — "Show the model k windows, not one"
**Not a paper yet. 1 model, 2 benchmarks.**

Ph 58: k windows at B₀/k in **one** forward pass, no selection step — **multi4 − top1 = +7.9pp** at
identical passes and tokens, and **+9.5pp over the budget axis** on single-region questions. Ph 59:
the innovation transfers to 4K (**+8.0pp**), the win does not (**−7.3pp** vs the axis).

The mechanism predicted both its win and its multi-region failure in advance, which is the best thing
about it. But it has the same disease as DCR — it beats the axis only in a stratum — and it is one
model. **Two reruns from a paper; not one today.**

---

## P8 — "A learned re-ranking head for crop placement" (DCR)
**Rejected as a method. Retained as evidence inside P1/P4.**

39.3→52.9% coverage and **+12.0 / +10.5pp** over the unmodified model on 2 models, +28.3 / +20.9pp
over random placement, and it pays exactly below the cliff (+13.8pp under 0.15 tokens, nothing
above). But **0 of 2** against spending the same compute on a bigger image, best case **−0.6** on the
lower bound, and the "depth-contrast" mechanism in its name failed twice outside Qwen3-VL.

Becomes a paper only if the per-item window sizer closes one point (Ph 78's **+11.0pp** of stable
headroom, predictor untried).

---

## P9 — "Where does 'where to look' come from?"
**Short paper. Coherent, but thin on models.**

Ph 76: every read of the vision tower's 24 layers is **at or below the 2.3% chance rate** (best 1.6%,
learned 1.0%, mean 0.5%) while the LM on the same items reaches 39.3%. Ph 48: attention from **the
noun naming the target** is near chance (`gt_pct` 0.367 vs 0.500) — localisation is not lexical
lookup. Ph 6c: attention is genuinely query-conditional (4.4× collapse when only the question moves).
Ph 95: and it does not exist until half-way up the stack.

One claim: **question-conditioned localisation is constructed by the language model, late, and is not
present in the image encoder or in the noun.** 76 and 48 are **one model each** — that is the gap.
Note this also bounds ram's Vision-CLS selector result, which is the natural merge point.

---

## P10 — Not papers
- **Ph 11 (RePOPE audit)** — confident model disagreement is a **7.2× enriched** label-error detector.
  Useful, but RePOPE is published; ours is a confirmation and a tool. → Methods section.
- **Ph 4 (the dissociation)** — the founding observation, 43.2%. Fully explained by P1. → P1's §1.
- **Stream G (retractions)** — the double-normalisation bug, the ROI artefact, the wrong-image join,
  the capture-fraction mispairing, phase 92's invented restriction. → P1/P2 appendix, and worth more
  than most appendices.

---

## Ranking

| | paper | venue | blocking gap |
|---|---|---|---|
| 1 | **P1** encoding cliff | main | survey; second model on 51/77 |
| 2 | **P2** matched-budget audit | main / benchmark | second model on 47/53 |
| 3 | **P3** serialisation sink | short | survey only |
| 4 | **P9** where "where to look" comes from | short | second model on 76/48 |
| 5 | **P6** read vs prune depth | workshop | nothing — it is what it is |
| — | P4, P5 | sections of P2 and P1 | — |
| — | P7, P8 | not yet | one point / two reruns |

**Two full papers, two short ones, one workshop.** P1 and P2 share almost no findings, so both can
be written from this material without cannibalising each other — P1 owns the cliff and the
interventions, P2 owns the audit and the exchange rate, and P4 bridges them as P2's mechanism.
