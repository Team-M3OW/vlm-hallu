# Results atlas — every phase classified, rated, and clustered

Built by reading all 78 phase files plus `FINDINGS.md` §14A–§14S for phases 82–96 (which have no
phase file). Phases 67–68 do not exist.

**Class:** `METHOD` (a thing you could run) · `FINDING` (a fact about VLMs) · `NEGATIVE` (a
pre-registered or well-controlled null) · `INFRA` · `RETRACTED` · `VOID`.

**Impact /10 is re-scored here, and it is not the phase file's "keep in paper" rating.** That
rating asked *how much of the paper does this deserve*. This one asks *would it change what someone
does*, after two things the old ratings predate: the replication standard (a claim on one model is
not a claim) and the prior-art survey (`TRACK_A_PRIOR_ART.md`). Where they differ the reason is in
the note. **†** marks a result anticipated by published work.

---

## 1. The atlas

### Phases 0–22 — problem definition

| # | class | result | impact |
|---|---|---|---|
| 0 | INFRA | Logit read-out, POPE↔COCO join, tokens **measured** from `image_grid_thw` | 2 |
| 1 | FINDING | Accuracy falls with object size on both architectures, and a *perfect* per-size-bin threshold buys almost nothing — not miscalibration | **7** |
| 2 | NEGATIVE | Position matters on Qwen3-VL, which does no destructive cropping → "it's LLaVA's preprocessing" is out | 4 |
| 3 | NEGATIVE | Centrality survives both controls | 2 |
| 4 | FINDING | **The dissociation**: on confidently-wrong items the model still boxes the object 43.2% of the time | **8** |
| 5 | NEGATIVE | No "detected then suppressed" — the final layer is the best layer. Re-confirmed 60 phases later after a bug said otherwise | 6 |
| 6 | FINDING | Attention is query-conditional (4.4× collapse when the question moves, answer held fixed); forcing attention onto the object does nothing; object- and random-targeted patching indistinguishable | 6 |
| 7 | FINDING | Oracle zoom flips 12.8% of confident denials; size-matched random crop flips **none** | 7 |
| 8 | INFRA | Factorial over zoom format; set up 9/10 | 2 |
| 9 | FINDING | Crop **alone** recovers 40.4% at FP 3.5% — Phase 7 had measured "resolution minus a distractor" | 6 |
| 10 | FINDING | Any full scene in the prompt costs ~25pp; image count and prompt format exonerated. Pre-registered "re-anchoring" hypothesis **refuted** | 5 |
| 11 | INFRA/FINDING | RePOPE audit: confident disagreement is a **7.2× enriched** label-error detector; findings get *stronger* on the clean pool | **7** |
| 12 | INFRA | fp16 cohort rebuild (bug #6) | 1 |
| 13 | FINDING | **The median confident-denial object is 0.4 merged tokens** — sub-token. Clean dose–response 0.4→32 tokens = 0%→72.2%. **Pointing does nothing**: red box 5.6%, coordinates in words **0.0%** | **9** |
| 14/15 | FINDING | Category-specific encoding is real but **modest** — 0.88 AUROC collapses to ~0.71 once the control is within-positives | 6 |
| 16 | NEGATIVE | Activation steering at L16, direction fit on disjoint items: null. First of the intervention family | 6 |
| 17 | **METHOD** | **The founding result**: query-placed allocation at 150 tokens beats uniform at 589. Beats a size-matched random region at **every** budget | **9** |
| 18 | NEGATIVE | CUB does not transfer (median 53.8 tokens) — **bounds** the claim to under-resolved targets | 6 |
| 19 | NEGATIVE | CUB parts still 20.5 tokens — venue closed permanently | 3 |
| 20 | METHOD | V\*Bench: 292 tokens beat uniform at 1176; on uniform-wrong items query recovers 86–93% vs random's 16–19% | **9** |
| 21 | FINDING | Monotone over a 400× size range on a second benchmark, FP unchanged. Negative half **not testable** (uniform saturates) and reported as such | 7 |
| 22 | NEGATIVE | Our prediction that pruning drops small targets is **refuted** — attention retention keeps them +20–35pp above chance. We had conflated attention *mass* with *rank* | 6 |

### Phases 23–46 — the localiser, coverage, and the gate

| # | class | result | impact |
|---|---|---|---|
| 23 | RETRACTED | AnyRes arm had a 17% token advantage over the arm under test | 2 |
| 24 | RETRACTED | `obj` probe AUROC 1.000 was an **ROI-selection artefact**; `last` curve circular | 4 |
| 25 | INFRA | LLaVA-NeXT gate tightened; several tables void | 3 |
| 26 | FINDING | A **use** failure, not a routing failure | 4 |
| 27 | FINDING | **The exchange rate is a lower bound, not an estimate: ≥26×.** No uniform rung ever reaches the crop arm | **9** |
| 28 | FINDING | Placement beats random on **4 architectures**; dynamic range spans 3 orders of magnitude; LLaVA-NeXT has **no axis** — reported as "NO AXIS" | 8 |
| 29 | NEGATIVE | The read-out is not inflating V\*Bench; the **oracle arm** is where the gap comes from | 5 |
| 30 | FINDING | **The localiser is in the map; the argmax is not.** GT cell in the top 3.4%, top-1 almost never the target. Sinks absorb 41–45% of selected mass | **8** |
| 31 | NEGATIVE | Pre-registered: a fixed always-crop policy **does not earn its second pass** (60.2%, fails the 66.0% bar, captures 10% of oracle) | **8** |
| 32 | METHOD | The conditional allocator clears the bar (68.6%). **Learned gates lost to a rule** — later explained: the two-pass comparison is an unsupervised coverage detector | 8 |
| 33 | FINDING | Confidence route transfers to 4K (+8.0pp CircularEval), text rule fails (−7.5pp). Control: attn−rand **+13.6pp** → the localiser transfers, allocation doesn't | 8 |
| 34 | FINDING | The sink is **columnar, not cornered** — last column 5.1×, first 2.9×, rows null. Transpose test rules out register tokens; replayed interior indices land **below** chance | **8** |
| 35 | RETRACTED | Region-count explanation superseded by coverage | 4 |
| 36 | FINDING | A perfect crop **wins** on relational questions (+19.7pp) → region-count is dead; the GT box is a **union**, 7.9× larger | 7 |
| 37 | FINDING | **Coverage is the mediator.** Dose–response crossing zero at ~25%; **half of all windows miss**, and on those allocation *costs* 15.6pp | **9** |
| 38 | FINDING | **Difficulty held fixed: 70.8pp** between covered and missed windows in one stratum; missed windows score **13.0%, below chance**. Interior optimum withdrawn | **8** |
| 39 | INFRA | Audit: capture fraction 83%→88% (the mismatched pairing would have **inverted** the conclusion); exogenous instrument inconclusive; interior optimum retracted | 6 |
| 40 | NEGATIVE + FINDING | Multi-window pooled effect **zero**; pre-registered subgroups **+38.9pp** where the second window added coverage, **+50.0pp** where it rescued zero coverage | 7 |
| 41 | FINDING | **The sink is a serialisation artefact, across 4 architectures.** Implicit row boundary → last column 3.4–4.5×; a learned `image_newline` → the sink sits **on the separator**, 2.0–2.3×. A prediction only this account makes | **9** |
| 42 | FINDING | Mechanism and method transfer to Qwen2-VL; gate AUROC **0.885**, better than on its home model | 8 |
| 43 | RETRACTED | "Rows carry no excess mass" is **not** universal — LLaVA bottom rows 2.1–3.0×. Bucketing bug put 59.8% of positions in a ≤5% bucket | 6 |
| 44 | FINDING | 8 of 8 of Qwen2-VL's best layers fall inside the transferred block — cross-architecture numbers stand as measured | 6 |
| 45 | METHOD | **`peak` predicts coverage at AUROC 0.788 for free** and beats a fitted multi-feature model. Caveat carried: random routing already earns +3.3pp, so `peak`'s own contribution is **+1.9pp** | 8 |
| 46 | NEGATIVE | The adaptive policy takes 4K from −3.1pp to **−0.5pp** — removes the failure, does not create a win | 6 |

### Phases 47–66 — matched-budget audit, interventions, the cliff

| # | class | result | impact |
|---|---|---|---|
| 47 | **FINDING** | **Prior art at matched budget: nobody beats the budget axis.** Zoom Eye has the top raw accuracy and loses to one uniform image of its own budget. Grounding −6.2pp with **median IoU 0.029** | **10** |
| 48 | FINDING | The deployed query token is best of seven; attention from the **noun naming the target** is near chance → localisation is not lexical lookup | 6 |
| 49 | VOID | Abandoned mid-run (prompt-level, not internal). Idea resurfaced correctly in 58 | 1 |
| 50 | NEGATIVE | Sink suppression null — and **biasing *any* 4.8% of image tokens moves ≤2pp while the same count of text tokens costs 17pp** | **8** |
| 51 | **NEGATIVE** | **The cleanest causal result in the project.** Oracle-targeted attention amplification gains +5.8pp; the pixel-space oracle crop gains +36.2pp at the same budget — **16%**. The oracle amplification set is **1 cell of 294** | **10** |
| 52 | NEGATIVE | A shared "allocate to evidence" direction exists (cos 0.53–0.75) and is **answer-irrelevant**: the transferable mean gives +0.0pp at every layer | 7 |
| 53 | **FINDING** | **The negative replicates at 4K, n=800, with significance.** Tree search −15.3pp against its own bar and −10.4pp vs uniform@1200 at 2.2× the tokens. And **our own headline does not replicate** (−0.2pp) | **10** |
| 54 | NEGATIVE | Composite halves the damage (−17.3 → −3.5pp), still negative; splitting a budget across two images costs on its own | 4 |
| 55 | MIXED | `peak` is **not** proposer-agnostic (it measures its own proposer). But gating recovers **+5.7pp for Zoom Eye while cutting its tokens 53%** | 7 |
| 56 | NEGATIVE | Coarse-to-fine at 4K: cells 237→83px, peak moves 489px, accuracy **+0.0pp [−3.0,+2.9]** | 6 |
| 57 | RETRACTED | Refuted **our own** scale-boundary mechanism: 4× finer cells made the margin *worse*. Coverage absorbs it | 7 |
| 58 | **METHOD** | **Multi-crop in one pass**: k windows at B₀/k, one pass, same budget. **multi4 − top1 = +7.9pp**; beats the budget axis by +9.5pp on single-region. Predicted its own failure on multi-region | **9** |
| 59 | MIXED | The **innovation** transfers to 4K (+8.0pp); the **win** does not (−7.3pp vs the axis). Correction: the boundary is the budget axis being unusually productive at 4K, not proposal precision | 8 |
| 60 | FINDING | The answer forms **abruptly at L22**; oracle advantage ≈0 before L21, +38.7pp at L22. (Also the origin of the normalisation bug) | 8 |
| 61 | RETRACTED | Every finding was the double-normalisation artefact. Kept as a cautionary record — it was coherent, mechanistically plausible, and survived its own falsification test | 5 |
| 62 | RETRACTED/FINDING | Composition claim withdrawn; the **allocation main effect +19.9pp survives** and agrees with 58 | 4 |
| 63 | NEGATIVE | Corrected: **no layer-decoding rule beats the model's own logits.** DoLa +0.0, DeCo −0.7, read-at-L24 +0.0 — with their hyper-parameters swept in their favour | **8** |
| 64 | NEGATIVE + INFRA | Component ablation null (≤2.4pp); **and it exposed the bug that invalidated three sections** (agreement 100% at L24, 82.2% at the final layer) | **8** |
| 65 | FINDING | **AttWarp scored against the budget axis it never ran: +0.6pp, break-even.** Its gains do not survive being charged for the second pass | **8** |
| 66 | **FINDING** | **Localisation is EMPTY, not mis-routed.** On localise-but-wrong items the answer is decodable at **no layer** (72.2% vs 68.1% for window-missed). The oracle crop fixes **94.4%**. Median tokens on target **0.14** | **10** |

### Phases 69–96 — the method arc

| # | class | result | impact |
|---|---|---|---|
| 69 | NEGATIVE | **No free pass-1 signal predicts the required budget**: 9.4% exact-rung vs a 35.6% majority baseline — worse than a constant | 7 |
| 70 | METHOD | Learned re-ranking head 39.3→**52.9%** coverage; position-only 3.7%, shuffled-label 2.6%. Captures **27.6%** of an 88.5% ceiling | 8 |
| 71 | METHOD | It converts: **+12.0pp** vs vanilla. Predicted +7.2pp before the run from the coverage exchange rate, measured +8.4pp vs the old proposer | **9** |
| 72 | MIXED | Re-ranking transfers zero-shot to HR-Bench (+4.9pp, +6.5pp CircularEval) with nothing refitted; the **allocator still loses** at 4K (−12.4pp) | 8 |
| 73 | FINDING | Layers disagree and the mean cancels them. ⚠ the **signed-contrast** reading was later rejected on 2 further tests | 6 † |
| 74 | REPLICATION | Averaging defect holds on Qwen2-VL (+8.4pp); **final-layer anti-correlation and signed contrast both fail** | 7 † |
| 75 | FINDING | **90% of visual tokens deletable at zero cost if you rank late; layer-2 ranking loses 22pp and is below random** | 8 † |
| 76 | NEGATIVE | **The vision tower is at or below chance** for question-conditioned localisation (1.6% / 1.0% / 0.5% vs 2.3% chance) while the LM gets 39.3% | **9** |
| 77 | NEGATIVE + causal | Contrastive decoding fails (+2.1pp, ceiling +2.6pp) but proves the evidence region is load-bearing: **3.32×**, −10.5pp when masked | **9** |
| 78 | FINDING | **W=0.25 beats the deployed 0.15** (71.7 vs 68.6); +11.0pp sizer headroom survives the monotonicity check; **no predictor found** | 7 |
| 79 | **FINDING** | **Cropping restores answer formation at L21.** Flat for 20 layers, then +15.3pp. The failing arms jump at L2 and never improve — their 28-layer max **equals** their final answer | **10** |
| 80 | REPLICATION | +10.5pp vs vanilla on Qwen2-VL ✔; vs the argmax it replaces **demoted**; **vs equal compute, 0 of 2** | **9** |
| 81 | NEGATIVE | Rank-only multi-crop **−3.1pp** against a predicted +5.5pp; top-4 still beats four random crops by +27.7pp. Prompting confound excluded (81b) | 6 |
| 82 | INFRA + FINDING | LLaVA extraction unblocked → the read-out defect holds on **3 of 4** across two families; absolute localisation 39.3→12.6% | 8 † |
| 83 | REPLICATION | Pruning result replicates (+11.5pp); **the mechanism prediction fails** — Qwen2-VL ranks worse early and is hurt *less* | 8 † |
| 84 | REPLICATION | L21 mechanism on Qwen2-VL: **+23.8 covered / +0.0 missed** | **10** |
| 85 | INFRA | Figures from real inference: attention, masks, crop pixels, real answers | 7 |
| 87 | FINDING | It is a **threshold, not layer 2** — all of L0–L14 catastrophic, L16+ free | 7 † |
| 88 | FINDING | **Attention quality (L17–19) and answer formation (L21) are dissociated**, ρ=0.297; attention top-1 is *higher* before the answer forms | **9** |
| 89 | VOID | Hung 36 min at 0% CPU — root disk 100% full | 0 |
| 90 | FINDING | **W=0.25 survives held-out selection** (71.7→71.6%, folds pick it 99/100) → +15.0pp vs vanilla, **+7.7pp [−0.6,+16.0]** vs the bar | **9** |
| 91 | FINDING | All layers beat any contiguous block or quality-selected subset — **poor layers are negative evidence, not noise** | 6 |
| 92 | VOID | Premised on the head reading L16–26. It already reads all 28; the "gain" was against a restriction I invented | 0 |
| 93 | FINDING | The collapse is **larger on general VQA**: layer-2 is 15.0pp (POPE) / 25.5pp (MMBench) below random. Refutes the V\*Bench-flattery objection | 8 † |
| 94 | FINDING | Early attention is **no better than a coin-flip even for coarse culling** (+1.0pp vs a random first cut) | 6 † |
| 95 | FINDING | **Attention is question-blind until ~50% of depth** — divergence 0.001 → 0.14, a 13× jump at L14→L16 on both models. Label-free | 8 † |
| 96 | REPLICATION | The locator finds the **region, not the layer**: divergence rises L13–14, damage recovers L14–16; L14 does not clear zero, L16 does. Flat floor → one transition → plateau | 7 |

---

## 2. The streams

### Stream A — **The encoding cliff: what is not in the tokens** ★ strongest
`1 · 4 · 7 · 9 · 13 · 14/15 · 16 · 26 · 27 · 50 · 51 · 52 · 63 · 64 · 66 · 76 · 77 · 79 · 84`

One claim with an unusual amount of support: **below ~0.25 merged tokens of target extent the
evidence is not encoded, and nothing downstream can recover it.** The cliff is architecture-invariant
(2 models, identical oracle values); the oracle crop is flat across it at 95.8/100%, so items below
it are not *harder*; and **seven independent internal interventions are null** — steering, sink
suppression, attention amplification, residual injection, DoLa/DeCo, component ablation, contrastive
decoding aimed with ground truth.

The two results that make it more than a list of failures: **51** (aim perfectly and you get 16% of
what cropping gets, because the oracle amplification set is one cell of 294) and **77** (the region
is verifiably load-bearing — 3.32×, −10.5pp when masked — and still cannot be converted). Plus **79 +
84**: supplying the pixels restores an answer-formation step at L21 on both models, and only where
the crop actually delivered.

**Paper:** *Vision-language models cannot answer what they never encoded.* Method-shaped only if the
detector in §3 works; otherwise a findings-and-negatives paper with a real mechanism section.
**Impact 9.** **Unsurveyed** — this is the survey `TRACK_B.md` §4 asks for.

### Stream B — **Matched-budget evaluation: nobody beats the budget axis** ★ most under-used
`17 · 20 · 27 · 28 · 31 · 45 · 46 · 47 · 53 · 54 · 55 · 56 · 57 · 59 · 65 · 72 · 80 · 90`

Charge every method for the tokens it actually spends and the literature's crop/warp/search family
breaks even or loses: **Zoom Eye** has the highest raw accuracy in the table and loses to one uniform
image of its own budget; **grounding** loses 6.2pp with median IoU 0.029; **AttWarp** is +0.6pp
against a bar its own paper never ran; **tree search at 4K** is −15.3pp against its own bar at 2.2×
the tokens, n=800. And the same discipline is applied to us: our method is **0 of 2** against the bar,
and Phase 47's "ours is the only positive margin" **does not hold at 4K**.

This is the cluster with the sharpest teeth, the largest n, the most models, and the least prior art
risk — because it is a claim about *how the field evaluates*, not about a mechanism someone else
might have found first. It also carries the exchange rate (**≥26×**, a lower bound set by the
architecture) and the "NO AXIS" result, which tells you the audit is not even runnable on some
checkpoints.

**Paper:** *What do those tokens buy? A matched-budget audit of visual token allocation.*
**Impact 9.** Needs a survey for existing matched-compute critiques, but the four method families are
already measured.

### Stream C — **Serialisation sinks** ★ cleanest self-contained result
`30 · 34 · 41 · 43 · 50`

Four architectures, two families. The sink is **columnar, not cornered** (last column 5.1×, rows
null), a **transpose test** rules out register tokens, replayed interior indices land *below* chance,
and the account makes a prediction nothing else does: models that splice a learned `image_newline`
at each row end put the sink **on the separator itself** (2.0–2.3×), which is not a corner, not a
pixel and not image content. Then the honest coda: **suppressing it buys nothing** (50), and biasing
*any* 4.8% of image tokens moves ≤2pp while the same count of text tokens costs 17pp.

Small, complete, replicated on four models, with a risky prediction confirmed and a causal null
attached. **Impact 8**, and the most likely of all the streams to survive a survey intact as a
short paper — though sinks are heavily studied in LLMs, so check first.

### Stream D — **Coverage: the variable that decides the sign** 
`35 · 36 · 37 · 38 · 39 · 40 · 42 · 45 · 55 · 58 · 59 · 81`

Not *whether* you reallocate but *what fraction of the question's evidence set the window covers*.
Dose–response crossing zero at ~25% coverage; **half of all windows miss**; **70.8pp** between covered
and missed within a single difficulty stratum, with missed windows at **13.0% — below chance**. It
absorbed two superseded explanations (region count, the scale boundary), predicted multi-crop's win
*and* its multi-region failure in advance, and predicted the multi-window subgroup effects before the
run. It also got one prediction wrong and said so: rank-only multi-crop, predicted +5.5pp, measured
−3.1pp — the coverage model has no term for distractor cost.

Best used as the mechanism section of Stream B, not as its own paper. **Impact 8 as a component, 6
standalone.**

### Stream E — **Read depth: where attention is read** †
`29 · 44 · 48 · 61 · 62 · 73 · 74 · 75 · 82 · 83 · 87 · 93 · 94 · 95 · 96`

Averaging across depth dilutes the read-out (4 models); layer-2 pruning is below random (2 models,
3 benchmarks, larger on POPE/MMBench than on V\*Bench); attention is question-blind until ~50% of
depth. **Largely anticipated** — see `TRACK_A_PRIOR_ART.md`. What survives: the read-depth /
prune-depth decomposition (prune point pinned at K=2, only the read source moves), the reconciliation
with Wang et al.'s opposite-looking result, the **failed mechanism prediction** (ranking quality does
not predict pruning damage), and the **selection-bias caution** (in-sample best-of-28 41.9% → 36.1%
out-of-fold, below the block mean it appeared to beat).

**Paper:** short/workshop, retitled away from "the wrong depth". **Impact 5 after the survey**, 8
before it.

### Stream F — **Where the answer forms**
`5 · 60 · 64 · 66 · 79 · 84 · 88 · 95`

The answer forms abruptly at **L21** and not gradually; the failing arms peak at L2 and their maximum
over all 28 layers **equals** their final answer; attention quality peaks at **L17–19** and correlates
with decodability at only **ρ=0.297**, so attention is *best* in the layers *before* the answer forms.
Replicated on two models. This is the mechanism section for Stream A and the interpretability half of
the required flow. **Impact 8 as a component.**

### Stream G — **Methodological credibility**
`11 · 23 · 24 · 25 · 29 · 39 · 42 · 43 · 57 · 61 · 62 · 63 · 64 · 65 · 81b · 92 · 96`

Retractions found by our own controls, not by reviewers: a double-normalisation bug that invalidated
three sections and was caught because two implementations agreed on L24 for 100% of items and on the
final layer for 82.2%; an ROI-selection artefact behind an AUROC of 1.000; a wrong-image join
affecting 26.5% of rows; a capture-fraction mispairing that would have **inverted** a conclusion; a
phase built on a restriction that did not exist; a mechanism of ours refuted by our own ablation.
Plus RePOPE: confident model disagreement is a **7.2× enriched** label-error detector, and the
findings strengthen on the clean pool.

Not a paper. An appendix that is worth more than most appendices, and the reason the numbers in
A–F can be believed.

---

## 3. What I would build

**Stream B as the paper, Stream A as its mechanism, Stream D as the bridge, Stream G as the
appendix.** That uses 60+ phases, needs no new capability, has the largest n and the most models,
and its central claim — *charge methods for their tokens and the wins evaporate* — is one we have
measured on four published method families and cannot be scooped out of, because it is a statement
about evaluation practice.

Stream A's detector (`does the model know it cannot see`) is the one live route to a *method*
inside this material. Stream C is the best short paper. Stream E is a workshop paper. Stream F is a
section, not a paper.
