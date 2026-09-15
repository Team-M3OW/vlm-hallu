# Research plan — RQs, experiments, and falsification conditions
**Written 2026-09-09**, after Phases 17–22. Supersedes the forward half of PAPER_PLAN.md v2 and
NEXT_EXPERIMENTS.md. Every RQ below states what a NEGATIVE result would mean, before the run.

---

## 0. THE SCOPE DECISION (yours to make — everything downstream depends on it)

**(a) Mechanism paper, one model, oracle stated.** Submittable after RQ2b + the writing. E4 optional.
**(b) Mechanism + generality paper.** Requires RQ3 (cross-arch/patch-invariance) AND RQ4 (proposer).

This is not an ordering detail: it decides whether E4 (~3h, and the arm that overlaps V*/SEAL most)
is required or skippable, and whether HR-Bench is worth 3h. **Plan below assumes (b); mark items
`[b-only]` to drop for (a).**

---

## 1. Claims already established (do not re-run)

| id | claim | evidence |
|---|---|---|
| C1 | At matched **realized** budget, query-conditional allocation beats uniform **and beats size-matched random placement** | V*Bench +30.4pp [+23.0,+37.7] n=191; POPE +24–35pp |
| C2 | The cost is **irreversible at inference time** | coords 0.0%, box overlay 5.6%, attention patching null, steering null vs norm-matched random |
| C3 | It has a **domain**: binds <~2 tokens on target, gone >~4 | Phase 21, n=600, stratified by design, FP flat at 7% |
| C4 | **Efficiency consequence**: 292 query-placed tokens beat 1176 uniform (+15.7pp) | Phase 20 |
| C5 | **Boundary (negative)**: does not explain fine-grained over-acceptance | CUB: 70.7% false-accept at 53.8 tok; part-allocation ΔAUROC +0.01, CI spans 0 |
| C6 | **Methodological**: confidence-selected cohorts are 7.2× enriched for label errors | RePOPE join, n=4053; cleaning doubled the measured effect |

### 1.1 AUDIT RESULT — Phase 6c is demoted (do this before drafting)
Phase 6c measures attention **MASS** enrichment. Phase 22 confirms mass is sub-proportional
(0.49–0.95×), so the mass claim stands. But FINDINGS §4C used 6c to support *"there is nothing at
that location to attend to"* — and Phase 22 shows attention **RANK** marks the target ABOVE chance
(+20 to +35pp) precisely in the sub-token strata.

⇒ **6c may no longer carry the "nothing to attend to" claim.** That claim is carried by the probe
(0.71 AUROC, Phase 15) and the intervention nulls (C2). 6c becomes a supporting correlate:
*attention mass on a denied object is not enriched*. Every remaining use of 6c must state
**mass or rank**. Rewrite FINDINGS §3.1's "strongest mechanistic result" label accordingly.

---

## 2. Open research questions

### RQ1 — Is uniform, query-independent allocation a causal *bottleneck* or merely a suboptimality?
**Status: ANSWERED** (C1 + C2). The random-placement control makes it causal; the four nulls make
it a bottleneck rather than a tunable. No further work.

### RQ2b — Does allocation ever HURT? *(the weakest link in the domain claim)*
C3's large-object end is **unfalsified**, not confirmed: Phase 21's large strata sat at ceiling
(uniform 97–99%), so a negative delta was undetectable. A reviewer who notices treats all of C3 as
soft.
- **Experiment E5.** HR-Bench `cross`-instance category (multi-region questions, already on disk),
  arms = uniform / alloc_query / alloc_random at matched realized budget.
- **ADMISSIBILITY CONDITION (state before running):** the result is only informative if uniform
  accuracy is **well below ceiling** (target <85%). If uniform saturates again, the experiment is
  void and must be reported as still-unfalsified rather than as a null.
- **Prediction:** alloc_query < uniform on multi-region questions (it funds one crop it doesn't
  need, halving scene resolution).
- **If refuted** (allocation never hurts anywhere): C3 becomes "benefit decays to zero" rather than
  "there is a crossing" — weaker but still honest, and the abstract must not say "crossing".
- Cost ~2h.

### RQ3 — Is the perceptual boundary set by TOKENIZATION or by OPTICS? `[b-only]`
The decisive mechanistic question, and it subsumes cross-architecture replication (which is a
robustness check, not an RQ, and should not be listed separately).
- **Experiment E1.** `Qwen2-VL-7B`, `InternVL2-8B`, `llava-onevision-7B` on V*Bench, same four arms,
  matched **realized** tokens per model. Two analyses from one run:
  (A) does C1 replicate; (B) does the failure threshold sit at a constant value in **token** units
  while sitting at very different **pixel** sizes?
- **Prediction:** agree in tokens, differ in pixels ⇒ tokenization is the boundary. C1/C3 become
  architecture-level claims.
- **IF IT COMES BACK AGREEING IN PIXELS: the paper's central framing is WRONG.** "Allocation, not
  resolution" would be false, and the paper becomes a resolution paper with an allocation-shaped
  intervention — a materially weaker and different claim. Write that version's outline before
  running, so the result is not rationalised after the fact.
- **Gates:** verify realized-token accounting per model on one item first (Qwen dynamic-resolution
  vs InternVL tiling vs AnyRes) — bug #18. Use `lean_loader.py`, one model at a time — bug #19.
- Cost ~4h. **DO FIRST.**

### RQ4 — How much of the oracle gain survives without ground-truth supervision? `[b-only]`
Converts C4 from an upper bound into something deployable, and is the only thing that answers
"is this a method or an analysis?"
- **Experiment E4.** Arms at matched budget: oracle / self-grounded / attention-peak / random.
- **Report as "% of oracle gain captured"**, never as a competing method score — V*/SEAL own
  propose-then-crop, and matched budget is the sole differentiator.
- **Honest prior:** Phase 4 measured grounding IoU>0.5 at 29–43%, so ~40% of oracle gain is the
  expected outcome. That is a result, not a failure — say so in advance.
- **NOTE (new, from Phase 22):** attention-peak is now a *better-motivated* proposer than we thought
  — attention RANK marks small targets +20–35pp above chance. This arm may beat the grounding arm.
- Cost ~3h.

### RQ5 — What is the optimal allocation policy?
Phase 21 found 25/75 (crop/scene) **beat** 50/50 in the scarcest stratum (80.0% vs 74.2%) — a knob
never tuned, i.e. every number in C1/C4 is a **lower bound**.
- **Experiment E3.** Sweep crop fraction {10, 25, 50, 75}% × the five size strata.
- **Prediction:** optimum shifts toward the crop as the target shrinks — an allocation policy curve.
- **If flat:** the split does not matter, C1/C4 stand as-is, and one sentence retires the question.
- Cost ~1h. Cheapest item here; do it alongside E1.

### RQ6 — Does the account generalize beyond object presence?
**Status: ANSWERED, NEGATIVE** (C5). Keep as the scope paragraph. Do not invest further.

---

## 3. Sequencing

1. **E1 (RQ3)** — first, because it can invalidate the framing. 4h.
2. **E3 (RQ5)** — 1h, run alongside; makes C1/C4 lower bounds explicit.
3. **E5 (RQ2b)** — 2h, closes the unfalsified end of the domain claim.
4. **E4 (RQ4)** `[b-only]` — 3h, only if scope (b).
5. HR-Bench full run — optional, only if time remains after 1–4.

## 4. Not doing (with reasons)
- **Fine-tuning/LoRA** — nulls are scoped to *training-free*; a LoRA negative invites "train longer".
  State as a limitation.
- **Frontier-model replication** — no API keys/SDKs on this box (verified); and closed APIs do not
  expose visual token counts, so the mechanism cannot be run there regardless.
- **Video / human baselines / counting / OCR-TextVQA** — each needs a new benchmark + readout mapping
  + FP control (CUB lesson: two experiments minimum apiece).
- **Explaining the ACL pruning result** — attempted in Phase 22, **refuted**, withdrawn.
- **A design-rule predictor** fitted to our curves and "validated" on numbers we did not reproduce.

## 5. Method invariants (non-negotiable, all experiments)
Match on **realized** `image_grid_thw`, never intended · always carry a **size-matched random** arm ·
read verdicts on **discrimination**, never raw recovery · every recovery number carries its FP rate ·
**pre-register the null criterion before running** · state mass-vs-rank whenever citing attention.

---

## 6. RQ0 — THE REFRAME (added 2026-09-09; higher value than everything in §2)

### The hole this closes
Our `uniform` arm is a **downsampled whole image**. Production VLMs do NOT do that — LLaVA-NeXT,
InternVL2 and Qwen-VL use **AnyRes / dynamic tiling**: a large budget, tiled uniformly over space,
**chosen before the question**. So (i) a reviewer can call our uniform arm a strawman, and (ii) the
real target is bigger than we have been claiming.

### The claim it unlocks
> The high-resolution pathway in **every modern VLM** allocates its token budget uniformly over
> space and independently of the question, forfeiting ~4× of it.

A claim about a near-universal design choice, not about one 2B model on POPE.

### Experiment E0 — add an `anyres_tiling@B` arm
Tile the image as LLaVA-NeXT / InternVL2 do, at **matched realized budget**, alongside `uniform@B`
and `alloc_query@B`. One extra arm in the existing Phase 20 harness — no new benchmark, no new
readout mapping. Folds into the E1 cross-architecture run.

- **If query-placed beats properly-implemented tiling at equal tokens:** the claim becomes about
  production systems, and the strawman objection dies in the same table.
- **If tiling closes most of the gap:** our "uniform" baseline WAS a strawman, C1/C4's effect sizes
  shrink to whatever survives against tiling, and the paper must be rewritten around that smaller
  number. This is the outcome that would hurt, and it must be reported.
- **Gate 1:** for InternVL2 / llava-onevision the *native* path already is the tiled path — verify
  the arm is a genuine reimplementation at our budget, not accidentally the model default. Log
  realized tokens and confirm they differ from native.
- **Gate 2:** Qwen3-VL uses dynamic resolution, not fixed tiling, so the AnyRes comparison is most
  meaningful on the two models that ship tiling. Do not generalize from Qwen alone.

### Writing blocker
**Do not write "modern VLMs allocate uniformly" until this arm exists.** Today that sentence is an
inference about how those systems work, not a measurement — and it is the first thing a reviewer who
builds those systems will check.

### Honest ceiling (recorded so it is not re-litigated)
- With E0 + RQ3: a strong analysis paper about how VLMs are built.
- Without them: a careful result about one model.
- **Not reachable with current assets:** a method paper beating V*/SEAL on their leaderboard. The
  oracle does too much work; a proposer at ~40% of oracle gain will not top a tuned visual-search
  system. Do not pursue.

### Revised sequencing
**E0 + E1 together (RQ0 + RQ3)** → E3 (RQ5, 1h) → E5 (RQ2b, 2h) → E4 (RQ4) `[b-only]`.

---

## 7. Sequencing note (user, 2026-09-09): mechanistic interpretability comes AFTER headlines

> "we may divert to a mechanistic interpretability route after we have a few incremental headlines"

Recorded as the intended order: **establish the behavioural headlines first (E0/RQ0, E1/RQ3), then
go mechanistic.** Not a pivot away from the allocation result -- the mech-interp turn is the natural
explanation layer for **C2 (irreversibility at inference time)**, which is currently supported only
by four controlled nulls and has NO mechanism attached.

### What is already built for that turn (do not re-derive)
| Phase | Result | What it constrains mechanistically |
|---|---|---|
| 14 | linear probe recovers the target above chance from hidden states | the information **is** encoded -- kills "capacity floor / nothing there" |
| 22 | attention **rank** marks the target +20-35pp above chance at L2/4/8 | attention **finds** the region -- kills "nothing to attend to" |
| 16 | steering at L16 gives no real recovery (`rand_last` exposed it as bias shift) | a single-layer additive intervention does **not** reopen the channel |

Read together these already say something sharp and non-obvious: **the target is encoded, attention
locates it, and yet the answer readout cannot use it -- and one-layer steering will not fix that.**
That is the mechanism question worth a section: *where between "attention ranks it" and "the readout
ignores it" does the signal die?*

### Entry points when we get there (not started, not committed)
- **Layer sweep of the probe**, not just L16 -- if probe AUROC rises then collapses, that localizes
  the loss to a layer band instead of asserting it.
- **Probe on the visual-token positions vs the answer position** -- distinguishes "never routed out
  of the visual tokens" from "routed but overwritten".
- **Attention knockout**: ablate the target's visual tokens and measure how much the answer moves.
  If the answer barely moves while the probe says the information is present, the readout is not
  reading those positions at all -- which is a mechanism for C2, not a restatement of it.

### Precondition (keep the discipline that has been the project's asset)
Every one of the above needs a **bias-shift control** of the `rand_last` kind before any number is
reported. Phase 16 produced a 55.6% "recovery" headline that was pure bias shift; a probe or
knockout result without its matched random control is not a finding.

---

## 8. REVISED after the 2026-09-09 prior-art check -- the mechanism is the paper now

`LITERATURE_GAPS.md` records the hit in full. Summary: **Q-CueGraph (2608.04452)** already has the
area-matched anti-region control AND a shuffled-question control on **V*Bench, n=191, MC accuracy**
-- our exact setup -- and **RUTA (2608.04132)** already has the random-region control on Qwen3-VL.
C1 is a replication.

### What that does to the sequencing
| was | now | why |
|---|---|---|
| E0 (AnyRes) = the reframe that raises the ceiling | **E0 = still worth finishing, but it is a TABLE, not a paper** | its "generalize the claim" half is gone; only the matched-tokens-vs-matched-area methods upgrade remains. Running (Phase 23). |
| E1 (cross-arch / patch invariance) next | **E1 CANCELLED for now** | it was scoped to generalize C1, and C1 is no longer what needs generalizing. 4 GPU-hours on a demoted claim. |
| mech interp "later" | **mech interp IS the next run** | C2 is the only surviving claim two concurrent papers cannot scoop, because neither touches internals. |

### Next run: Phase 24, the probe layer sweep (`phase24_probe_layer_sweep.py`, written, queued)
Converts C2 from *an assertion backed by four nulls at scattered layers* into a **localized claim**.
All 29 layer outputs x 4 positions (`obj`, `rand`, `last`, `last_blank`), cross-validated, and --
the thing Phase 14 could not do -- **split by whether the model answers the item correctly**. C2
lives on the failures; on items it already gets right, "the information is present" is not news.

Pre-registered readings are in the analyzer, including the two that would hurt:
* `obj` and `last` both above controls while the answer is wrong => the readout **has** it and
  ignores it, and "irreversible" is the wrong word -- rewrite C2.
* `obj` never beats `rand` on the wrong-answer cohort => **contradicts Phase 14**; the capacity-floor
  reading returns and must be reconciled, not buried.

### Citation boundary (do not get this wrong)
Q-CueGraph §4.4 already reports the **behavioural** shadow of C2 -- 14% of crops that contain the
gold answer still produce a wrong answer, and a 43% "reader failure" class. **Cite them for the
phenomenon; claim only the mechanism.** Their evidence is present *in the pixels*; ours is present
*in the hidden states and the attention ranking*. Claiming the phenomenon would be a scoop violation.

---

## 9. Phase 24's `obj` probe has a confound; the C2 follow-up is a Phase-15-style design

Observed in the first layer rows (2026-09-09): `obj` AUROC is **0.960 at L0, the EMBEDDING layer**,
before any transformer block has run. That cannot mean the model has decided the object is present.

**Cause.** For positives, `obj` pools the GT bbox tokens; for negatives the object is absent, so
there is no box and the code pools a **random region**. The probe therefore partly separates
"patch embeddings of a real object" from "patch embeddings of a random patch of scene" -- a generic
object-vs-background distinction sitting in the raw visual features. Same failure family as bug #19:
a real number that does not license the claim it appears to license.

**Consequences, both already applied:**
1. The caveat is printed by `phase24_analyze.py` itself, so it travels with the numbers.
2. **The `obj` curve is not the C2 evidence.** The answer-relevant curve is `last` vs `last_blank`:
   does presence information ever reach the position that emits yes/no? That contrast is clean --
   both blocks are the same position, same prompt, differing only in whether the image is real.

**The follow-up that WOULD be query-conditional** (not started, do not run before Phase 25 lands):
extend **Phase 15's** within-positive design across depth -- probe the **queried** object's tokens
against a **size-matched OTHER category's** tokens in the SAME image. Both are real objects, so the
object-vs-background confound cancels exactly, and what remains is whether the representation is
conditioned on the question. That is the probe C2 actually needs, and Phase 15 already has
`build_full_lookup()` and the category-matching machinery to build it.

**Cost note:** the layer sweep costs ~35 min of CPU per cohort in analysis alone. A query-conditional
version should sample layers (e.g. every other one) unless a specific band is already implicated.

---

## 10. THE AMBITIOUS PROGRAM (2026-09-09) -- new experiments, not re-analysis

Framing the whole paper is aiming at:
> **Attention is not the bottleneck; allocation is. And allocation costs O(1/target area) if you
> don't know where to look, and O(1) if you do.**

The unnamed number that motivates it, from Phase 25 at matched realized tokens:
`uniform`@1464 -> 47.6%; `anyres`@2144 -> 55.0% (+47% budget, +7.4pp); `alloc_query`@2144 -> 88.5%
(same tokens, placed by the question, +33.5pp). A spatial prior is worth ~4.5x its weight in
compute on that one comparison. **The ratio is interesting; the asymptote and the scaling are the
results.**

### RQ-A -- the asymptote. Does query-independent budget EVER catch up?  [Phase 27, RUNNING]
`crop_only@300` held FIXED; `uniform@B` swept over a MEASURED realizable ladder --
140 / 300 / 588 / 1200 / 2352 / 4800 / 8112 realized tokens, a **58x range**. Both arms single
image, so unlike Phase 23 the sweep is format-clean at every point.
* crossing at B* => report the exchange rate B*/300 with a CI; this is an efficiency claim.
* **no crossing to 8112 tokens** => *query-independent budget does not substitute for placement in
  this regime*, stated as a BOUND with B_max named -- never as a claim about all budgets.

### RQ-B -- the scaling law. THE surprising result, if it holds.  [Phase 27 data, no extra GPU]
Predicted: the uniform budget needed to resolve a target scales as **1/area**, while the placed
budget needed is **constant**. Phase 27 stores `bbox_area_frac` per item, so the crossing budget can
be fit per size stratum and the exponent read off. If the exponent is near -1, the headline is a
scaling law rather than a benchmark delta.
* Falsified if crossing budget is flat in area, or if no stratum crosses (then only RQ-A's bound).

### RQ-C -- is the exchange rate ARCHITECTURE-INVARIANT?  [new runs, ~30GB to hdd2]
Measure the same exchange rate on tokenizers that disagree about everything: Qwen3-VL-2B (16px
patch, dynamic resolution), LLaVA-NeXT-7B (336px tiles, AnyRes), Qwen2-VL-7B (14px patch),
llava-onevision (384px tiles). **If the rate is roughly constant across them, it is a property of
the TASK, not of any model** -- which is the strongest form this paper can take. If it varies
wildly, the honest result is that it is architecture-specific and the claim narrows to a survey.
Disk: hdd2 has ~290GB free after LLaVA-NeXT.

### RQ-D -- where does placement HURT? (the boundary that makes it science)  [new run]
**Admissibility condition, pre-registered:** uniform accuracy must be <85% on the chosen split, or
the test is void (no headroom to detect harm). Target: HR-Bench `cross`, whose questions need
evidence from multiple distant regions. A clean regime where query-placed allocation is WORSE is
worth more to the paper than another benchmark where it wins.

### RQ-E -- the mechanism for RQ-D: relational breakdown  [new run]
V*Bench already hints at it: `direct_attributes` +30.5pp vs `relative_position` +14.5pp. Dedicated
test: benefit vs the NUMBER of regions a question requires (1, 2, 3+). Predicted monotone decay,
possibly crossing zero. This converts "placement helps" into "placement helps iff the evidence is
localizable", which is the honest scope and matches Q-CueGraph's operating-regime account -- cite
theirs, extend with the budget axis they do not have.

### NOT doing (and why)
* **"Does the model know it can't see?"** Selective prediction from hidden states is one
  cohort-selection mistake away from being 4L again; two retractions today came from that family.
* Per-size decision-threshold retuning -- **already tested and refuted**, see below.

### Refuted cheaply today (report it; it kills an obvious reviewer objection in one line)
"Isn't this just miscalibration?" No. Retuning the yes/no threshold per size stratum buys **0.2pp**
at the smallest stratum (93.7% -> 93.9%), because negatives sit at **median p_yes = 0.0000** and the
distributions barely overlap. The failure is not threshold placement. That also sharpens 4N:
"graded evidence survives to the readout and is not used" is the defensible reading.
