# Next experiments — planned against the results as they now stand (2026-09-09)

Supersedes the forward-looking half of PAPER_PLAN.md v2 (written before Phases 17–21).

## What the results actually are (so the plan follows from them)
The mechanism (uniform, query-independent allocation) is a **known design fact**, not a finding.
Four things are ours:
  R1  causal quantification at matched budget vs a **size-matched random-placement** control (+30pp)
  R2  irreversibility — 4 controlled nulls, no inference-time intervention recovers it
  R3  a quantified domain — binds below ~2 tokens on target, gone above ~4
  R4  an efficiency consequence — 292 query-placed tokens beat 1176 uniform (+15.7pp)
Every experiment below must strengthen one of these or bound it.

---

## E1. Cross-architecture + patch-size invariance   [~4h, models cached]  ← DO FIRST
One run, two analyses. V*Bench, same four arms, matched **realized** tokens per model.
Models: `Qwen2-VL-7B-Instruct`, `InternVL2-8B`, `llava-onevision-qwen2-7b-ov-hf`.

  A. **Replication** of R1 — the question every reviewer asks first.
  B. **Patch-size invariance** (the mechanistic payoff, near-zero marginal cost):
     does the failure threshold sit at a CONSTANT value in *token* units across models with
     different patch sizes / tiling / AnyRes, while sitting at very DIFFERENT *pixel* sizes?
     - agree in tokens, differ in pixels  => tokenization is the boundary, not optics. R1/R3 become
       architecture-level claims rather than one-model observations.
     - agree in PIXELS instead           => "allocation, not resolution" is WRONG. Must know this
       before submission, not after.

**Gate:** realized-token accounting differs per model (Qwen dynamic-resolution, InternVL tiling,
llava-onevision AnyRes). `fit_to_budget` calibrates on measured `image_grid_thw`, so it transfers in
principle — verify per model on one item before trusting the match, exactly as bug #18 required.
**Memory:** use `lean_loader.py`, one model at a time (build_items peaks at 17GB — bug #19).

## E2. Why attention-based pruning fails on small objects   [~2h]
Supplies the mechanism for a **published, unexplained** result: the ACL'25 pruning paper reports
methods "catastrophically fail on fine-grained localization" and that random/uniform beats
attention-guided selection, without explaining why. Phase 6c already has the answer (attention to a
denied object looks like attention to an object nobody asked about).

Measure **target-token RETENTION**, not just downstream accuracy — retention is the mechanism and is
the number the ACL paper is missing.
  arms: attention-top-k selection vs uniform/random selection, matched retained-token budget
  strata: by `tokens_on_object` (reuse Phase 21 strata)
  prediction: attention-top-k drops the target's tokens far above chance in the sub-token stratum;
              uniform preserves them — which is exactly why uniform wins.
Buys: R2 extended into a live efficiency literature, and an audience beyond hallucination.

## E3. Allocation-split sweep   [~1h]
Phase 21 found 25/75 (crop/scene) **beat** 50/50 in the scarcest stratum (80.0% vs 74.2%) — a knob
we never tuned. Sweep crop fraction {10, 25, 50, 75}% × the five size strata.
Buys: turns R3 from "there is a domain" into an **allocation policy curve**; likely free accuracy.

## E4. Remove the oracle — training-free proposer   [~3h]
Arms at matched budget: oracle region / self-grounded region / attention-peak region / random.
Report as **"% of the oracle gain captured"**, not as a competing method score — V*/SEAL own
propose-then-crop, and the matched budget is the only thing that differentiates us.
Expected ceiling: Phase 4 measured grounding IoU>0.5 at 29–43%, so ~40% of oracle gain is the honest
prior, not a failure.
Buys: converts R4 from an upper bound into something deployable.

## E5. The harm test that Phase 21 could not run   [~2h]
Phase 21's large-object strata were at **ceiling** (uniform 97–99%), so the pre-registered
prediction — allocation halves scene resolution to fund a crop it does not need, and should HURT —
is **untested, not refuted**. Needs hard questions about large targets: HR-Bench `cross`-instance
category is the best candidate on disk.
Buys: closes the one place where a reviewer can say the domain claim is unfalsified.

## E6. HR-Bench as a second regime   [~3h, optional]
1600 items, already downloaded, MCQ apparatus identical to Phase 20 so it is near-free to run.
Buys: n, and a second independent high-resolution regime. Do only after E1–E3.

---

## Explicitly not doing
- **Fine-tuning / LoRA** — the nulls are scoped to *training-free* interventions; that is defensible
  today. A LoRA negative invites "train longer" and has no natural stopping point. State as a
  limitation instead.
- **Frontier-model (GPT-4V/Gemini) replication** — no API keys or SDKs on this box (verified). Also
  scope-limited even with a key: closed APIs do not expose visual token counts, so the *mechanism*
  cannot be run there — only the phenomenon.
- **Video / human baselines / counting / OCR-TextVQA** — each needs a new benchmark, a new readout
  mapping AND a new false-positive control (the CUB lesson: two experiments minimum apiece).
  TextVQA is the only one worth revisiting if E1 lands clean and time remains.
- **A design-rule predictor** fitted to our own curves and "validated" against published numbers we
  did not reproduce.

## Standing method invariants (all experiments)
Match on **realized** `image_grid_thw`, never intended · always carry a **size-matched random**
arm · read verdicts on **discrimination**, never raw recovery · every recovery number carries its
false-positive rate · pre-register the null criterion before running.
