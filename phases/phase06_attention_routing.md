# Phase 6 (6b–6e) — Is attention routed to the queried object, and can redirecting it fix anything?

## 1. Research question
Three linked questions: (6c) does attention to an object depend on the *question* or on the
*answer*? (6d) is the failure just too few decoding steps? (6e) if we force attention onto the
object, does the answer change?

## 2. Finding and contribution (plain English)
**6c:** attention is genuinely query-conditional, not answer-driven. Holding the answer fixed at
"yes" and moving only the question, attention on an object collapses ~4.4×. And when the model is
about to confidently deny an object, its attention to that object looks like attention to an object
nobody asked about.

**6d:** the "not enough decode steps" story is refuted — content-free filler tokens beat a genuine
image-grounded chain of thought, so the effect is template shift, not re-perception.

**6e:** forcing attention onto the object does nothing, and — importantly — object-targeted and
random-targeted patching are indistinguishable, so the tool wasn't discriminating. We report this as
*"we could not build a clean test"*, not as "attention causality refuted".

## 3. Numbers that changed
- 6c, polarity held fixed at YES: attention enrichment on X falls **3.145 → 0.713 (~4.4×)** when the
  question moves from X to Z.
- 6c target-modulation index: denial **[0.93, 1.28]** (includes 1.0) vs correct **[3.65, 5.39]**.
- 6d: median `P(yes)` 0.00019 → CoT 0.056 / **filler 0.119**; filler beats CoT on **108/141**,
  p=8.4e-11.
- 6e: only **6/141** items improved; object-target vs random-target CIs span zero.

## 4. Keep in paper: 6/10
6c's tautology-killer is a genuinely good control and worth keeping. 6d and 6e are eliminations —
one paragraph. **Important caveat to carry:** Phase 22 later showed 6c measured attention *mass*,
not *rank*, and the stronger reading ("attention fails to find the object") was withdrawn.

## 5. Experiment, step by step
1. **6c:** fix the image and the object-X token set. Vary only the question: ask about X, about an
   absent Y, and about a present-other Z. Measure attention mass on X's tokens from the final
   position, averaged over heads and the last 4 layers, normalised by token count (enrichment).
2. Compare within the *correct* group alone, with the answer polarity held at YES, to kill the
   "attention follows the answer" objection.
3. **6d:** force extra decode steps two ways — a real image-grounded CoT, and **token-count-matched
   content-free filler**. If filler matches or beats CoT, the effect is not re-perception.
4. **6e:** monkeypatch attention at the final position to give object tokens a target share F.
   Run with the object as target and with a **random region** as target. If the two are
   indistinguishable, the intervention is not discriminating and the test is inconclusive.
