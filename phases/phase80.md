# Phase 80 — Does the method replicate on a second architecture?

**Classification: MIXED — the headline survives, two supporting claims are rejected**

## 1. Research question

The standard adopted 2026-09-16 is that every claim must hold on more than one model or be
rejected. DCR's end-task result was single-backbone. Does it replicate on Qwen2-VL-7B?

A rejection rule was fixed before the run: *the head must beat the argmax proposer with a CI clear
of zero, or DCR is a Qwen3-VL-2B artifact and the method section is cut.*

## 2. Finding, in simple English

**The method's headline survives. Two things I had been claiming alongside it do not.**

Against the plain model, DCR wins on both architectures and the confidence intervals clear zero on
both: **+12.0pp** on Qwen3-VL, **+10.5pp** on Qwen2-VL. That is the claim the paper keeps.

Against the proposer it replaces — the ordinary attention argmax — it wins by +8.4pp on Qwen3 but
only **+4.7pp on Qwen2, with a confidence interval spanning zero**. Same direction, half the size,
significance gone. By the rule written before the run, that claim is demoted rather than argued
around.

And the claim that hurts most: **DCR does not beat simply spending the same two forward passes on a
bigger image.** +4.7pp on Qwen3, +3.1pp on Qwen2 — neither clears zero. I had been carrying this in
METHOD.md on the strength of a single stratum of a single benchmark. It is withdrawn.

The pipeline is sound, so none of this is a measurement artifact: on the 70 items where the head and
the argmax happened to choose the same cell, the two arms score **identically**, as they must.

What this leaves is honest and smaller than it was this morning: *DCR beats doing nothing, by about
ten points, on two architectures — and we have not shown it beats the trivial alternative of using
more tokens.*

## 3. Numbers that changed

| arm | Qwen2-VL-7B |
|---|---|
| uniform@300 (vanilla) | 50.8% |
| uniform@600 (2-pass bar) | 58.1% |
| argmax@0.15 (incumbent) | 56.5% |
| **head@0.15 (DCR)** | **61.3%** |
| rand@0.15 | 40.3% |
| oracle@0.15 | 91.1% |

| contrast | Qwen3-VL | Qwen2-VL | verdict |
|---|---|---|---|
| vs vanilla | +12.0 [+4.2,+19.9] ✔ | **+10.5 [+2.6,+18.3] ✔** | **survives** |
| vs random placement | +28.3 ✔ | **+20.9 [+11.5,+30.4] ✔** | **survives** |
| vs argmax proposer | +8.4 [+2.6,+14.7] ✔ | **+4.7 [−1.0,+11.0] ✗** | demoted |
| vs 2-pass bar | +4.7 [−3.7,+13.1] ✗ | +3.1 [−5.2,+11.0] ✗ | **withdrawn** |

Proposal coverage: 39.3→52.9% (Qwen3) · 35.1→44.0% (Qwen2). Internal control +0.0pp [+0.0,+0.0],
n=70. All arms within 2.0% of their token target.

## 4. Keep in paper: **9/10**

Not because it is a good result — because it is the result that makes the other numbers
trustworthy. A method section claiming +16.2pp with an unreplicated mechanism would not survive
review; one claiming a replicated +10–12pp with the budget comparison explicitly withdrawn will.

## 5. Experiment, stepwise

1. **Reuse the Qwen3 feature construction verbatim**, pointed at Qwen2-VL's attention maps, so the
   two models' heads cannot drift apart in implementation.
2. **Train and freeze proposals out-of-fold first (80a)**, on CPU. The GPU script then does no
   fitting and cannot leak.
3. **Write the rejection rule into the file before running it.** This is what turned a tempting
   "+4.7pp, same direction as Qwen3" into a demotion instead of a claim.
4. **Six arms in one code path** at matched, measured budgets; bf16 with finite-logit assertions.
5. **Check the internal control before reading anything else.** Identical-cell items must split 0.0
   — they do, so a null cannot be blamed on the pipeline.
6. **Apply the rule as written.** The temptation here is real: +4.7pp with a lower bound of −1.0 is
   *probably* a true positive at larger n. It is still not a claim, and the ledger records it as
   provisional rather than quietly keeping it.
