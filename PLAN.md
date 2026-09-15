# PLAN — 2026-09-11. Decided after five prior-art sweeps + advisor.

## A. THE REFRAME (decided, not optional)

**The paper is an EVALUATION-PROTOCOL paper, not a method paper.** Five sweeps each returned
"close but not identical"; that pattern is the finding. The method space is saturated — FAVE
(foveated encoding), ViCrop/RAP/ZoomEye/Q-Zoom (crop-and-re-encode), ViRGo (routing),
FastV/RoRA/D2Pruner (pruning), AttWarp (warping), LLMind (non-uniform sampling). A sixth variant
of attention-guided cropping is not a contribution.

**What nobody has is the axis.** Every one of those papers controls a *different* resource:

| family | resource held fixed | what stays at full cost |
|---|---|---|
| pruning | **LLM-side** tokens | the encoder |
| LLMind / Gizdov | **pixels** | the token count (all arms upsampled back to full res) |
| ViRGo | **seconds** | tokens uncontrolled |
| crop/zoom/FAVE | nothing | both |
| **ours** | **realized visual tokens**, gated at 10% spread | — |

Token count is the right axis: it is what KV-cache, context limits and serving cost scale with.
Pixels are a sensing cost; seconds are hardware-dependent.

**The claim:** *the field has been comparing on the wrong axis; here is the right one; here is what
changes when you use it.*

### The four things the protocol contributes
1. Budget matching on **realized `image_grid_thw`**, with a gate voiding contrasts above 10% spread.
2. **Four documented failure modes of the naive version** — analytic solve (19% advantage to the
   arm under test), per-image quantization (19% spread), step-function plateaus (82%),
   aspect-dependent unpadding (42%). **Each produced the SAME headline accuracy and a different
   admissibility verdict.** This table is the paper's argument.
3. **Two measured architectural constraints** any future budget-matched comparison must respect:
   Qwen3-VL's 64-token/image floor; LLaVA-NeXT's 1416-token floor with only two realizable settings.
4. **Applying it changes conclusions**: `alloc_random` at 39.8% *loses* to tiling by 15.2pp — which
   no area-matched or time-matched protocol would surface.

## B. BLOCKING before any writing — ✅ BOTH RESOLVED

1. ~~Reconcile ViRGo's RAP = 78.9% vs our `crop_only@300` = 93.7%~~ — **RESOLVED (Phase 29).**
   I suspected the readout: we take argmax over pooled {A,B,C,D} logits, ViRGo generates and
   exact-matches, and constrained argmax cannot emit an unparseable answer. **Measured on the same
   items, same images, same budgets, same model — the readout offset is EXACTLY ZERO:**

   | arm | logit argmax | generate + exact-match | offset | unparseable |
   |---|---|---|---|---|
   | `uniform@2400` | 70.6% | 70.6% | **+0.0pp** | 0 |
   | `crop_only@300` | 97.6% | 97.6% | **+0.0pp** | 0 |

   **FINAL n=191: uniform 75.4% / 75.4%, crop_only 93.7% / 93.7%, offset +0.0pp, 0 unparseable,
   and only 2 of 191 items disagree at all (they cancel).** The model always emits a parseable
   "(A) rubber"-style answer. **So our readout is
   not inflating anything — the gap is the ORACLE**, and that is now quantifiable rather than a
   hedge:

   > A published training-free proposer (ViRGo's RAP, 78.9%) captures roughly **31-44% of the
   > oracle gain** on this backbone and benchmark — 44% against ViRGo's own baseline (64.2%), 31%
   > against ours (70.6%). The remaining ~56-69% is what knowing the ground-truth box buys.

   **This is the number §9 should carry.** It converts "we use an oracle" from an apology into a
   measurement. Caveat to state: the comparison is cross-paper and budgets are not matched between
   their RAP and our arms, so it is a bound, not a controlled contrast.
2. ~~Check ViRGo Fig 3a against our §4.1~~ — **RESOLVED from the papers already read.**
   * **ViRGo Fig 3a bins by OBJECT SIZE** (V*Bench + GQA bins 0-7), not by question type. It does
     **not** cover our split.
   * **Q-CueGraph Table 12 DOES bin by V*Bench question type** (direct_attributes n=115,
     relative_position n=76) and reports: relative-position accuracy rises 0.737 -> 0.855 when the
     **top-2 regions are unioned**, because "relative-position questions need both referenced
     objects". So the *phenomenon* -- relational queries behave differently from attribute queries
     under region selection -- is **theirs**, and they even supply a fix (composition).

   **Verdict for §4.1:** cite Q-CueGraph for the phenomenon. Our addition is strictly the **budget
   axis**: they show a *composition* fix at unmatched area; we show that on relative-position
   questions **enough query-independent budget catches up on its own** (-3.9pp, CI spans zero at
   7957 tokens) while on direct_attributes it does not (-12.2pp [-19.1, -5.2]). Present §4.1 as
   "the boundary, re-measured on the token axis", not as a discovery.

   ⚠️ Note the tension worth stating honestly: their composition result implies relational queries
   need *more regions*, ours implies they need *more uniform budget*. These are compatible (both say
   a single tight crop is the wrong tool) but the paper should not imply we found the boundary.

## C. EXPERIMENTS

| # | what | status |
|---|---|---|
| **E-a** | **Phase 28 — architecture invariance of the exchange rate** (Qwen2-VL-7B, llava-onevision, LLaVA-NeXT). Now the **empirical core**: "this axis matters across four tokenizers" is what makes the protocol worth adopting. | **running** (downloads 25GB/32GB) |
| E-b | RQ-B on RePOPE — the 1/area scaling exponent. V*Bench cannot test it (all targets tiny); RePOPE spans 0.08–290 merged tokens. | queued |
| E-c | RQ-D harm test on HR-Bench `cross`. Pre-registered admissibility: uniform accuracy <85% or void. | queued, lower priority |

## D. CUTS (decided)

* **Drop the pruning/FastV section entirely** — not just the reimplementation. The citation-grounded
  version (110.5% of unpruned vs a 96–107% literature range) has a margin too thin to survive a
  reviewer citing RoRA's 106.9%, and it is not load-bearing once the paper is a protocol paper.
* **Do not claim a method.** §8 says plainly: we do not propose a better cropper.
* **No sixth literature sweep.**

## E. ORDER OF WORK

1. Let Phase 28 finish → it decides whether the protocol claim is architecture-general or a survey.
2. Resolve B1 and B2 (both are reads/analysis, no GPU).
3. Rewrite `PAPER_FLOW.md` as a protocol paper: protocol → failure modes → constraints → what
   changes → empirical core (exchange rate, no-crossing, architecture invariance) → boundaries.
4. E-b if time; E-c only if the paper needs a harm result.


---

## F. Measured architectural constraints — now THREE, and they differ sharply (protocol §2 material)

| model | per-image token floor | token cap | budget knob |
|---|---|---|---|
| Qwen3-VL-2B | **64** / image | none found in range | continuous |
| **Qwen2-VL-7B** | **4** / image | ~~**~6510**~~ **none — see correction** | continuous |
| LLaVA-NeXT-7B | — | 2242 | **two settings only**, aspect-dependent |

The floor differs by **16x** between two models of the same family. **Any budget-matched comparison
must measure these per model rather than assume them** — which is precisely the protocol claim.

### ⚠️ CORRECTION (same day): the "Qwen2-VL caps at ~6510 tokens" constraint was NOT REAL.
I recorded it because the probe's 4800 and 7800 requests both realized 6510. That was an artifact
of the **coarse ladder**, not the architecture: with the hybrid refinement in place the same model
realizes **4845** and **7776** for those requests. There is no cap.

This is worth keeping in the paper's failure-mode table, because it is the most dangerous kind of
protocol bug: **a measurement artifact that looks exactly like an architectural finding.** The naive
protocol did not just mis-match budgets — it would have had us publish a hardware constraint that
does not exist. It was caught only by re-running the gate after the fix.

## G. Phase 28 protocol bug found and fixed (a FOURTH failure mode for the Table-1 receipts)
The coarse geometric ladder is required for tiled models (it crosses token-count plateaus a
proportional solver cannot) but is far too coarse for a continuous one: on Qwen2-VL a requested
**300** landed on a rung realizing **391 (30% off)** — a spread the budget gate would void.
**Fix: hybrid — ladder to land on the right plateau, then proportional refinement from there.**
Step-function models keep the ladder's pick; continuous models converge. This is a fifth entry for
the failure-mode table and it was caught by the gate, not by inspection.

---

# DECISION (this session): the paper gets a method, and this is it

Three times the response to this draft has been "no methods." Two answers were given and both were
wrong: first a method *section* (which was really a protocol description), then "the protocol IS the
contribution." Reframing is not an answer to that objection. The paper needs an operation someone
can run. Recorded so this is not relitigated a fourth time.

## What was considered and rejected

**Region packing** (compose disjoint GT regions onto one canvas, discard inter-region dead space,
match realized tokens). Rejected -- see FINDINGS §4R. Not for lack of novelty: the narrow prior-art
check came back clean (AwaRes appends crops as separate images and lets token count grow; Visual
Scaffolds overlays annotations without cropping; neither holds realized tokens fixed). Rejected
because it is **structurally untestable on public benchmarks**:
  - packing is computed from oracle boxes, so on a RELATIONAL question the arrangement it
    reconstructs *is* the answer -- the preprocessing does the discrimination being measured;
  - V*Bench `direct_attributes` is 115/115 single-box (packing == cropping);
  - V*Bench `relative_position` is the only multi-box split (54 items) and is entirely relational;
  - HR-Bench ships no box annotations at all.
Constraint is public benchmarks only, so there is no venue. **Keep the 81% dead-space measurement**
(median fraction of the union bbox containing no evidence, n=54) -- it stands alone as a fact about
a published method's operation and needs no new arm.

## What is being built instead: Phase 30, the attention-guided budget-preserving allocator

Training-free. Two forward passes. Reads the localizer already inside the model.

    pass 1   render the whole image at the floor budget B0, read layer-2 attention from the final
             prompt position to the image tokens
    propose  bounding box of the top-25% image tokens by attention
    pass 2   re-render THAT REGION at the SAME realized budget B0, answer

The budget-preserving constraint is what makes it ours rather than the crop/zoom literature's: every
published proposer either adds tokens (AwaRes, DeepEyes, ZoomEye) or removes them (FastV, VisionZip).
None hold realized tokens fixed and move them.

**The signal is real and was measured before any of this** (Phase 22, same model). That phase
pre-registered that attention would select AGAINST small targets and **the prediction was refuted**:
attention retention exceeds the keep rate at every stratum and most strongly for the SMALLEST
objects (+34.8pp at median 0.26 tokens-on-object, decaying monotonically to +3.1pp at 104.9), with
random/uniform calibrated to the keep rate within 1.6pp. That refutation is what makes a proposer
plausible, and it is also a standalone finding: the ACL 2502.11501 result that attention pruning
fails on small objects is NOT explained by attention failing to mark them.

**The bar is pre-set, not chosen after the fact.** Phase 29 measured that a published proposer
captures 31-44% of the oracle placement gain. Oracle-capture fraction
`(attn_prop - uniform) / (oracle - uniform)` is the metric; >44% is a contribution, below it is a
reportable negative about how much allocation headroom is reachable without supervision.

**Running now: the cheap viability pre-check** (`phase30_attn_proposer.py`), one pass per item, no
downstream arms. It tests the one way the method dies that does not need the full 5-arm matrix: if
attention is diffuse, the box around the top-25% tokens is most of the image, the crop degenerates
to the uniform baseline, and nothing downstream can help. Phase 22 measured token RANK; it never
measured the AREA those tokens span, which is what a proposer must pay for. Decision rule fixed in
the script docstring before running.

# Correction to §4 forced this session: dynamic range is not comparable across architectures

See FINDINGS §4Q. Measured floors AND ceilings: Qwen2-VL 4->7776 (1944x), Qwen3-VL 64->7957 (124x),
onevision 1261->7329 (5.8x), LLaVA-NeXT 1416->2144 (1.5x). Consequences:
  - **onevision's "no crossing at 3.9x" was mis-read as the weak row.** 3.9x is ~67% of everything
    that architecture can express; Qwen's 26x sweeps cover ~2% and ~21% of theirs. Report the test
    multiple as a FRACTION of each model's dynamic range, never as a bare number.
  - **LLaVA-NeXT cannot run the exchange-rate experiment at all** -- its realizable ladder is a
    single rung. Not a null result about token scaling; the absence of an axis.
  - This also caused the host OOM-kill of the first LLaVA-NeXT leg (ladder chased an unreachable
    35568-token target up to a 54-megapixel resize). Fixed: ceiling measured like the floor,
    unreachable rungs skipped, ladder stops on two equal counts, 24 MP hard resize guard.

## Phase 30 success thresholds -- PRE-REGISTERED, computed before any proposer result exists

Denominator fixed from Phase 27 (Qwen3-VL-2B, V*Bench n=191), all at matched realized budget:

    uniform@300     56.5%   (294 tok)   <- baseline
    crop_only@300   93.7%   (300 tok)   <- oracle
    oracle gain     +37.2pp

    capture_fraction = (attn_prop@300 - 56.5) / 37.2

Bars, in increasing order of strictness:

| bar | what it is | accuracy the method must reach | capture |
|---|---|---|---|
| `uniform@300` | does allocation help at all | > 56.5% | > 0% |
| `uniform@600` | **compute-matched** -- the proposer costs 2 forward passes, so it must beat spending that same compute on plain resolution | > **66.0%** | > 25.5% |
| ViRGo RAP, low estimate | published training-free proposer | > 68.0% | > 31% |
| ViRGo RAP, high estimate | published training-free proposer | > **72.9%** | > **44%** |

The 44% bar is the binding one and it sits ABOVE the compute-matched control, so clearing it clears
everything. Reporting rule: quote capture fraction AND the compute-matched comparison every time.
A method that beats `uniform@300` but loses to `uniform@600` has not earned its second forward pass
and will be reported as a negative, not as a win.
