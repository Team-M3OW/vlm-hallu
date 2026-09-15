# Literature sweep (BFS) and gap analysis — 2026-09-07

Breadth-first across eight adjacent areas, checking each of our findings against prior art.
**Conclusion up front: most of our individual findings are already covered. Four gaps survive, and
the paper has to be built on those, not on the components.**

---

## 1. What is ALREADY TAKEN (our findings that are NOT novel)

### 1.1 ❌ Sub-token / small-object perception limits — TAKEN
- **The Last Visible Pixel: Probing Fine-Scale Perception in VLMs** (arXiv 2606.07861) —
  FineSightBench, controlled scales 4–48px, separates perception from reasoning. Finds
  **perception saturates ~12 pixels**; sub-patch (0.5P) and patch-sized (1P) objects are the hard
  regime; ~4P needed for robust discrimination.
- **How Much Information Can a Vision Token Hold? A Scaling Law for Recognition Limits in VLMs**
  (arXiv 2602.02539) — distinguishes a *visual legibility limit* from an *information capacity
  limit*.

⇒ Our token-budget dose-response (0.4 tok→0%, 1.8→28%, 4.1→50%, 32→72%) is **largely a
re-derivation**. Their 4P-for-robustness matches our 4.1-tokens-for-50% almost exactly. Cite as
concordant replication; **do not present as a finding.**

### 1.2 ❌ "The probe knows but the readout fails" — HEAVILY TAKEN
- **The Count Is There, but Misaligned** (arXiv 2607.09544) — VLM counting: count is internally
  encoded but *weakly coupled to the subspace driving verbalized output*. Nearly our exact framing.
- **Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express**
  (arXiv 2605.25891) — the yes/no readout gap, explicitly.
- **When Do Internal Probes Beat Reading the Answer?** (arXiv 2609.04582) — miscalibrated readouts,
  behaviour-concealed knowledge.
- LVLM probes reaching **>95%** on counterfactual object questions vs much lower generation accuracy.

⇒ **§4D is not a contribution.** It is background. See Gap 2 for the part that survives.

### 1.3 ❌ Label errors found via model disagreement — TAKEN as a method
- **Are LLMs Better than Reported?** (arXiv 2410.18889) — flags items where models confidently
  disagree with the label; ensemble + confidence.
- **UQ-LED** (arXiv 2405.09602), confident learning, NoiseRank; **RePOPE** (arXiv 2504.15707) already
  relabelled POPE.

⇒ Our 7.2× number is an *instance* of a known method on an already-relabelled benchmark. See Gap 4
for the framing that survives.

### 1.4 ❌ Query-aware / text-guided visual token selection — CROWDED
FastV, **FlashVLM** (2512.20561), VisPruner, **CARES: Context-Aware Resolution Selector**
(2510.19496), **OccamToken** (2605.29657), **Balancing Saliency and Coverage** (2603.14892),
TokenFLEX, AsymVLM, LRCP; video: Q-Frame, QTSplus, VideoRouter, LongVU.

⇒ "Spend tokens where the query points" **exists**. Crucially, though, it is framed as
**efficiency** — *how few tokens can we keep before accuracy drops* — not as a causal account of a
perception failure. See Gap 1.

### 1.5 ❌ Attention / activation steering for hallucination — VERY CROWDED
RUDDER, DMAS, Revis, DAMRO, EAZY, Nullu, OPERA, VCD, PASTA, contrastive neuron steering, bi-causal
steering. ⇒ Our nulls (6e, 16) are not contributions in themselves; they are *evidence for* Gap 2.

### 1.6 ❌ Zoom / crop / visual search — PRIOR ART (already known, FINDINGS §0 constraint #3)
V*/SEAL, CropVLM, Chain-of-Spot, Zoom Eye, Visual CoT, DualFocus.

---

## 2. GAPS THAT SURVIVE — build the paper on these

### ✅ Gap 1 — Matched-budget *placement* as a causal test, not an efficiency result
The compression literature matches **average retained tokens across methods** and asks how much
accuracy is *retained*. Nobody appears to run the inverted experiment we ran in Phase 17:

> Hold the **realized** token budget fixed. Compare **query-placed** vs a **size-matched
> random-placed** allocation of the *same* budget. Measure *recovery of confident hallucinations*.

The `alloc_random` control is what makes it causal rather than a method comparison, and it is
absent from the efficiency papers (their baseline is uniform/other-method, not random-placement).
Our result — **+24 to +35pp over random placement at every budget**, and **150 query-placed tokens
beating 589 uniform tokens** — is a statement about *why* the failure happens, not about throughput.
Closest prior: *Balancing Saliency and Coverage* (matches average tokens, but compares allocation
policies, not placement-vs-random).

### ✅ Gap 2 — A capacity-limited regime where the readout gap is UNFIXABLE
The probe literature (§1.2) reports encoded-but-unverbalized with probes at **>95%**, and the
implied remedy is steering or probe-based readout. We find the opposite regime:

> When the object is sub-token, the encoding is **real but faint** (0.71 AUROC, same-image
> size-matched control) — and **every** inference-time intervention fails: pointing 0.0%, box
> overlay 5.6%, attention patching null, activation steering null against norm-matched random.

⇒ **The readout-gap story has a capacity floor beneath which it stops being a readout problem.**
That refinement is not in the literature, and our four nulls are exactly the evidence for it. This
reframes our nulls from "failed attempts" into the contribution.

### ✅ Gap 3 — Failure taxonomy indexed by token budget (with a working/not-working intervention)
Two opposite-polarity failures, separated by an intervention that fixes one and not the other:
- **confident denial** of under-resolved objects — capacity-limited; allocation recovers it
  (Phase 17)
- **fine-grained over-acceptance** — 70.7% false-accept on same-genus CUB species at **53.8 tokens
  on object**; allocation to diagnostic parts does **nothing** (Phase 19, ΔAUROC ≈ +0.01, CI spans 0)

Hallucination is treated as one phenomenon in nearly all of §1.5. Demonstrating a budget-indexed
split, with a causal intervention that dissociates them, is open.

### ✅ Gap 4 — Selection-induced label contamination in *interpretability cohorts*
RePOPE quantified **aggregate** POPE label noise (9.3% of positives). The open point is the
*selection* effect:

> Every mechanistic hallucination paper builds its cohort by **selecting on model confidence**.
> That selection enriches label errors **7.2×**: our confident-denial cohort is **30.9% mislabelled
> + 27.2% ambiguous = 58% not clean**, while an unselected sample sits at base rate.

This is a methodological warning aimed at the **interpretability literature**, not at POPE — and it
is directly actionable (we show the effect roughly *doubles* after cleaning). Not found in the sweep.

---

## 3. Untested and high-value (not yet ours)
- **Does Gap 1 survive a *learned* query-conditional allocator?** Everything is oracle-region.
- **Does it survive scale?** Everything is Qwen3-VL-2B. Cached: Qwen2-VL-7B, InternVL2-8B,
  llava-onevision-7B.
- **Denser-object regimes** (aerial, dense scenes, documents) where sub-token objects are the norm
  rather than 1.6% — the fix for the thin-base problem that CUB failed to solve.

---

## 4. Recommended framing

Lead with **Gap 1** (the causal matched-budget result) as the contribution; use **Gap 2** to convert
the nulls into the mechanism; use **Gap 3** as the scope boundary that makes the claim credible; use
**Gap 4** as the methodological section that justifies why our numbers differ from published ones.
Cite §1.1 and §1.2 as concordant prior work rather than pretending to find them.

---

# PRIOR-ART HIT (2026-09-09): C1 is substantially anticipated. Read this before writing C1 as novel.

I checked prior art before E0's result lands, per the standing rule. Two Aug-2026 papers overlap
hard with C1. **This is bad news and it demotes C1 from "the headline" to "a replication with a
tighter budget control."**

## 1. Q-CueGraph (arXiv 2608.04452, 2026-08-05) -- the closer hit
Same benchmark, same n, same readout. `Qwen2.5-VL-7B` frozen reader, **V*Bench all 191 items, MC
letter accuracy**. Their Table 9:

| condition | acc | area |
|---|---|---|
| full_image_highres | 0.696 | 1.000 |
| center_crop | 0.524 | 0.360 |
| random_crop | 0.435 | 0.122 |
| **anti_crop** (area-matched wrong region) | **0.393** | 0.134 |
| **shuffled_question** (same policy, another item's question) | **0.618** | 0.397 |
| qcg_visual_top1 | 0.791 | 0.134 |
| qcg_visual_top2 | **0.833** | 0.192 |

They already have **both** of our deciders: an **area-matched anti-region** control (0.393 vs 0.791
at the same 0.134 area) and a **shuffled-question** control that isolates query-conditionality
(0.618, and it got *more* area than the treatment -- 40% vs 13% -- which makes their result stronger,
not weaker). They also report a per-benchmark **operating-regime account** (`Delta_q` query
discrimination, `Delta_loc` location selectivity, resolution bottleneck) across six benchmarks --
which is most of what we called "a failure taxonomy indexed by budget."

## 2. RUTA (arXiv 2608.04132, 2026-08-04)
Query-conditioned localizer (Grounding DINO) -> crop -> resize -> encode, on LLaVA-NeXT-7B and
**Qwen3-VL-8B**. Their Fig 6(a) compares localizers against **random-region and full-image
controls** and finds query-conditioned localization consistently wins. Training-based.

## WHAT IS LEFT -- and it is narrower than yesterday
The single thing both papers explicitly decline to do is **match the visual token budget**:
* Q-CueGraph matches **image-area fraction**, and reports rendered-pixel ratios separately. Area is
  not tokens under dynamic resolution -- Phase 17 bug #18 is exactly this confound, where an
  analytic budget solve handed the favoured arm a 19% token advantage.
* RUTA states it outright: *"Because the variants produce different average token rates, the results
  demonstrate robustness to localizer choice rather than a strictly rate-matched ranking."* Their
  rate-matched ablation, Fig 6(b), varies **how many** tokens per sample, never **where** spatially.

So the surviving contributions, honestly ranked:

| # | Claim | Status after this check |
|---|---|---|
| 1 | **AnyRes tiling at matched realized budget** (E0/Phase 23) | **Open.** Neither paper runs it. Still the highest-value experiment. |
| 2 | **Tokens vs pixels** -- patch-size invariance across architectures (E1/RQ3) | **Open.** Nobody has tested whether the effect indexes on tokens or on pixels. |
| 3 | **C2: irreversibility at inference** -- probe/steering/patching nulls, and a mechanism for them | **Open, and now MORE valuable.** Neither paper touches internals. This is why the mech-interp turn is worth taking. |
| 4 | **C6: selection-induced cohort contamination** (confidence-selected cohorts 7.2x enriched for label errors) | **Open.** Untouched by both. |
| 5 | ~~C1: query-conditional placement beats random placement~~ | **DEMOTED.** Anticipated by RUTA Fig 6(a) and Q-CueGraph anti_crop/shuffled. We may report it as a **budget-matched replication**, never as the discovery. |
| 6 | ~~C3: where the benefit stops (crossing point)~~ | **Weakened.** RUTA Fig 5(b) already shows the advantage narrowing with budget (VisionZip overtakes at 1024 tokens); Q-CueGraph's regime table covers "when does it help". |

## Consequence for the write-up
**Do not write C1 as a discovery.** The defensible sentence is narrower and must be checked against
E0's result before it is written at all:

> Prior work establishes that query-conditioned region selection beats random and anti-region
> selection at matched image *area*. We ask whether it survives at matched *token budget* against
> the allocation policy production systems actually ship -- AnyRes tiling -- which no prior work
> compares against.

If E0 shows tiling closes the gap, even that sentence dies and the paper's centre moves to #3/#4.

## Checked: is Q-CueGraph §4.4 already the behavioural shadow of C2? -- YES, PARTLY. Cite it.
Their localization-vs-utility analysis reports that **2,699 of 19,221 crops whose OCR contains the
gold answer (14%) still yield a wrong crop-only answer**, localization/utility correlating at
r=0.77; their failure taxonomy (Table 15) names **"reader failure -- the evidence lies inside the
window and the frozen reader still answers incorrectly"** as 43% of TextVQA failures.

That is the **behavioural** observation that localization does not imply answerability. It is NOT
our C2, and the distinction is the whole contribution:

| | Q-CueGraph §4.4 | our C2 |
|---|---|---|
| where the evidence is shown to be present | **in the pixels** handed to the reader (answer string in the crop's OCR) | **in the model's hidden states** (Phase 14 probe) and **in the attention ranking** (Phase 22, +20-35pp over chance) |
| what fails | the frozen reader, given a good crop | the readout, given an internally-encoded and attended target |
| evidence type | black-box, input-side | internal, activation-side |
| interventions tested | none | steering (Ph16), attention patching (Ph6e), pointing/redbox/coords (Ph13) -- all null |

**Write-up rule:** cite Q-CueGraph for the *phenomenon* ("localization does not imply
answerability" is theirs, and their numbers are better-powered than ours). Claim only the
**mechanism** -- encoded, attended, still unusable, and not reachable by inference-time
intervention. Claiming the phenomenon would be a scoop violation; claiming the mechanism is clean
because neither concurrent paper touches internals.

---

# SECOND PRIOR-ART HIT (2026-09-11): the METHOD space is closed. Read before proposing any fix.

Checked before pitching, per the standing rule. Two papers kill the two strongest method ideas.

## ViRGo — "Look Before You Zoom" (arXiv 2606.21968, 2026-06-20)
Kills: **"decide WHETHER to zoom"** routing, **"use the VLM's own localization heads as the
proposer"**, and **RQ-D (where does placement hurt)**.
* Routes per input between **global / attention-crop (ViCrop) / patch-retrieval (RAP)**.
* Router features are exactly what we would have used: **implicit object scale from the VLM's
  localization heads**, top-1 token confidence, source image resolution. Lightweight XGBoost.
* **Runs on Qwen3-VL-2B — our exact model** — plus LLaVA-1.5-7B/13B, LLaVA-ov-0.5B.
* Benchmarks: V*Bench, HR-Bench4K, GQA size bins, TextVQA.
* They already name our RQ-D as their motivating finding: the **"resolution-context trade-off"** --
  patch retrieval *falls below the global baseline* on large objects. Their Fig 3a is our
  §4.1 result, found first.
* Qwen3-VL-2B: baseline 64.4, ViCrop 60.5 (**below baseline**), RAP 68.2, ViRGo 70.2, **Oracle 77.1**.

## "Same Attention, Different Truths" (arXiv 2608.07302, 2026-08-07)
Kills: the **probe/lens-based readout correction**, and substantially anticipates our §6 mechanism.
* Their headline finding is our Phase 22: **real and hallucinated objects receive equal attention
  magnitude** -- so hallucination is not an attention-strength failure.
* They then do what we proposed: apply **Logit Lens to the attended visual tokens** and show real
  objects decode to consistent tokens while hallucinated ones do not.
* Training-free **detect + mitigate**: Logit-Lens Consistency Check, High-Attention Region Masking,
  Visual Evidence Enhanced Decoding. SOTA on CHAIR and AMBER across four LVLMs.

## What this means
**Do not pitch:** when-to-zoom routing; localization-head proposers; foveated/adaptive-resolution
encoding (FAVE 2609.04392, Foveated Reasoning 2604.21079); learned zoom policies (2609.03206);
lens-based hallucination detection or mitigation; "zooming hurts large objects".

**What no one has done** -- and it is all downstream of the one asset we built:
1. **Nobody in this field matches the VISUAL TOKEN BUDGET.** ViRGo compares wall-clock *latency*;
   Q-CueGraph compares image *area*; RUTA states outright it is not rate-matched. Every published
   ranking of these methods is therefore confounded with how many tokens each method spends.
2. **Nobody charges the proposer's cost to the budget.** Every two-pass method pays for pass 1 and
   none of them count it in the comparison.
3. **The exchange rate itself** (§4O: no crossing at 26x) is unmeasured anywhere, because only a
   rate-matched protocol can ask it.

---

# SECOND PRIOR-ART SWEEP (2026-09-11): the METHOD space. Read before claiming any method.

## Hit 1 — ViRGo (arXiv 2606.21968, 2026-06-20). Closest to our RQ-D/RQ-E, and it uses OUR backbone.
"Look Before You Zoom: Adaptive Routing for the Resolution-Context Trade-off in Visual RAG."
* Identifies the **resolution-context trade-off**: patch-zooming recovers small targets but
  fragments large objects; attention-cropping preserves context but misses tiny detail.
* Routes among **global / ViCrop (attention-crop) / RAP (patch-retrieval)** using a **learned
  XGBoost router** (558 training samples) over three zero-shot features: implicit object scale from
  the VLM's **localization heads**, top-1 token confidence, and image resolution.
* Benchmarks: **V*Bench (191, same as ours)**, HR-Bench4K, GQA bins, TextVQA. Backbones include
  **Qwen3-VL-2B -- our model.** Reports an Oracle row.

**What this takes off the table:** (i) "router that decides whether to zoom" as a novel method;
(ii) "zooming hurts for large objects" as a novel boundary finding.

**What it does NOT do, and this is the whole opening:** ViRGo's Pareto axis is **inference TIME
(seconds)**, never token budget. It never matches tokens across compared strategies. Latency is
implementation- and hardware-dependent; **token count is the architectural quantity that KV-cache,
context limits and serving cost scale with.** Their router is also *learned*; ours is a matched-
budget measurement, not a trained component.

**⚠️ ACTION BEFORE WRITING §4.1.** Their Fig 3a covers accuracy-vs-object-scale for RAP/ViCrop/
global. Check whether it already covers our `relative_position` boundary on the same split. If it
does, **cite them and present §4.1 as a replication that adds the budget axis**, not as a finding.

**⚠️ NUMBER TO RECONCILE BEFORE PUBLISHING.** Their Qwen3-VL-2B row reports **RAP at 78.9% on
V*Bench**; our `crop_only@300` is **93.7%**. Same benchmark, same backbone. Most likely our
**oracle is doing much more work than their proposer** -- which is exactly why the oracle caveat in
§9 is load-bearing and must not be softened. Alternative explanation: wildly different budgets.
Resolve this before submission; do not quote 93.7% next to their 78.9% as if comparable.

## Hit 2 — RoRA (arXiv 2608.07088) and the pruning family (FastV, D2Pruner, HoloV, DART, VisionZip)
Training-free **visual token pruning**: given a retention budget, choose which already-encoded
tokens survive into the LLM. RoRA splits the budget into semantic core / context / detail using
Attention-Anchored Regions. Reports matched-budget comparisons and beats D2Pruner by ~5% on
Qwen3-VL at 75-90% pruning.

## ⚠️ MY FIRST FRAMING OF THE METHOD GAP WAS WRONG -- corrected
I wrote that "pruning cannot recover what was never encoded." **That is false as stated** and a
pruning researcher would kill it in one line: FastV/D2Pruner/RoRA prune *after* a full
AnyRes/dynamic-resolution pass -- thousands of tokens at native resolution. The information WAS
encoded. They are choosing what survives into the LLM.

**The correct and defensible distinction:**
> **Pruning selects among a fixed spatial sampling. Allocation changes the sampling.**

A pruner cannot spend more resolution on a region than the encoder's grid already gave it.

**And the asymmetry that follows, which the paper should state explicitly:** pruning reduces
**LLM-side** cost only -- the encoder still pays full price. Allocation reduces **both**. So at a
matched *LLM-side* budget a pruner starts from strictly more information than our allocator, which
makes it a **conservative** comparator: if allocation still wins, the result is strong.

## Hit 3 — FAVE (2609.04392, 2026-09-03) and LLMind (2603.14882, CVPR'26). The method space is SATURATED.
* **FAVE** — foveated adaptive visual encoding. A **trained** variable-resolution ViT (5.7M params,
  NaViT-style factorized positional encoding) that encodes *externally selected* regions at native
  geometry. Explicitly "separates **where to look** from **what to encode**" and owns the latter.
  Uses **oracle GT boxes** in its controlled study -- the same framing as ours.
* **LLMind** — training-free non-uniform sampling (Mobius/cortical-magnification warp) with
  test-time SPSA optimization against a frozen VLM. Runs on **Qwen3-VL-2B**, our backbone.
* Also in the space: **AttWarp** (attention-guided image warping, ICLR'26), **ViCrop / RAP /
  ZoomEye / Q-Zoom / SD-RPN** (crop-and-re-encode), **Visual CoT / Foveated Reasoner** (acquisition
  inside the reasoning loop), **Mini-Gemini / Cambrian-1** (multi-encoder).

### ⇒ DECISION: stop searching for a method gap. There isn't one. There IS a measurement gap.
Five sweeps returned "close but not identical". That pattern is itself the finding. **Every one of
these papers controls a different resource, and none controls realized visual token count:**

| family | resource held fixed | what stays at full cost |
|---|---|---|
| pruning (FastV, RoRA, D2Pruner) | **LLM-side** token count | the **encoder** |
| LLMind / Gizdov "information-matched" | **pixels** sampled | the **token count** (every arm is upsampled back to full resolution) |
| ViRGo | inference **time** (seconds) | tokens uncontrolled |
| ViCrop / RAP / ZoomEye / FAVE | nothing matched | both |
| **ours** | **realized visual tokens** (`image_grid_thw`), gated at 10% spread | -- |

**Token count is the right axis** because it is the architectural quantity that KV-cache, context
limits, and serving cost scale with. Pixels are a sensing cost; seconds are hardware-dependent.

---

# THIRD SWEEP (2026-09-15): full-text reads. Two 09-11 action items CLOSED, one NEW hit on §2.

The 09-11 entry logged ViRGo from its abstract and left two ⚠️ ACTION items. Both are now resolved
against the **full text**. One new paper (SmartRes) was not in any earlier sweep and it is the first
prior art that beats uniform **at a matched token ratio**.

## ✅ CLOSED — "does ViRGo's Fig 3a already cover our §7 cross/single boundary?" NO. And they name
## the gap themselves.
ViRGo bins **GQA by relative object SIZE** (bins 0-7, Fig 1/3a). That is the *size* axis, which our
§3 already retired as a rule in its own right. They have **no region-count / multi-region split
anywhere**, and their own §7 Limitations says so in as many words:

> "these mainly capture geometric aspects of visual perception, leaving other factors that may
> influence routing decisions — such as **relational reasoning, compositional understanding, or
> multi-object interactions** — unexplored. This limitation is further reflected in the substantial
> gap between ViRGo and the Oracle routing performance, suggesting that the current routing signals
> **do not yet fully characterize the underlying perception difficulty**."

**This is the single best positioning sentence available to us and it is in their paper, not ours.**
Evidence-set coverage (§3) *is* a candidate for "the quantity that characterizes perception
difficulty", it *is* defined over multi-region evidence, and their oracle gap (70.2 → 77.1 on
Qwen3-VL-2B, **6.9pp left on the table**) is the headroom it would explain.
⇒ **Write §3 and §7 as answering ViRGo's stated open problem. Cite the limitation verbatim.**
⇒ But state honestly: they found the resolution-context trade-off (our §7 "cross" story) **first**,
   on the size axis. §7 is *not* a novel boundary finding; it is a novel **explanation** of one.

## ✅ CLOSED — the 78.9 vs 93.7 reconciliation. Three reasons, none of them favourable to quoting
## them side by side.
1. **Different n.** Their Table 1 splits V*Bench 191 into **38 train / 30 dev / 123 test**; every
   V*Bench number in their Table 2 is on **n=123**, not 191. Ours is n=191.
2. **Different budget.** Their "Baseline" is Qwen3-VL-2B at **native dynamic resolution** (64.2%);
   our `uniform@300` is a 300-token fit (56.5%). Their arms are not token-matched to ours or to
   each other — which is the whole premise of our §2.
3. **Different oracle.** Their `Oracle` row (77.1) is a **routing** oracle: pick the best of three
   *strategies* per item. Ours (92.7) is a **placement** oracle: the GT box. These are not
   comparable quantities and must never appear in the same column.
⇒ **Do not quote 93.7 against 78.9.** The §9 oracle caveat stands and is load-bearing.

## 🆕 NEW HIT, and it bites §2 — SmartRes (arXiv 2608.01638, 2026-08-03, NUS/HKUST/NTU)
"Dynamic Resolution Routing for Efficient Egocentric Grounding." Not in any earlier sweep.
* LR pass → lightweight MLP **router** over LR features → binary mask → re-encode *selected patches*
  at HR → order-preserving interleave of HR and LR tokens in one sequence. Qwen2.5-VL-3B.
* Trained end-to-end, **supervised by box-projected token labels** + a margin-hinge loss for
  foreground/background imbalance.

**Why it LOOKS like it bites §2 — and why it does not.** It is the first paper we have found
that reports a token-ratio column and compares against uniform down-scaling at that ratio, with
allocation winning **while spending fewer tokens**:

| arm | ratio | RefCOCO-family Overall P@0.5 |
|---|---|---|
| Vanilla (full res) | 100% | 87.95 |
| **Down-scaling** | 50% | 64.78 |
| Down-scaling | 32% | 54.11 |
| **SmartRes-Pro** | **42%** | **70.88** |

That is the exact shape of the §2 claim (policy at budget B beats uniform at budget > B), and they
say it out loud: *"uniform down-scaling provides a competitive low-cost baseline on REC… SmartRes-Pro
improves over Down-scaling-50% … by 6.10."*

**⚠️ THE DEFENSE IS §1'S OWN INSTRUMENT, AND IT IS DECISIVE. Their 42% is a MEAN, not a budget.**
Verified against the full text:
* Table 5 caption, verbatim: *"Ratio denotes **average** HR-token retention."*
* §4.4, verbatim: *"while using only 42% of tokens **on average**."*
* The configuration is **SmartRes-Pro (r_LR=10%, r_HR=100%)**, and the table's own section header is
  **"Adaptive: 10% → 100% tokens"** — a **10× per-item span**.
* Down-scaling is **fixed per image** (isotropic resize to the target ratio, Eqs. 10-12).
* **No per-item distribution, variance, histogram or spread is reported anywhere in the paper.**
  Fig. 6(a) sweeps τ to produce different *mean* ratios; it is a frontier over means, not a spread.

**Our budget gate voids any contrast spreading >10%. Theirs spreads 10×.** And the correlation is
not incidental — the router is **trained on GT-box-derived labels**, so per-item budget is allocated
by a ground-truth-informed difficulty signal by construction. An adaptive arm whose per-item spend
correlates with per-item difficulty beats a fixed-budget arm at equal *mean* spend **without
allocation doing any work**. This is the confound §1 was built to void, and SmartRes has it.

⇒ **§2 does NOT need the retreat I first wrote here.** State the defense in our own currency:
  > SmartRes reports allocation beating uniform at a lower mean token ratio. Its ratio is an average
  > over a 10×-wide per-item range set by a GT-supervised router, against a fixed-budget uniform arm;
  > no per-item budget distribution is reported. Under our budget gate (>10% spread voids the
  > contrast) it is not a matched-budget comparison.
⇒ Three secondary qualifiers, worth one clause each but **not** the load-bearing argument:
  **trained** (vs our inference-time comparators), **grounding/P@0.5** (vs VQA accuracy — and our §4
  result is exactly that localization ≠ answering), and **routes before encoding** (so the §4 clause
  "crops cost passes the budget axis then outruns" is not true of SmartRes and must not be stated
  universally).

**And it supports §5.** Their `Attn2HR` ablation is a training-free attention-top-K selector (visual
self-attention, layer 30, K=25) built for exactly this purpose. It scores **36.12** against the
learned router's **49.39** at the same ~33% budget, with **FG-Recall 18.5% vs 54.0%**. Independent,
published confirmation that a **training-free attention peak is a weak proposer** — which is our §5
("half of all windows miss") measured by someone else on another backbone and another task.
⇒ Cite Attn2HR as external corroboration of §5. It also closes off "attention peak as a *proposer*"
  as a novel method; ours is a *diagnostic*, and must be written that way.

## Net effect on the paper
| section | status after this sweep |
|---|---|
| §2 negative | **survives intact.** SmartRes is the only challenge and it fails our budget gate (10× spread on a mean ratio). Add one clause naming it and the spread; do not retreat to qualifiers. |
| §3 coverage | **survives, strengthened** — answers ViRGo's named open problem |
| §4 interventions | **untouched** — nobody does internal-repair ablations at matched tokens |
| §5 serialization sink | **survives**; Attn2HR corroborates the weak-proposer half |
| §6 gate | **weakest section, but two real discriminators** (below). Frame as **matched-budget**, **training-free**, and **predicting coverage, not strategy**; never as "we introduce gating on an internal signal." |
| §7 boundary | **not novel as a finding** (ViRGo Fig 3a, first); novel as an **explanation** |

---

# SWEEP OF PROPOSED *NEW DIRECTIONS* (2026-09-15). Verdicts on four candidate pitches.

Context: these were evaluated as **possible next directions**, not as claims in PAPER_FLOW v10.
v10 makes no "cliff" claim; item (4) below concerns a *candidate* finding, not a live one.
Recorded here so the arXiv IDs and the numbers survive.

## (1) ❌ DEAD — patch-grid jitter ensembling (average predictions over sub-patch offsets)
**Phase Marginalization for Patch-Grid Instability in Vision Transformers (arXiv 2606.08132).**
Essentially the identical proposal: patch-grid phase φ=(dx,dy) as a nuisance variable, *Uniform
Phase Marginalization*, training-free and post-hoc, K=4 offsets Φ₄={(0,0),(0,P/2),(P/2,0),(P/2,P/2)},
inverse-align outputs, average logits, frozen backbone.
Three reasons it is worse than plain duplication:
* **The structured-vs-generic margin is tiny.** Their compute-matched Cityscapes table (DINOv3, 4
  forwards each): single pass 52.90 → random subpatch-shift TTA 53.00 → integer-shift TTA 53.22 →
  Phase Marginalization **53.53**. That is **+0.31 mIoU over plain shift TTA.** K=8 unchanged; K=16
  +0.68 at **16.7× latency**.
* **It needs a DENSE output to inverse-align.** Evaluated on segmentation/depth/matching only. A VLM
  emitting one MCQ token has no field to align, so it degenerates to generic TTA over shifted images
  and forfeits exactly the +0.31 that distinguishes it.
* **Their documented failure mode is the obvious implementation.** "Pre-Transformer Patch-Embedding
  Averaging" scores **44.48 mIoU against a 51.94 baseline** — averaging phase-shifted embeddings
  before attention *collapses*. Aggregate after aligned outputs, never at the embedding level.

**PPOM (arXiv 2608.13969)** already ports phase marginalization to CLIP vision-language prompt
tuning. ⚠️ **Read at ABSTRACT LEVEL ONLY** — title and abstract are unambiguous, full text unchecked.
Also: **Alias-Free ViT (2510.22673)**, **SPoT: Subpixel Placement of Tokens (2507.01654)**,
Rojas-Gomez et al. *Making Vision Transformers Truly Shift-Equivariant* (CVPR 2024). If ViTs are
being made shift-equivariant by construction, ensembling over shifts buys progressively less.

**The deeper objection is mechanistic.** 2602.02539 separates reversible *patch-alignment
sensitivity* from irreversible *information-capacity exhaustion*. Jitter ensembling targets the
first; our oracle-crop control (flat at 96-100%) points at the second. Wrong mechanism.

## (2) ❌ CROWDED — gated uniform upscaling
Dense field: LLMind (2603.14882), Q-Zoom (2604.06912), FAVE (2609.04392), AdaTok (2606.07185),
RUTA (2608.04132), SmartRes (2608.01638), plus native dynamic resolution in Qwen2.5/3-VL and
InternVL tiling. **2602.02539's Discussion proposes it as future work verbatim** — *"simpler
documents can be processed with aggressive compression, while visually dense documents can
dynamically trigger higher resolutions or tiling strategies… such 'content-aware' computation is
essential"* — and their scaling law predicts the required budget **without a forward pass**, a
stronger gate than ours. See also ViRGo above, which owns the internal-signal gate.

## (3) ❌ DEAD — "multi-scale in one pass" is shipped architecture
Global thumbnail + higher-res tiles in one sequence is InternVL's *Tile-based High-Resolution*
strategy, i.e. the baseline, not a proposal. Our Phase 54 already showed the composite loses.

## (4) ⚠️ A "cliff" finding would NOT be pre-empted — but two papers are close. Cite both.
**How Much Information Can a Vision Token Hold? (arXiv 2602.02539).** Three-regime phase transition
(Stable / Instability / Collapse) with a "Hard Wall". Scaling law **Z = w₀ + a·log V + α·log G**,
G = chars/token (load), V = lines-per-patch (density), a=2.91, α=5.53. Transition width
N_collapse/N_stable ≈ **2.2, resolution-independent**. Transfers across DeepSeek-OCR,
InternVL3.5-8B, Qwen2.5-VL-8B with frozen exponents. **It has our control's analogue**: a *Visual
Density Alignment* experiment pasting Stable- and Collapse-group images onto a fixed 3584×3584
canvas to match character size, refuting a "legibility" account. **But**: dense rendered text / OCR,
edit distance, and an *aggregate load* quantity — not a single small target in a natural image, not
VQA accuracy.

**The Last Visible Pixel (arXiv 2606.07861).** FineSightBench probes minimum perceivable object size
at 4-48px on 448×448 canvases. Headline: **"perception saturates around 12px, while reasoning
remains limited even at larger scales."** RT₅₀ = smallest size reaching 50% accuracy. Wide coverage
incl. Qwen3-VL 2B/4B/8B/30B, InternVL3.5, Gemma-4, Gemini-2.5-Flash, gpt-4o. **But**: threshold in
**pixels, never normalised to visual tokens**; largely synthetic stimuli; its "oracle" is a **text
oracle** (+10.7% to +50.7%), not an oracle crop. ⚠️ **Chase its ref [59]** — cited for prior
intervention-based causal analysis of small-object difficulty.

**What would survive as ours:** a threshold in **merged visual tokens** (0.15-0.25) for a **single
target in a natural image**, established by an **oracle-crop-flat** control, plus "localisation is
empty" and "tokens-on-target is a sufficient statistic."
**What is already damaged:** "architecture-invariant threshold" is weak — 2602.02539 transfers a
scaling law across three architectures and 2606.07861 shows ~12px saturation across a dozen models.
Honest framing is *"we express the limit in the model's own token currency and isolate encoding with
a crop oracle,"* never *"we discover that VLM perception has a limit."*

## Least-crowded direction found
A **matched-budget evaluation of the adaptive-resolution field**. None of LLMind / Q-Zoom / FAVE /
AdaTok / RUTA scores against a uniform-budget bar. Evaluation contribution, not a method — but it is
the least crowded thing available and the harness already exists. Also unexamined:
**HAFI-VLM (2608.02124)**, which diagnoses fine-grained perception failure via **"spectral response
rigidity"** (frequency-domain) — read before proposing anything spectral; and **2609.03429**,
image-free object-token edits at the representation level, adjacent to our failed internal
interventions and carrying an **answer-key-free protocol** we might reuse.

## 🔎 MINED FROM ViRGo — the single most useful external fact §2 has, and a landmine if we miss it
**ViRGo Table 2, dynamic-resolution block, Qwen3-VL-2B — our backbone, BOTH our benchmarks.
The sign is against us on both, including the one we lead with:**

| | baseline | ViCrop | **RAP** | ViRGo | Oracle | our §2 best crop arm |
|---|---|---|---|---|---|---|
| **V\*Bench** (n=123 test) | 64.2 | 57.7 | **78.9 (+14.7)** | 78.9 | 88.6 | Zoom Eye **−5.2** |
| **HR-Bench 4K** (n=512 test) | 54.1 | 52.3 | **70.9 (+16.8)** | 71.1 | 77.5 | Zoom Eye **−15.3** |

⚠️ The 09-11 note logged 78.9 only as "a number to reconcile against our 93.7 **oracle**". That is a
different question. **The real exposure is that it contradicts our §2 SIGN, on the benchmark §2
leads with.** We are citing this paper approvingly for its Limitations sentence; a reviewer will
open Table 2.

**The reconciliation is in ViRGo's own Table 3, and it is our argument, made by someone else:**
RAP total inference time **89,185s vs baseline 1,335s ≈ 67×**. An independent group shows a crop
policy buying +16.8pp for **67× the compute**, on our benchmark and our backbone, with **tokens
never matched** (their baseline is native dynamic resolution; their axis is seconds).
⇒ **Put this in §2 or §9.** It is external corroboration that the crop literature's wins are
  purchased with unaccounted budget. State their n: HR4K test split is **512 of 800**.

## 🔎 §6 vs ViRGo — two discriminators we have and did not claim
1. **ViRGo's router needs an ANSWER KEY.** Their Label Construction runs all three strategies per
   item and records which was correct, over **558 training samples**. `peak` is computed from pass 1
   with **no labels and no training**. That is a deployability difference, not a cosmetic one.
2. **Different target variable.** ViRGo predicts *which strategy wins*. `peak` predicts *whether the
   proposed window will contain the evidence*. **Coverage is not strategy-selection**, and coverage
   is the quantity their Limitations says is missing.
3. Calibration, for our own morale: their **Table 4** ablation gives **61.4-62.2** for single
   features vs **63.2** for the full router. Their individual signals are weak too. Our AUROC 0.788
   is unremarkable, not embarrassing.
