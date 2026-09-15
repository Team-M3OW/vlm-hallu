# Literature Review — VLM Localization-vs-Perception Gap

Working notes for the paper: "VLMs localize well but don't answer/perceive/classify well
enough" — culprit search (autoregression / image-text entanglement / something else) +
test-time fix. Origin: `darpa.md` (DARPA Triage Challenge trauma-detection pipeline,
`llava-lens` repo: github.com/arsh-6626/llava-lens).

Scope note: DARPA/trauma framing and private dataset have been dropped from the paper
itself (see conversation) — kept here only as originating context. Paper is now a
general VLM mechanistic study using public benchmarks.

---

## 1. Current best hypothesis (after this literature pass)

**Original 3-way question:** is the culprit (a) autoregressive decoding, (b) image/text
token entanglement, or (c) something else?

**Where that lands after review:** all three have substantial, specific prior work.
Nothing in the "generic" version of any of these three is open. The gap is a specific,
narrow intersection nobody has run:

> Does shrinking **evidence area** (fraction of image occupied by the diagnostic detail)
> cause an information-routing failure — via a shift from the *direct* image-token
> readout pathway to the lossier *text-mediated* pathway (Salazar et al.), and/or via
> numerical dilution of evidence-patch attention mass at the answer position — **that is
> mechanistically separable from already-documented arbitration failure / language-prior
> override**, tested with prior-conflict held constant?

Critical complication found late in the review: **the exact term "critical evidence
dilution" is already coined** by Xiu et al. 2026 ("Beyond Scene Priors," §2 below) for
essentially this failure mode, with a working (trained, architectural) fix. Our
differentiation has to be: (1) mechanistic/causal internals (logit lens, pathway
patching, activation patching) — they never open the model; (2) training-free /
test-time only fix — theirs requires LoRA + auxiliary losses; (3) continuous
area-as-independent-variable with a prior-conflict-held-constant control — nobody does
this stratification; (4) general-purpose VLM (LLaVA-1.5-class), not domain-locked to
traffic.

---

## 2. Papers read in full (methods + findings extracted from PDF, not just abstract)

### Arbitration / language-prior override

**Nooralahzadeh et al., "Arbitration Failure, Not Perceptual Blindness"** (arXiv 2604.09364)
- Visual-Counterfact dataset (color/size attribute conflicts, e.g. blue banana). 10 VLMs, 7B–72B.
- LogitLens → Multimodal Arbitration Crossover (MAC): layer where visual logit stably beats prior logit. Depth varies 36–71% across models.
- Linear probes: visual attribute is linearly decodable (AUC>0.99 by mid-depth) **regardless of whether the model ultimately grounds correctly** — encoding is NOT the bottleneck (encoding-grounding dissociation).
- Causal: **full-sequence activation patching** works (60–84% flips); **last-token patching does not** (0–1% flips) — critical methodological finding, must not build ablations on last-token-only patching.
- Image-token-only patching carries ~all the causal effect in larger models.
- Fix: linear/SAE activation steering, training-free, +up to 3.8% grounding accuracy. Best intervention layers are *earlier* than the diagnostic MAC layer.
- **Gap left open:** doesn't vary evidence area at all (all objects roughly full-frame in Visual-Counterfact).

**Wang et al., "Same Attention, Different Truths"** (2608.07302)
- LLaVA-1.5-7B/13B, Shikra, Qwen2-VL. COCO/CHAIR/AMBER captioning.
- LogitLens over top-k *attended* image tokens. Finding: real and hallucinated objects get **comparable attention magnitude** — the discriminator is *semantic consistency* of what's decoded from attended regions, not attention amount.
- Two hallucination types via masking-intervention: **Visual Uncertainty** (masking evidence removes the hallucination, ~2/3 of cases) vs **Contextual Prior** (masking doesn't help, attention just moves elsewhere, prior wins, ~1/3).
- Fix: training-free Detect (LLCC: logit-lens/attention consistency check) + Mitigate (HARM = mask uncertain regions; VEED = fuse masked-image logits with attended-patch logits to boost visual evidence). SOTA on CHAIR/AMBER across 4 models.
- **Gap left open:** no area/size axis; task is open-ended captioning, not existence/classification; doesn't test whether the visual-uncertainty/contextual-prior split correlates with target size.

**Zhou et al., "Diagnosing Visual Ignorance in VLMs"** (2606.06890)
- Qwen2.5-VL-3B/7B, LLaVA-1.6-Mistral-7B. 12 VQA benchmarks (RePOPE, HR-Bench, HallusionBench, V*Bench, VLMBias, etc.)
- **Methodological critique we must adopt:** raw zero-shot LogitLens is "unstable/uninformative" in early-intermediate layers because hidden states aren't yet vocab-aligned. They instead use **counterfactual layer replacement** + **supervised layer-wise MLP probing** (trained classifiers per layer, not zero-shot vocab projection).
- Finding: **multi-stage bottleneck**, not a single crossover — intermediate layers fail to *retrieve* visual info; separately, late layers *actively suppress* surviving visual signal in favor of text bias. Two distinct failure stages, contra the single-crossover framing in Nooralahzadeh.
- External diagnostic: **progressive Gaussian-blur decay test** — track whether the model's answer stays invariant as the image is blurred through 8 kernel sizes; a "consecutively-identical-answer" metric gives a statistically strong (not-by-chance) lower bound on language-prior reliance. Result: 20–40% of samples on RePOPE/MMMU/etc. are answered identically under total blur — **benchmarks routinely fail to penalize visual ignorance.**
- **Actionable for us:** before trusting any area-stratified subset we build from POPE/LVIS, run this blur-invariance filter to certify the subset actually requires vision (not language-prior-solvable) — otherwise our area-accuracy curve could just reflect which items happen to be text-solvable.

**Zhang, Zhan, Wang, "When Visual Signals Mislead"** (VISOR, 2608.11024)
- Qwen2.5-VL-3B, InternVL3.5-4B, LLaVA-1.5-7B. VAW dataset (attribute yes/no probing, not object existence).
- Directly tests and **rejects** "language-prior dominance" as the driver of *attribute* hallucination (as opposed to object-existence hallucination, where prior-dominance holds per Nooralahzadeh/Wang). Decomposes decision into δ_vis (real image logit margin) vs δ_prior (blank/gray-image logit margin). δ_vis strongly predicts false positives (ρ=0.76–0.84); δ_prior is near chance (ρ=0.07–0.18).
- Two visual-signal failure modes: **Mechanism A (Visual Incertitude)** — color/state attributes, correct direction but low margin → fixable by threshold calibration (Calib). **Mechanism B (Visual Representation Inversion)** — material attributes, signal-to-noise collapses in **late decoder layers (L28–L36 of 36)**, from SNR≈35 at L17 to <0.9 at L36 — a genuine late-layer degradation of an initially-good visual signal, not present-but-overridden.
- Fix: routed operators (Calib / Abstain / Adapt-LoRA) selected by diagnosed mechanism; beats VCD/ICD (which get ~0pp on color, since there's no dominant prior to subtract).
- **Relevance:** shows the mechanism taxonomy is task-type-dependent (existence ≠ attribute). Also gives a directly analogous "late-layer signal collapse" phenomenon we can test for *area* instead of *attribute-type* as the independent variable.

### Circuit / pathway mechanics

**Liu et al., "Dual-Pathway Circuits of Object Hallucination"** (2605.13156)
- 5 architecturally diverse VLMs (Qwen3-VL-8B, LLaVA-v1.6-7B, Llama-3.2-Vision-11B, InternVL3-8B/14B). POPE-adversarial.
- Activation patching (Gaussian-noise corrupt → restore one component at a time) finds a consistent **dual-pathway organization** in every model: a **grounding pathway** (components whose restoring effect is bigger on correct trials) concentrated mid-to-late depth, and a **hallucination pathway** (bigger effect on error trials) concentrated early/boundary layers. Sizes/wiring differ per model but the macro pattern is shared — suggests it's a property of *training*, not architecture.
- Novel diagnostic (Conditional Pathway Analysis): grounding-pathway components show a **polarity flip** — the *same* components that support the correct answer on correct trials flip to supporting the *hallucinated* answer on error trials. I.e., the grounding machinery itself gets hijacked, not merely outvoted by a separate prior circuit. Redundancy is preserved in both regimes (not a synergy breakdown).
- Fix: scale down hallucination-pathway component outputs at inference (no retraining) → 40–76% hallucination reduction at ≤2pp accuracy cost, beats ITI/mean-diff-projection baselines. Transfers to relational hallucination, not to attribute hallucination (consistent with VISOR's claim that attribute failure is a different mechanism).
- **Terminology warning:** "pathway" here = *which transformer components* (attention/MLP sublayers) causally support correct vs wrong answers. This is a **different sense** from Salazar et al.'s "pathway" below (which tokens carry the info to the answer). Don't conflate the two in our writing.
- **Gap left open:** no area axis; existence-only (POPE); doesn't test image vs text token identity as the corrupted unit (corrupts all visual embeddings at layer 0, not comparing to text-mediated routes).

**Salazar et al., "Pathways of Visual Information Flow in VLMs"** (2607.03358)
- Qwen3-VL-4B (main), replicated on LLaVA-1.5-7B and InternVL3.5-4B. Synthetic shapes + natural (COCO/VG/What'sUp/VSR).
- Causal patching (paired counterfactual images, same text) distinguishes **direct pathway** (last token reads image tokens directly) vs **text-mediated pathway** (image info first copied into text query tokens, then read by last token).
- **Object recognition → direct pathway** (100% image-dominated on synthetic + natural). **Spatial relations → text-mediated** (98%/91% text-dominated). **Localization → depends on data**: direct on clean synthetic scenes, **flips to text-mediated on natural/cluttered scenes with distractors.** This "clean→direct, cluttered→text-mediated" shift is the closest existing analogue to an area/complexity-driven pathway shift.
- Backup-pathway result (important for our ablation design): corrupting the image to noise and patching in text-token hidden states from a real-image run **still recovers 82–100% accuracy**, even for recognition (which normally uses the direct pathway) — meaning text tokens already carry a redundant copy of enough visual info by mid-network, so a naive "ablate image tokens, see if answer flips" test can under-detect real effects (the model reroutes through the backup pathway and looks unaffected).
- Attention knockout on the direct pathway (r_I→last) → **negligible accuracy drop across all tasks** (the model doesn't need direct attention to the image at the final token at all) whereas knocking out the text-mediation route (r_I→T) causes large drops (up to −34.9pp on relations). Necessity (knockout) and usage (patching) can give opposite-seeming answers — must combine both.
- **Directly actionable methodology to replicate on our area-stratified set:** run this same three-way patching (image tokens / text-query tokens / last token) across evidence-area bins to test whether pathway choice shifts from direct→text-mediated as area shrinks, mirroring their clean-vs-cluttered localization result.

### Entanglement / connector geometry / token interpretability

**Kachko et al., "Through the LENS"** (2608.00561)
- LLaVA-1.5-7B, Qwen3-VL-8B-Instruct. CC3M captions.
- Mixture-of-Factor-Analyzers decomposition of residual stream into local low-rank neighborhoods, labeled by a VLM judge. Measures modality mixing directly.
- **LLaVA: progressive late fusion** — mixed (both-modality) neighborhoods go 0 (layer 8) → 55 (layer 16) → 472 (layer 29), i.e., 0%→0.6%→5.7% of components. **Qwen3-VL: early mixing, re-segregation, late recombination** (non-monotonic, architecture-specific — DeepStack injects visual features at multiple early layers).
- Causal: interpolating activations toward mixed-neighborhood centroids steers generation within- and cross-modally; beats DiffMeans/VL-SAE baselines.
- **Reading for us:** entanglement (mixing) is real, quantified, and depth/architecture-dependent — but it's a *global* geometric statistic, never conditioned on evidence area or patch count. Doesn't tell us whether small-evidence patches are more or less likely to end up in a "mixed" (entangled) neighborhood than large-evidence patches.

**Li et al., "Lost in Embeddings: Information Loss in VLMs"** (2509.11986) — Cambridge/Copenhagen/Microsoft
- LLaVA (MLP connector), Idefics2 (perceiver resampler), Qwen2.5-VL (patch merger). 6 datasets (SEED-Bench, VizWiz-Grounding, VQAv2, CUB, Flickr30k, COCO).
- k-NN overlap ratio: connector projection changes 40–60% of each image patch's nearest-neighbor structure (geometric distortion at the connector).
- **Patch-level reconstruction**: trains a model to reconstruct pre-projection embeddings from post-projection ones; per-patch reconstruction loss localizes information loss spatially.
- Key result: on VizWiz Grounding VQA (has answer-relevant region masks), **reconstruction loss in answer-relevant patches specifically predicts LLaVA/Idefics2 QA failure** (significant negative Spearman correlation, p<1e-5); loss in irrelevant patches doesn't hurt as much. Qwen2.5-VL shows no such correlation (its patch-merger connector behaves differently).
- **This is nearly our exact hypothesis but at the wrong granularity**: they identify *which* patches are high-loss post-hoc; they never stratify by *how many patches* the target occupies (evidence area). Our "dilution" hypothesis is about patch-*count* minority in downstream attention aggregation; this paper is about per-patch geometric/reconstruction fidelity at the connector. Related but distinct mechanisms — worth explicitly distinguishing in our related work as "micro" (sub-patch/connector-level) vs "macro" (cross-patch/attention-aggregation-level) information loss.

**Krojer et al., "LatentLens"** (2602.00462, ICML 2026)
- 15 VLM configs incl. LLaVA-1.5-7B, Qwen2-VL-7B, LLaVA-NeXT-34B, Molmo-7B/72B, plus a controlled 3-LLM × 3-vision-encoder training setup.
- **Major methodological correction for us:** raw LogitLens badly *underestimates* visual-token interpretability — only 24% of visual tokens are "interpretable" via LogitLens (averaged across models/layers) vs **68%** via their method (LatentLens: compare visual-token hidden state to a large bank of *contextualized text token* representations at every layer, not to the vocab/embedding matrix). LogitLens interpretability is artificially low at early/mid layers and only rises near the output layer — this is a property of the lens, not of when the visual token "becomes informative."
- **Directly undercuts our own darpa.md claim** ("informative patches are all in the last 10 layers", "soft prompts" have <1% confidence) — that finding may be a LogitLens artifact, not a real statement about when patches carry information. Must re-run patch-confidence analysis with a lens that doesn't have this early-layer blind spot (LatentLens-style contextual retrieval, or the supervised MLP probing from Zhou et al. above) before making any claim about "which layers carry the real signal."
- **Mid-Layer Leap:** visual tokens at the LLM's *input* layer already best match *contextualized* text representations from the LLM's own **mid layers (8–16)**, not surface/lexical embeddings — i.e., the projector targets a semantic (not lexical) subspace from the very first layer. Complements Lost-in-Embeddings (which shows local geometric distortion) and LENS (which shows *modality* mixing grows with depth): tokens can be both semantically well-aligned *and* geometrically distorted *and* not yet "mixed" with text — these are three different axes, not redundant findings.
- **Synthesis point:** if visual tokens remain substantially interpretable/semantically rich at essentially all layers (per LatentLens) despite connector distortion (Lost in Embeddings) and growing modality mixing (LENS), that weakens a pure "information-loss-at-encoding" story and strengthens the case that failure lives downstream — in *routing/aggregation* to the answer position (our dilution/pathway-shift hypothesis) rather than in the visual representation itself.

### Small-object / evidence-area behavioral benchmarks (no internal mechanism analysis)

**Xiu et al., "Beyond Scene Priors: FGTR-Bench + TSR-MLLM"** (2607.04149) — ⚠️ closest sibling, term collision
- Qwen3-VL-4B backbone. FGTR-Bench: 40,236 traffic MCQs (signs/signals/roadside-micro/participant-micro), built via multi-agent generation + expert audit; DriveQA-V (CARLA) for OOD transfer.
- **Coins "critical evidence dilution"**: "MLLMs tend to over-attend to backgrounds, overwhelming crucial small objects during visual-language alignment" — this is our exact phenomenon, already named.
- Fix (TSR-MLLM / TG-SOF): query-conditioned salience map over vision tokens (bilinear query-vision attention score + local-contrast sharpening term) → sparse Top-K gated residual added to only the most relevant vision-token slots, applied once at the decoder boundary, single forward pass, no external detector/re-encoding. **Requires supervised fine-tuning** (frozen backbone + trained TG-SOF adapter + decoder LoRA + 3-part loss: answer CE, bbox-alignment KL, hidden-state consistency). Beats same-scale fine-tuned baselines by 2.1pp overall, larger gains on evidence-local tracks (Nighttime Signal, Roadside Micro, Participant Micro).
- **What it does NOT do (our differentiation):** zero internal/causal mechanistic analysis — no logit lens, no probing, no activation patching, nothing about *why* dilution happens inside the network beyond "background overwhelms small tokens during fusion" as a stated intuition, not a measured/tested causal claim. No training-free variant. No continuous area-vs-accuracy curve (uses 5 categorical tracks, not area as a continuous IV). Domain-locked to traffic scenes only. Doesn't hold prior-conflict constant / doesn't test whether dilution is separable from language-prior override.
- **Action:** we cannot call our phenomenon "critical evidence dilution" without citing and clearly building on/past this paper. Our contribution has to be framed as: *mechanistically explaining and testing whether critical evidence dilution (Xiu et al.) is causally distinct from arbitration failure (Nooralahzadeh) / contextual-prior hallucination (Wang), and fixing it training-free* — i.e., we complete the causal/mechanistic half that this benchmark-and-architecture paper doesn't attempt.

**Han et al., "SOUBench" ("Can MLLMs Truly Understand Small Objects?")** (2604.22884)
- 15 models (GPT-5.2, Gemini-3, Qwen3-VL, InternVL3.5, etc. — no LLaVA-1.5). SOU-VQA: 18,204 VQA pairs from SODA-D/VisDrone/AI-TOD/SODA-A/RUOD (driving/aerial/underwater), COCO-standard small-object definition (≤1024px² absolute area). 6 subtasks (enumeration, counting, recognition, location, specific-count, peripheral-ID).
- Purely behavioral: best model (GPT-5.2) 58.97% vs human 82.5%. Scaling helps (InternVL3.5 1B→14B: 55.71%→59.53% driving). Supervised fine-tuning (SOU-Train) improves +1.2–1.7pp.
- No internal mechanism analysis, no training-free fix, fixed small-object threshold (not continuous area stratification), all data/code public (github.com/Hanfj-X/SOU) — useful as an extended small-area validation set / generalization check, not as a mechanistic contribution.

**Huang et al., "UltraVR"** (2606.05576)
- Ultra-resolution diagnostic benchmark (CCTV/PANDA, RS/DOTA, WSI/TCGA-BRCA, industrial AD/MVTec-LOCO). Structured GT chain-of-thought with operation labels (Grounding/Perception/Quantification/Integration/Inference) — a **black-box** process-diagnosis, not internal representations.
- Best model (GPT-5.5) only 44.9% macro accuracy; text-only control drops to 7.7% (confirms genuine visual dependence, unlike some benchmarks per Diagnosing-Visual-Ignorance's blur test). Errors concentrate in early evidence-grounding/local-perception stages; **downstream inference recovers once correct intermediate facts are supplied (GT-Prefix)** — a black-box analogue of "the problem is upstream evidence acquisition, not downstream reasoning," consistent with (but not proof of) our routing-not-encoding framing.
- Not applicable to single-forward-pass LLaVA-style mechanistic work — designed for agentic/tool-calling frontier models with multi-turn crop tools.

**Chen et al., "VisualNeedle"** (2605.26380)
- Agentic crop-tool benchmark (300 hand-curated questions). Key methodological device: **crop-black** ablation — replace returned crops with black patches of identical size, preserving the interaction trajectory, to test whether tool-enabled gains reflect genuine use of visual evidence. On VisualNeedle, crop-black brings strong models back down near text-only (confirms genuine evidence dependence); **on V\*Bench and HR-Bench, crop-black accuracy stays high (75.9–87.4% on V\*, up to 86.5% on HR-Bench)** — meaning those benchmarks can often be "solved" without truly using the returned small-region evidence.
- **Actionable warning:** if we use V\*Bench as a generalization check, its accuracy may not cleanly reflect small-evidence use — treat it as secondary validation only, not primary evidence, consistent with what was already flagged earlier in this project's discussion.

---

## 3. Papers surfaced but only abstract-level (not yet fully read — candidates for later)

- **Schaumlöffel, Vilas, Roig, "Mechanisms of Object Localization in VLMs"** (2605.19792) — LLaVA-1.5, InternVL-3.5; token ablation/attention-knockout/causal mediation; finds localization driven by a small set of causal heads via "containerization" (object-aligned tokens define spatial extent), largely distinct from classification heads. Read at abstract-depth only so far; worth a full read since it's the most direct support for darpa.md's original "localization ≠ classification" observation.
- **"Dual-Pathway Circuits..."** — done above (moved from this list after full read).
- HAFI-VLM (2608.02124) — frequency-domain account ("spectral response rigidity") of fine-grained perception failure — an entirely different "something else" candidate mechanism, unexplored relative to area/dilution. Not yet read in depth.
- "The Last Visible Pixel" (2606.07861) — fine-scale perception probing. Not yet read.
- "Where To Look? Causal Tracing of Vision Encoders" (2608.10758) — not yet read.
- "Where Vision Becomes Text: OCR Routing Bottleneck" (2602.22918) — routing-bottleneck concept for OCR specifically; methodologically relevant analogy. Not yet read.
- "From Drop-off to Recovery: Segmentation in MLLMs" (2603.17228) — not yet read.
- "MLLM-Microscope" (2606.00909) — intrinsic dimension/linearity analysis of hidden reps. Not yet read.
- "The Hidden Evolution of Disguised Visual Context inside the VLM" (2606.20077) — not yet read.
- LASER (2607.01707) — attention-sink suppression fix for long-horizon decoding visual forgetting. Not yet read; relevant as another training-free fix baseline to compare against.

---

## 4. Methodological corrections accumulated (must incorporate into any experiment design)

1. **Use full-sequence activation patching, not last-token.** Last-token patching gets 0–1% flips in VLMs (Nooralahzadeh); it would produce a false null.
2. **Run a forced single-token choice test first** (compare P(yes) vs P(no) in one forward pass, no sampling) as the cheapest experiment that isolates whether autoregressive decoding-as-a-process matters at all, before investing in decoding-time hypotheses.
3. **Don't trust raw zero-shot LogitLens for early/mid layers.** It's known to be unstable/uninformative there (Diagnosing Visual Ignorance) and to systematically *underestimate* interpretability everywhere (LatentLens: 24% vs 68%). Use supervised layer-wise MLP probing and/or LatentLens-style contextual retrieval as the primary tool; treat raw LogitLens results (including our own darpa.md "informative only in last 10 layers" claim) as provisional until cross-checked.
4. **Any patch-confidence claim needs null controls**: unrelated-concept tokens, and blank/shuffled-image patches, to rule out vocabulary-frequency-prior artifacts rather than real visual content.
5. **Fix the `ablate_inputs` bug in `hooked_llava.py`**: it's written as a generator (`yield` inside) without the `@contextmanager` decorator — currently cannot function as a context manager. Must fix before any ablation/patching experiment builds on it.
6. **Beware backup-pathway confounds in ablation design.** Text-query tokens already carry enough copied visual signal by mid-network to recover 82–100% accuracy even for direct-pathway tasks (Salazar et al.) — a naive "ablate image tokens, see if accuracy drops" test can silently fail to detect a real effect because the model reroutes. Combine necessity (knockout) and usage (patching) tests; don't rely on either alone.
7. **Certify any area-stratified benchmark subset against the progressive-blur invariance test** (Zhou et al.) before trusting an area-vs-accuracy curve — otherwise the curve may just reflect which items happen to be language-prior-solvable at each area bin, not genuine vision-dependent difficulty.
8. **V\*Bench and HR-Bench are not clean "genuine small-evidence use" benchmarks** — VisualNeedle's crop-black ablation shows both stay highly solvable (75–87%) even with the small-evidence region blacked out. Use them only as secondary/generalization checks, not primary evidence.
9. **Terminology collision:** "pathway" means different things in Salazar et al. (which *tokens* carry info: image vs text-mediated) vs Liu et al. Dual-Pathway Circuits (which transformer *components*/circuits causally support correct vs wrong answers). Keep these separate in our own writing; do not present them as the same finding.

---

## 5. Positioning statement (current draft, for the paper's related-work section)

Prior work has independently shown: (a) visual info is often encoded/decodable even when
the final answer is wrong, for attribute conflicts (Nooralahzadeh) and object hallucination
(Wang); (b) this is a routing/arbitration problem, not perceptual blindness, and is
training-free-fixable via activation steering (Nooralahzadeh) or attention-consistency
detection + logit fusion (Wang); (c) a dual-pathway circuit organization (grounding vs
hallucination components, Liu et al.) and a separate direct-vs-text-mediated *routing*
distinction (Salazar et al.) both exist and are causally validated; (d) the connector
introduces real geometric/patch-level information loss correlated with QA failure (Li et
al., "Lost in Embeddings"), and (e) small/fine-grained visual evidence specifically causes
failures in safety-critical domains (traffic: Xiu et al., who name this "critical evidence
dilution" and fix it with a trained architectural intervention) and in general small-object
tasks (SOUBench, V*Bench, UltraVR, VisualNeedle) — but always evaluated behaviorally, never
via internal causal mechanisms.

**No existing work varies evidence area as a continuous independent variable inside a
causal/mechanistic pipeline (logit lens + pathway patching + activation patching), with
prior-conflict held constant as a separate control, to test whether "critical evidence
dilution" is mechanistically identical to arbitration failure/contextual-prior
hallucination or a distinct routing phenomenon (e.g., an area-driven shift from the direct
to the text-mediated pathway) — and no existing fix for evidence dilution is training-free.**
That combination is the paper's target contribution.

---

## 6. Open items / next steps

- [ ] Full read: Schaumlöffel et al. (localization circuits) — closest support for the
      original darpa.md observation; needed before finalizing related work.
- [ ] Decide: keep chasing the full compound hypothesis (pathway-shift + connector-loss +
      area) or scope down to the cheapest single leg (pathway-shift-with-area) as the
      paper's core claim, per the fork point raised earlier in the conversation.
- [ ] Go/no-go pilot (not yet run): 1 concept, 4 area levels, prior-neutral only, on
      LLaVA-1.5-7B — patch-position supervised-probe/LatentLens confidence (not raw
      LogitLens) vs answer-position probe accuracy vs attention mass to evidence, per
      area bin.
- [ ] Public benchmark assembly (no dataset construction): POPE + COCO instance
      annotations joined for area-stratified existence accuracy (primary); V*Bench/SOUBench
      as secondary generalization checks (with VisualNeedle's crop-black caveat in mind for
      V*Bench); Visual-Counterfact as the external prior-conflict reference point.
- [ ] Fix `hooked_llava.py`'s `ablate_inputs` (`@contextmanager` missing) and extend to
      full-sequence patching before running any causal experiment.
