# Consolidated Report — VLM visual-token allocation and the depth of the attention read-out
*State of the project as of 2026-09-18 (FINDINGS.md §1–§30, commits through `ac8ffbf`).*

## 1. The problem
Vision–language models (VLMs) fail fine-grained visual questions in a specific way: **they localise but cannot perceive.**
Shown a high-resolution image at a fixed visual-token budget, the model's attention points at the right region while the
answer is wrong, because the evidence was merged away during encoding. The field's remedy is *attention-guided cropping*
(ViCrop, ZoomEye, VEA, LASER …): read the model's own attention map, crop where it peaks, re-encode the crop, answer.
Every such method must decide **where in the network's depth to read that map**, and every published one hand-picks a
layer or averages a block. This project asked: is that read-out depth right, what does reading at the wrong depth cost,
and can the depth dimension be used systematically — under an honest evaluation.

## 2. Standing rules (the evaluation discipline)
- **Equal compute.** Every method is scored against *the same tokens spent on plain resolution* (uniform@600 vs
  localise@300 + crop@300). Published methods do not do this.
- **Two models or nothing.** A claim must hold on ≥2 checkpoints (Qwen3-VL-2B and Qwen2-VL-7B as the core pair, plus
  Qwen2.5-VL-7B, Qwen3-VL-8B, LLaVA-NeXT, LLaVA-OneVision where extracted) or it is rejected.
- **Pre-registration.** Primary contrasts and guards are written before each run; context arms are ineligible as the method.
- **Stratify by question type** (single-instance vs cross-instance/relational).
- **Public benchmarks only** (V*Bench, HR-Bench 4K/8K, POPE, MMBench, TextVQA); no datasets built; no routers,
  question classifiers, or oracles in the method; no image preprocessing.
- **Coverage is an intermediate metric**; only end-task accuracy at matched compute is a claim.
- Honest negatives are logged with diagnoses; two of the project's own diagnostics were withdrawn when controls failed.

## 3. The method (as it stands): depth re-ranking with a ridge depth filter
1. Encode the image at 300 visual tokens with the full prompt (question + options + answer instruction — the read-out
   must be taken at the **answer-emission position**; without the instruction line the map is near-useless, −36 to −42pp).
2. Read the final prompt token's attention over image cells at **all 28 decoder layers** (raw heads averaged, then
   normalised — the convention is load-bearing).
3. Score every cell with **one closed-form ridge solve**: `Σ_ℓ w_ℓ·log A_ℓ + Σ_ℓ v_ℓ·rank_ℓ + a·log(3×3 nb) + geometry`
   (≈65 weights; fit on ~50–191 boxed items; no boosting, no tuning; the weights *are* the depth filter).
4. Crop a window of 25% width/height at the argmax cell (ring-masked), re-encode at 300 tokens, answer.
Total compute = 600 visual tokens = the bar. Label-free variant: max over the four "question-conditioned" layers
(gate-max), no training.

## 4. Experiments, in order (190 phases; the ones that decided something)
| block | what was run | decided |
|---|---|---|
| §4–§13 | budget sweeps, oracle crops, internal interventions (steering, patching, pointing, attention re-weighting ×7) | the **encoding cliff** at ~0.25 merged tokens; 84–87% of errors oracle-fixable; nothing downstream recovers unencoded evidence |
| §14–§16 | per-layer attention dumps (4 architectures), pruning by attention read at each depth, learned re-ranking head (GBT), W sweep, question-blindness probe | layers disagree; the block mean cancels; layer-2 pruning is below random; question-blindness until ~half depth; head clears the bar on single-object |
| §17 | reviewer-driven: learning curve, latency, HR-Bench 8K, Qwen2.5-VL, Qwen3-VL-8B, routing diagnostics | ~50 boxes → 70% of the gain; 1.4–2.0× latency; replications (tree placements) |
| §18A–§18O | nine question-type adaptation designs; sink weighting; ILVAD, LASER, CLD, SLED/ASL/EMA ports; sensitivity weights; global+local; expected-accuracy window | all rejected with diagnoses; "no router" constraint held |
| §19–§20B | what the head does; causal masks of text→image attention per layer, prefix/suffix | the head *subtracts* late layers; **image→text transport completes by ~L16**; the answer token never reads the image directly |
| §21, §24–§25 | ridge vs GBT (coverage, end task, fold control), all-arms baseline table | **ridge ≡ GBT as a ranker, better on the end task**; beats LASER/ViCrop at equal compute on both models |
| §22–§23 | 14-cell scope analysis; equal-compute baseline table | **scope law**; fixed-layer cropping catastrophic; family worth ~2pp at equal compute |
| §26–§30 | transport-scheduled pruning, pyramid schedules, reinvesting savings in crop / localiser resolution | pruning at L16 free (2/2); no reinvestment converts on both models |

## 5. Results (accuracy = % correct on V*Bench, n=191, ~600 visual tokens; brackets = 95% bootstrap CI)
### 5.1 Headline table (§25, all arms in one run, ridge placements)
| pooled | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| bar: whole image @600 | 63.7 | 58.1 |
| fixed layer L14 (field convention) | **37.9** | **38.2** |
| block-mean argmax (ViCrop-style) | 62.1 | 59.2 |
| LASER (2602.04304) | 64.7 | 61.3 |
| gate-max (ours, label-free) | 68.9 | 60.7 |
| **ridge depth filter (ours)** | **72.6** | **70.2** |
| oracle placement | 87.9 | 87.4 |
ridge − LASER **+7.9 [+1.1,+14.7] / +8.9 [+3.1,+14.7]**; ridge − bar **+8.9 [+0.5,+17.4] / +12.0 [+4.7,+19.9]**;
ridge − tree −0.5 n.s. / **+7.3 [+2.1,+12.6]**; fold-seed control ±1pp. Single-object: 77.2 / 71.3 vs bar 62.3 / 57.4.
Replications with **tree** placements (ridge pending): Qwen2.5-VL-7B +14.8 ✔; HR-Bench 8K +10.5 ✔; HR-Bench 4K +7.5/+8.5 ✔
(single); Qwen3-VL-8B +4.3 n.s.
### 5.2 Coverage (intermediate; top-1 cell covers ≥50% of the box at W=0.25)
deployed block-mean argmax 46.6 / 39.3 → gate-max 56.0 / 51.8 → ridge/GBT 63.9 / 55.0 (LLaVA-NeXT 33, OneVision 41).
### 5.3 Efficiency (§26C)
Drop 90% of visual tokens at L16: 0.0–0.5pp change at every keep ratio (10/25/50%), any ranking, 300 and 600 tokens,
both models → ~35% of prefill compute. Reinvesting it: whole-image 900 tokens +6.3 ✔ (Qwen3) / +2.6 n.s. (Qwen2 — its
resolution curve is flat beyond 600); crop resolution 0/2; localiser resolution 0/2.

## 6. Findings (the mechanism)
1. **Encoding cliff.** Below ~0.25 merged tokens on the target, nothing downstream recovers the evidence; seven
   internal interventions are null; only re-encoding (cropping) works. Architecture-invariant (2 models).
2. **Attention is question-blind until ~half depth** (4 architectures); the layer where maps start to depend on the
   question is a label-free locator of the usable read-out band (L17–20 Qwen3, L19–22 Qwen2).
3. **Layers disagree and the mean cancels them.** The learned filter has both signs; it *subtracts* late layers that the
   block mean adds (target-above-picked-cell 13→57% / 12→71%). Fixed-layer read-out is catastrophic (24.6/33.9 on
   single-object — chance level on Qwen3). Layer-2 pruning is below random (2 models × 3 benchmarks).
4. **Image→text transport completes by ~L16** (peak L11 Qwen3, L14–15 Qwen2; prefix masks saturate at L12–16, suffix
   masks ≈0 from L16; 42–45% of answers flip when all text rows are blocked, 3% when only the answer token is).
   **Every method reads attention after the window closes**, on layers whose image attention is causally inert
   (KL ≈ 0.001) — the map is a residue of where the model looked, not the channel carrying the evidence. This explains
   §19, the null interventions, and why placement rules that all read the same inert depth separate only modestly.
5. **The useful read-out ends at L20** (exit there: −0.5/+1.6; exit at L16: −9/−22; pruning the localiser at L16
   destroys it). The late layers are noise for placement.
6. **Layer *selection* cannot work in VLMs.** Final-layer entropy rises on 54–88% of items (CLD's "alignment tax"
   signature, 16% in LLMs) yet lens accuracy climbs monotonically to the last layer; DoLa/CLD/LASER/ASL/EMA/SLED ports
   all lose. *Re-weighting* across depth works; *selecting* a depth never does.
7. **Scope law (14/14 cells: 3 benchmarks × 4 checkpoints × 2 strata).** Attention-guided cropping helps single-instance
   questions (+4 to +16, 7/7) and hurts cross-instance ones (−14 to 0, 7/7) — for every method in the family, because a
   tight crop removes the second object. Relational responds to pixels, not crops. Unreported in the literature because
   nobody stratifies.
8. **Equal compute deflates the family.** ViCrop-style block-mean cropping is −1.6/+1.0 vs the bar; LASER +1.1/+3.1;
   only the oracle clears it on both (+24/+29) besides the ridge.
9. **Read-out conditions.** The prompt must end at the answer-emission point; heads must be averaged before
   normalisation; extraction is bit-exact deterministic and precision-invariant; run-to-run floor 1–2.5pp (tree head ±3
   with fold assignment; ~2pp eager-vs-sdpa on the 7B).

## 7. Ideas tried and rejected (each logged with its diagnosis)
Sizer; span/mass/proportional/(cell,W)/expected-accuracy windows; global+local second pass; windowed features; box-free
and teacher heads; smooth low-rank depth kernel (rank truncation, not smoothness); sink-weighted layers; ILVAD depth
difference; LASER per-sample layer + query-contrast; CLD/entropy-valley (map and answer); Att-JSD; ASL rank stability;
depth-EMA; SLED extrapolation; RippleKV-style sensitivity weights (void: answer token is inert); CNN/deeper-tree/ensemble
heads (capacity is not the limit at n=191); attention rollout, grad×attention, ReAttn rescaling, VEA denoising, CoRe
contrastive heads (leaked in-sample); causal pyramid pruning; crop- and localiser-resolution reinvestment; TSR as an
accuracy method (1/2). Withdrawn diagnostics: "resolution deficit explains the scope law" (partial r = −0.27 after
controlling stratum); "the depth filter is high-frequency" (roughness penalty costs nothing).

## 8. Conclusions
- **Method:** a one-solve ridge depth filter over the 28-layer attention profile places crops better than the best
  published rules at equal compute on two models (+8–9 over LASER, +9–12 over the bar), with ~50 boxes.
- **Audit:** the field reads attention at a causally inert depth, evaluates without matching compute, and does not
  stratify; fixing all three shows cropping is worth ~2pp as practised and actively harms cross-instance questions.
- **Mechanism:** transport L6–L16, read-out band L17–L20, inert beyond; the late layers must be subtracted, not averaged;
  no single layer is better than the last for decoding; the cliff bounds everything.
- **Efficiency corollary:** 90% of visual tokens can be dropped at the transport boundary for free.
- **Limits stated:** relational unwon by any method (oracle 75–78 vs bar 60–66); Qwen3-VL-8B +4.3 n.s. (tree);
  OneVision has no layer disagreement; text tasks need their own depth; n=191 → CI half-widths ~5pp; needs boxes; 1.4–2.0×
  latency (sequential passes; early exit at L20 and KV-continuation untested at the end task).

## 9. Directions (ordered by expected value)
1. Re-run HR-Bench 4K/8K, Qwen2.5-VL-7B and Qwen3-VL-8B with **ridge** placements (all current breadth numbers use the tree).
2. Print w(ℓ), v(ℓ) on both models — the central figure; overlay the transport profile (§20B) and the gate band.
3. Third family with layer disagreement (InternVL3) for the method; relational needs a two-object primitive, which no
   crop provides — state as open.
4. Latency: early-exit localiser at L20 (−0.5 coverage, 25% less localiser compute) and KV-continuation of the crop.
5. Paper: method-led (§25) → audit (§22–23) → mechanism (§20B, §19, §18N) → efficiency corollary (§26C) → scope/limits;
   predecessors as "same conclusion": LASER, ILVAD (ICML'26), TRACE, ViRGo, VisLens, VEA, KLAL, CoRe, ReAttn, TwigVLM.

## 10. Details
- Data: V*Bench (191 boxed items: 115 direct_attributes, 76 relative_position), HR-Bench 4K/8K (800; single/cross),
  POPE, MMBench, TextVQA (3000). Grid ≈ 300 merged tokens; W=0.25 window; ring mask; option-letter log-prob scoring.
- Models: Qwen3-VL-2B-Instruct (core), Qwen2-VL-7B-Instruct (core), Qwen2.5-VL-7B, Qwen3-VL-8B, LLaVA-NeXT-7B,
  LLaVA-OneVision-7B. transformers 5.3.0, torch 2.11, bf16, eager attention where hooks/pruning are needed.
- Statistics: paired item bootstrap (6–8k resamples), GroupKFold(5) by item × 3 seeds for every learned component,
  fold-seed control for the headline, budget gate (>10% token drift voids a contrast).
- Pipeline lessons: exact +0.0 contrast = fault; head-averaging convention; answer-position prompt; never edit a running
  queue script; kill by PID from /proc; VRAM guard for a shared GPU; pre-specified automated gates between phases.
- Files: FINDINGS.md (§1–§30), REPLICATION_LEDGER.md, PAPER_FLOW.md, RESULTS_ATLAS.md, PRIOR_ART_VEA.md, scripts/phase*.py,
  data/phase*.jsonl (git-ignored), paper/main.tex (v14, pre-§17; to be restructured).
