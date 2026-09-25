# Pre-registration: cross-modality transport, DWA and AVR (video, audio, VLA)

Written before any cross-modality run (2026-09-26). Standing rules apply: two-model rule for any
claim, negatives reported, exactly-zero contrast = pipeline fault, token layout must be measured
(not derived), budgets matched in token-layers where pruning changes cost.

## Hypotheses

- **H1 (transport).** In every modality, direct attention from answer-forming text positions to
  modality tokens becomes causally inert past a boundary at ~0.5–0.6 of decoder depth. If H1 fails,
  the 0.57 law is vision-encoder-specific — itself a major result.
- **H2 (free pruning).** Dropping 90% of modality tokens at the boundary costs ~0; moving the same
  cut one block earlier costs points.
- **H3 (scope law).** Selecting a modality region (crop) helps single-evidence questions and fails
  relational ones; relational needs coverage and is fixed by resolution/context, not by selection.
- **H4 (headroom).** Where a resolution/context ladder exists, the AVR-style gain equals measured
  headroom (slope ~1, r high); where no ladder exists, AVR is structurally inapplicable — recorded,
  never faked.

## Track V — video (Qwen3-VL-2B, no downloads)

Synthetic videos from V\*Bench images: `[target ×4, distractor ×4]`, 8 frames; Qwen3-VL merges 2
frames per temporal patch, so t=4. Video tokens = t·h·w/4.

- **V1** transport masking (phase176c protocol, text→video columns): per-layer, prefix, suffix →
  boundary fraction.
- **V2** scope: single (V\* question about the target half) vs relational ("which half shows the
  object"). Arms: uniform@~600 video tokens vs temporal crop (target half at full frame resolution,
  same total budget).
- **V3** prune 90% of video tokens at the boundary vs L4 vs L8.

Predictions: boundary ≈ L16 (0.57×28); V2 crop helps single, fails relational; V3 free at boundary.

## Track A — audio (Qwen2-Audio-7B-Instruct)

Synthetic clips with known event order (two utterances / two tones, ffmpeg); single (what was said /
what sound) vs relational (which came first). Transport masking text→audio columns; prune at
boundary; temporal crop (keep 25% of the clip at full frame rate, same total budget).

Predictions: H1–H4 as above. Risk: no resolution ladder for audio (AVR structurally inapplicable —
like InternVL/Gemma-3 in §80); then the result is "pruning is free but there is nothing to buy".

## Track R — VLA (SmolVLA via lerobot + LIBERO; OpenVLA-7B offline if needed)

- **R1** offline action transport: mask language→vision attention per layer on LIBERO demos, measure
  action-token KL / agreement → boundary fraction for action prediction.
- **R2** prune vision tokens at the boundary → action agreement, then closed-loop success in LIBERO.
- **R3** localisation: does attention at the boundary localise the manipulated object (DWA analog)?

Predictions: H1 for action tokens; pruning free at boundary; selection helps single-object pick,
fails relational tasks.

## Order and accounting

Start order: V1 (no downloads) → A1 (after model lands) → R1 (after checking SmolVLA hooks).
Everything budget-matched in visual/audio/video token-layers against a uniform bar; localise passes
charged in full, exactly as in the paper's image protocol.

## Handover notes (2026-09-26)

- Track V1 script `phase240_video_transport.py` is written and smoke-tested but hits a
  `StopIteration` in `Qwen3VLModel.get_rope_index` on `model(**inp)` with video inputs
  (`mm_token_type_ids` video group vs `video_grid_thw` iterator). The processor output itself is
  correct (`video_grid_thw` [[4,8,10]] for 8 frames with `do_sample_frames=False`; video tokens =
  t*h*w/4). Debug the model call before trusting any number from this script.
- Qwen2-Audio-7B-Instruct is downloading to `/media/kavinder/hdd2/hf_cache` (snapshot_download,
  started 2026-09-26).
- phase239 (crop+orig) is complete and logged as FINDINGS §87: negative, family rejected on Qwen3.

## Handover notes (2026-09-26, later) — Track V1 blocker diagnosed

`StopIteration` in `Qwen3VLModel.get_rope_index` is NOT a fault in `phase240_video_transport.py`.
Mechanism, verified with the processor alone (no model load):

```
video_grid_thw            [[4, 8, 8]]   -> 1 entry
mm_token_type_ids         4 contiguous runs of type 2 (video)
get_rope_index            grid_iters = {2: iter(video_grid_thw)}; next(...) once PER RUN
                          -> 4 next() calls against a 1-item iterator -> StopIteration
```

Qwen3-VL interleaves frame-TIMESTAMP text between video token groups (the processor warns
"Qwen3VL requires frame timestamps to construct prompts"), so the video span is split into one run
per temporal patch (t=4 here). `get_rope_index` expects one `video_grid_thw` row per run.

Two candidate fixes, untested (GPU was busy with Track A):
1. supply `video_metadata` (with fps) so processor and model agree on the grouping; or
2. expand the grid before the model call: `[[4,8,8]]` -> `[[1,8,8]]` x 4.
Fix (2) is a one-liner but must be verified to give the SAME video token count and a sane baseline
accuracy before any masking number from it is trusted.

## Track A1 deviation from the plan above (deliberate)

Using **MMAU** (TwinkStart/MMAU, 1000 items, 4-way MCQ) rather than synthetic ffmpeg clips:
a real public benchmark, already MCQ so the letter-logit scorer and flip metric port unchanged, and
it avoids the standing "public benchmarks only, do not build datasets" rule. Its own sub_categories
supply the strata the plan wanted from synthesis -- Temporal Reasoning / Temporal Event Reasoning /
Phonological Sequence Decoding / Counting / Emotion Flip (n=202) as the RELATIONAL analogue, and
Acoustic Source Inference / Sound-Based Event Recognition / Instrumentation (n=129) as the SINGLE
analogue. Synthetic clips remain the right instrument for H3 later, where controlled event order
matters.

Prefix/suffix stepped every 2 layers (not 4) because the pre-registered prediction is L18.2 and a
step of 4 would straddle it.

---

## Deviation log addendum — Route B (long-audio), 2026-09-26

**Scope decision (user, 2026-09-26): ONE MODEL PER MODALITY for the cross-modal tracks.**
The two-model rule ("make every claim on multiple models and if one doesn't survive reject that")
is therefore *not* met by Tracks A/V/R. These are demonstrations of transfer, and must be written
as such — the multi-model vision results keep the headline claims.

**Why Route B exists.** Track A on MMAU (10 s clips) returned: boundary CONFIRMED at 0.75 depth
(L24/32, invariant across music/sound/speech; per-layer peak L12 = 0.38 replicates vision's 0.39);
AVR cut-depth CONFIRMED (early@2 − avr@2 = −6.7 [−10.4,−3.3]); headroom FAILED (hi@2.0 − bar = −3.8)
and the headroom identity HELD EXACTLY, correctly predicting the null. DWA oracle ceiling FAILED
(crop_oracle − bar = −3.3 [−8.3,+2.2]) — placement has no room even with the label.
Diagnosis: time-stretch is a *token* knob, not a *resolution* knob. It adds no acoustic detail.

**RETRACTED claim.** An earlier note that "DWA requires ≥2 addressable axes" was presented as a
precondition derived from vision. It was not: vision never had fewer than two axes, so no
experiment tests it. It was inferred from the audio null post hoc. The ceiling failure is fully
explained by global-scope questions plus a 25% window of a 10 s clip. Route B is 1-D time also;
if its oracle ceiling is positive, the axes claim is falsified.

**H5 (pre-registered, Route B).** Past the encoder window a model must compress, so native encoding
recovers information compression destroyed — a real resolution knob. Predictions, in gate order:
- **G0** an ALM exists whose audio tokens grow linearly past 30 s. **RESULT: PASS.**
  Qwen2-Audio PLATEAUS at 750 tok (30 s window) → excluded. Qwen2.5-Omni LINEAR at 25 tok/s to
  6000 tok @ 240 s. Phi-4-MM processor-API error, AF3 not HF-loadable, Ultravox gated — not pursued
  (one model suffices under the scope decision). `data/phase245_longaudio_g0.json`.
- **G1** headroom `acc@native − acc@budget-B` > 0, CI-clear. FAIL ⇒ STOP, as on MMAU.
- **G2** DWA oracle ceiling `crop_oracle − bar` > 0 before any placer is written.
- **Boundary re-measure** at long length (suffix masks only): 0.75 is measured at ≤750 audio tokens
  and must not be assumed at 4.5k, since that is where AVR prunes.

**Bar construction is the design risk, pre-committed here.** `bar_trunc` (first 30 s) destroys
coverage and is the §81 straw bar plus a pipeline fault; it is DIAGNOSTIC ONLY and never the
comparison. The honest bars are `bar_voc` (phase-vocoder compression of the whole clip to B, pitch
preserved) and `bar_latent` (native encode, every k-th audio token visible via the 2-D attention
mask — same information budget, no out-of-distribution input). Headroom is taken against
**max(bar_voc, bar_latent)**. A `roundtrip` arm (compress ×r then stretch back) isolates vocoder
damage from token count, because a ×2 stretch alone cost ~4 points in Track A.

**Benchmark.** MMAU-Pro (`gamma-lab-umd/MMAU-Pro`, public, ungated, 5305 items), `long` +
`ultra-long` strata only (1219 items) — that stratum IS the experimental condition. Full split, no
sampling; items shuffled because MMAU-Pro is ordered by category (the CV-Bench/MMAU prefix trap).
Only genuinely open-ended items are dropped, counted and reported. Scope is stratified after the
fact by category; no router.
