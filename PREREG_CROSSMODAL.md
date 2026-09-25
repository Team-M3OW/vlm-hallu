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
