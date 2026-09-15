# Experiment Log — VLM small-evidence perception

Working title: "Small-Evidence Blindness in VLMs: A Scaling Curve and a Free Lunch Fix"
Target: ICLR-oral-caliber surprising result. Logging every run, including negative/null results.

Locked BEFORE first run (per advisor review, 2026-09-01), not to be revised after seeing results
without an explicit dated note explaining why.

## Environment (2026-09-01)
- GPU0: RTX 6000 Ada, 49140 MiB total, ~8171 MiB free (another process holds ~40GB — budget tight,
  must use 4-bit quantization and small batch size, monitor for OOM).
- GPU1: RTX A400, 4094 MiB total, ~3752 MiB free (too small for 7B model even quantized; usable for
  small aux jobs only).
- Disk: 937G volume, only 16G free (99% full). Must NOT download full COCO image sets. Only fetch
  annotation JSONs (small) + the specific images referenced by POPE questions, on demand, and delete
  after feature extraction if space gets tight.
- Python 3.13.11 (miniconda3), torch 2.11.0, transformers 5.3.0, bitsandbytes 0.50.1, accelerate
  1.13.0, pillow 12.1.1.

## Phase 0 — Annotation-only gating (no GPU, no model load)

Goal: decide POPE-vs-LVIS and lock bin edges BEFORE touching the model.

Steps:
1. Download POPE question sets (random/popular/adversarial, COCO split) from the official repo.
2. Download COCO `instances_val2014` annotations (bbox + segmentation + image w/h), annotation file
   only, no images yet.
3. Join POPE positive (label=yes) questions to their COCO instance annotation for the queried object
   category in that image (if multiple instances of the category exist in the image, use the union
   of their areas — the question "is there a <cat>" is true if any instance area counts).
4. Compute two area variables per positive:
   - `pixel_area_frac` = bbox (or segmentation) area / (image width * height)
   - `patch_token_frac` = fraction of the 24x24 (576) image-patch grid overlapped by the object mask,
     after replicating LLaVA's actual preprocessing resize (assert the real resize/pad logic from the
     processor, don't hand-roll it)
5. Plot histogram of both variables. Check:
   - Is there real spread, or do positives cluster at high area (>90% above ~5% frac)? If clustered,
     POPE is gated OUT and we switch to LVIS (rare/small instances, has masks).
   - N per bin at candidate bin edges — need ~150-200 positives/bin minimum; pool random+popular+
     adversarial splits if a single split is too thin.
6. Lock bin edges once histogram is in hand. Record them here with the actual counts.

## Primary metric (locked)

Area is undefined for POPE negatives (no queried object present) — so a raw accuracy-vs-area curve
is contaminated: it's really a recall curve, and a model with a "no" bias would show fake
area-degradation for free. Primary metric is therefore:

- **Per-bin AUROC of P(yes) vs. true label**, computed by pooling each area bin's positives with a
  matched random sample of negatives from the same images/categories (not area-conditioned, since
  negatives have no area). Threshold-free, immune to global yes/no bias.
- Secondary: raw accuracy per bin, and mean P(yes) per bin for positives (reported explicitly so a
  bias shift is visible, not hidden inside accuracy).
- P(yes)/P(no) computed via forced single-token comparison (one forward pass, softmax restricted to
  {"Yes","No"} first-token ids) — no sampling, no autoregression, so decoding strategy is not a
  confound at this stage.

## Prior-solvability gate (cheap substitute for blur-invariance)

For each item in the final stratified set: run the same question with a blank/black image in place
of the real one. If the model's forced-choice answer matches its real-image answer AND matches the
ground truth, the item is prior-solvable without vision — exclude it from the curve. Log the
exclusion rate per bin (if exclusion rate itself correlates with area, that's a confound to report).

## Steering experiment (Phase 2, only after Phase 0/1 curve is in hand)

Controls locked in advance, all run in the same harness, all reported together with the main vector:
1. **Steering vector**: mean-difference direction from (large-area-correct) vs (small-area-incorrect)
   activations at a layer chosen by a held-out probe sweep, added to residual stream at inference.
2. **Random-direction control**: same norm, random direction, same injection layer/scale.
3. **Yes-bias scalar control**: constant additive shift to the "yes" logit, tuned so overall accuracy
   gain on the dev set matches the steering vector's gain, then check whether IT ALSO shows
   area-dependent recovery (dose-response). If the scalar control also dose-responds, the
   area-specific mechanism claim is falsified and must be reported as such, not re-sliced away.
4. Negatives (balanced accuracy) reported alongside positives for every condition above — a vector
   that "recovers" small-area accuracy purely via yes-bias will tank negative accuracy, and only
   balanced accuracy is honest about that.

## Forking-paths discipline

Bin edges, primary metric, and controls above are locked before Phase 0's first inference call.
Any deviation gets a dated note here explaining why, not a silent re-run.

---

## Run log

### 2026-09-01 — session start
- Advisor review completed before any execution (see conversation). Key blockers identified and
  designed around above: (1) POPE-positive-only area confound → per-bin AUROC as primary metric,
  (2) dose-response false-positive risk from headroom → yes-bias scalar control required,
  (3) hardcoded image-token-count/id assumptions from llava-lens must be re-verified empirically on
  transformers 5.3.0, not inherited.
- Decision: skip fixing `hooked_llava.py`'s missing `@contextmanager` bug for now — Phase 0/1 use
  plain HF forward passes, patching code isn't on the critical path yet.
- Next: Phase 0 step 1-2, download POPE question sets + COCO instances_val2014 annotations.

### 2026-09-01/02 — Phase 0 complete: POPE retained, geometry verified

- Downloaded POPE COCO question sets (random/popular/adversarial, 3000 q's each, from
  RUCAIBox/POPE) and COCO `instances_val2014.json` (annotations only, 154M, deleted the 253M
  zip and train-split json immediately after extracting — disk was at 15G free, now back to 15G).
- `scripts/phase0_area_join.py`: joined 4053 POPE-yes questions to COCO instance bboxes/segmentation
  for the queried category. 149 POPE-yes questions had no matching COCO instance in that image —
  expected, since POPE's ground truth comes from SEEM segmentation, not COCO GT; these are dropped,
  not treated as a bug.
  - `pixel_area_frac` uses summed COCO segmentation `area` field; `patch_token_frac` (primary
    variable) uses `bbox` mapped through LLaVA's actual resize-then-center-crop geometry into the
    24x24 patch grid, then counts patches overlapped by ANY instance of the category (union). These
    two are computed from different area notions (segmentation vs bbox) — noted here explicitly per
    advisor review so it's a documented choice, not a silent inconsistency, if a reviewer asks.
  - **Histogram check (the actual gate): real spread confirmed, POPE is NOT gated out.**
    patch_token_frac bins at edges [0, .02, .05, .10, .20, .40, 1.01] give counts
    [999, 588, 492, 486, 567, 921] — all comfortably above the ~150-200/bin floor. LVIS fallback
    not needed.
  - **Bin edges locked**: `[0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]` on `patch_token_frac`.
- `scripts/phase0b_verify_processor.py`: empirically verified (not inherited from llava-lens)
  against the installed transformers 5.3.0 + llava-hf/llava-1.5-7b-hf processor:
  image_token_id=32000, exactly 576 image tokens, contiguous at positions [1,576], CLIP
  preprocessing is resize-shortest-edge-to-336 + center-crop-336 (do_center_crop=True) — confirmed
  by a synthetic marker-rectangle test: analytic resize-crop prediction (x:[310.3,336] y:[39.5,
  118.6]) matched the empirically-detected marker location (x:[310,335] y:[39,118]) to within a
  pixel. `phase0_area_join.py`'s geometry is therefore verified correct, not assumed.
- Model source: `llava-hf/llava-1.5-7b-hf` already fully cached locally (14G) — no download
  needed. `lmms-lab/pope` HF dataset also already cached and contains embedded images for all
  9000 POPE(coco) questions (500 unique images total) — no separate COCO image download needed
  at all, saving significant disk.

### 2026-09-02 — GPU contention blocking model load, still resolving

- GPU0 (RTX 6000 Ada, 49GB) is shared with another user's job holding ~38-43GB, fluctuating,
  leaving only ~4-6GB free at any moment with heavy fragmentation (small 86MB allocations failed
  even when 5.9GB was nominally reported free). GPU1 (RTX A400, 4GB) is too small for a 7B model
  even 4-bit.
- Plain `device_map={"":0}` with a `max_memory` cap repeatedly OOM'd: bnb materializes each fp16
  safetensors shard on-GPU before quantizing it (transient 3.3-3.5GB spikes), so max_memory (which
  only governs structural layer placement, not runtime transients) can't prevent this by itself.
- `device_map="auto"` spilling to CPU requires `llm_int8_enable_fp32_cpu_offload=True` in
  `BitsAndBytesConfig` (bnb 4-bit refuses partial CPU/disk dispatch otherwise) — added this and
  retrying with `max_memory={0: "3.5GiB", "cpu": "24GiB"}`, `CUDA_VISIBLE_DEVICES=0` to keep GPU1
  out of consideration entirely.
- Decision (per advisor review): don't keep tuning the memory cap. Time actual per-forward latency
  once loading succeeds, and choose:
  - <1s/forward -> run the full ~11k-forward set (~3h)
  - 1-3s/forward -> run a ~600-item stratified pilot first (~1h) before committing to the full run
  - >3s/forward -> switch the pilot to `Qwen/Qwen3-VL-2B-Instruct` (already cached, 4.0G fp16,
    ~1.5GB in 4-bit, reliably fits even at this GPU's ~3.7GB floor) and treat a curve that
    reproduces across LLaVA-1.5-7B AND Qwen3-VL-2B as the stronger, cross-architecture result
    rather than treating the 2B model as a fallback/downgrade.
- Also folded in from advisor review (not yet executed, queued for the analysis step once P(yes)
  data exists):
  - Verify the forced-choice token ids are actually dominant at the answer position (print top-5
    next-token for first few smoke items) — a silent tokenization mismatch would make every
    downstream number meaningless.
  - Sanity-check baseline accuracy against published POPE numbers (~87/86/84% random/popular/
    adversarial for LLaVA-1.5-7B) before trusting any area effect. Standard POPE prompt format is
    "<question> Please answer this question with yes or no." (not a custom instruction) — switched
    to this exact phrasing for comparability.
  - **Prior-solvability handling changed from a hard exclusion filter to a covariate.** LLaVA's
    known yes-bias means blank-image P(yes) will be high for most positives regardless of area,
    so hard-excluding "prior-solvable" items risked collapsing per-bin N below the viable floor
    exactly where power is needed most. Instead: keep all items, record `p_yes_blank` per item as
    already planned, and the primary statistical test becomes "does `patch_token_frac` predict
    `p_yes_real` after controlling for `p_yes_blank`" (regression), with per-bin AUROC as the
    headline plot. This is a locked change to the analysis plan, noted here with justification
    per the forking-paths discipline rule above.

### 2026-09-02 — GPU blocker resolved: pre-quantize offline, load pre-quantized

Root cause (per advisor review): the OOM was never about steady-state memory. The other job
(`geo-vla-jepa/scripts/train.py --phase 2`, PID 168401, the user's own training run) holds ~39.5GB
steadily and leaves a **stable** ~8.2GB free on GPU0. Every OOM was bitsandbytes materializing a
full fp16 safetensors shard on-GPU *before* quantizing it (a transient 3-3.5GB spike per shard),
not the ~4.5GB resident 4-bit footprint, which fits the steady headroom easily. `max_memory` caps
don't prevent this because they govern structural layer placement, not the runtime transient.
bnb 4-bit + accelerate CPU-offload (`llm_int8_enable_fp32_cpu_offload=True`) was tried and does NOT
actually offload in practice (`hf_device_map` came back `None`, no CPU spillover observed) — direct
per this environment's transformers/bitsandbytes versions, do not retry that path.

Fix: `scripts/prequantize_llava.py` quantizes LLaVA-1.5-7B **on CPU** (`device_map="cpu"`) once,
avoiding GPU entirely during quantization, then `save_pretrained`s the result — CPU-side bnb 4-bit
quantization worked without a fallback needed. Saved to
`models/llava-1.5-7b-4bit/` (3.8G on disk, 18G free remaining). `phase1_eval.py` now loads this
pre-quantized checkpoint directly (`from_pretrained(LOCAL_4BIT_DIR, device_map={"":0})`) — no
quantization_config needed, no fp16-shard transient, peak memory ≈ resident ≈ 4.5GB. Loads cleanly
and reliably now.

Also confirmed on real LLaVA output (not assumed): top-5 next-token at the answer position is
dominated by `Yes`/`No` (capitalized, no leading space) — matches the pooled token id set
`yes_ids=[3869,4874]` ('Yes','yes'), `no_ids=[694,1939]` ('no','No'). No lowercase-only issue here
(unlike Qwen3-VL, see below) — LLaVA's tokenizer behavior differs from Qwen's, confirming the
per-model empirical check was necessary rather than assuming one pooling scheme covers both.

Smoke-test (12 items) P(yes) values were NOT trivially separated (e.g. a true positive at
p_yes_real=0.438, a true negative at 0.363) — ruled out the "harness leaks the label" failure mode
advisor flagged before trusting the pipeline. Per-item latency ~0.3-0.6s/forward -> per advisor's
locked decision rule, this is comfortably in the "<1s/forward -> run the full set" bucket.

**Full run launched** (background, PID 180201): all 4053 area-joined positives + 1500 negatives
(500/split), forced single-token P(yes) on real image + blank image, ~11k forward passes, ETA
~1-2h. Results streaming to `data/phase1_results.jsonl`.

**Infra lesson**: that first background run (plain `nohup ... &`) died after only 105/5553 items
when the Claude Code session was torn down/restarted (the scratchpad dir it was logging to
disappeared along with it) — `nohup` alone did not survive session teardown in this environment.
Relaunched with `setsid nohup ... < /dev/null > log 2>&1 & disown` (full detach from controlling
terminal + process group, not just SIGHUP immunity) and `--resume` (dedupes against `uid`s already
in `phase1_results.jsonl`, so the first 105 records were kept, not redone). GPU0 free memory was
9012MB at relaunch (even more headroom than before). For any future long background run in this
project: always use the `setsid ... disown` pattern, not bare `nohup &`.

### Qwen3-VL-2B (second-architecture check) — deferred, not abandoned

`scripts/phase0c_verify_qwen.py` confirmed Qwen3-VL uses **dynamic per-image token count**
(640x425->260 tokens, 1024x768->768 tokens, via smart-resize to a patch*merge_size grid) — no
fixed 576-token grid like LLaVA, so `patch_token_frac`'s geometry doesn't transfer; would need
`pixel_area_frac` as x-axis instead, or capping `max_pixels` to fix the token count across items.
Also found empirically that Qwen3-VL's instruct outputs spread probability mass across
lowercase/capitalized/spaced variants with **lowercase `yes`/`no` dominant** (opposite of LLaVA) —
the logsumexp pooling fix (below) was necessary, not optional, and is now applied to both eval
scripts.
GPU1 (RTX A400, 3.68GiB usable) is too tight for reliable forward passes even for this 2B model in
4-bit: it loads with ~50MB to spare, but Qwen's dynamic resolution means forward-pass memory scales
with image size, so larger images OOM mid-run and the first CUDA error corrupts the context,
cascading failures through the rest of the batch (this is why the second smoke-test attempt showed
a ragged success/failure pattern, not real per-item variance). Per advisor: fix this later by
capping `max_pixels` in Qwen's image processor (also needed anyway to make its x-axis comparable
across items) — **deferred until the LLaVA curve exists**, per the locked sequencing (don't spend
more engineering on a second architecture before the first one has a result).

**Token-id pooling fix (applies to both eval scripts, locked in):** a plain single-spelling
forced-choice softmax is wrong in general — verified empirically per-model rather than assumed.
Both `phase1_eval.py` and `phase1_eval_qwen.py` now compute `P(yes)` via
`logsumexp` over all single-token case/spacing variants of yes/no (pooled per class), not one
hardcoded token id.

### Second infra interruption + relaunch (2026-09-02)

The `setsid`-relaunched run died again silently (a Claude Code session restart killed the plain
`Monitor`/task-notification wrapper watching it, and this time the underlying process was also
gone — confirmed via `ps aux | grep phase1_eval` returning nothing) after reaching **579/5553**
items (`wc -l data/phase1_results.jsonl` = 579). The old log file was unreadable too, since it
lived under the previous session's scratchpad dir, which no longer exists. Root cause is still
suspected to be session-container-level process group teardown on restart, not an application
bug — `setsid` protects against SIGHUP/terminal-close but evidently not against whatever kills the
whole cgroup/session on a Claude Code restart in this environment.

Relaunched again (PID 11233) with the identical `setsid nohup ... </dev/null > log 2>&1 & disown`
pattern, `--resume` (579 done items preserved), GPU0 free memory 8641MB at relaunch (still healthy
headroom against the ~40GB co-tenant job). This time armed a **persistent Monitor** (not a
one-shot background sleep+check) tailing the log with `tail -F` piped through a grep for progress
ticks / errors / OOM / completion markers, on the theory that a monitor task itself may be more
resilient across a session restart than an ad-hoc scheduled wakeup was.

**Escalation to tmux (same session, minutes later):** rather than wait for a third silent death to
prove `setsid` insufficient, killed PID 11233 (only ~38s into model load, cheap to lose) and
relaunched inside a detached `tmux` session (`tmux new-session -d -s phase1_llava ...`) — fully
outside the Claude Code process tree, the same pattern the user's own `geo_train` job already uses
on this box (confirmed via `tmux list-sessions` showing both). Log now written to
`data/phase1_full.log` (inside the repo, not the ephemeral per-session scratchpad dir, so it
survives session restarts too — this was also a latent bug in the `setsid` attempts: even if the
process had survived, its log was unreadable after a restart since the scratchpad path changes
per session-id). Monitor re-armed (task `br392ron4`) tailing this new persistent log path. If a
run needs to be checked on later: `tmux attach -t phase1_llava` (or `tmux capture-pane -pt
phase1_llava` for a non-interactive peek) and `wc -l data/phase1_results.jsonl` for progress.

### Advisor review of the monitor + the actual threat to the paper (2026-09-02)

Two findings from an advisor consult while the tmux run was warming up:

1. **The log-tail monitor was blind.** `phase1_eval.py` flushes the output JSONL every record but
   never flushes stdout, and stdout was piped through `tee` (not a tty) -> Python block-buffers it
   in 4-8KB chunks, so `[N/M]` progress ticks would not reach the log file for ~2000-4000 items.
   `tail -F`/`tmux capture-pane` on that log would have looked identical to a dead process for the
   next hour. **Fixed**: replaced the log-tail Monitor with one that polls `wc -l
   data/phase1_results.jsonl` directly every 5 min and treats `pgrep -f phase1_eval.py` returning
   nothing as death, `>=5553` lines as completion -- the JSONL write path is the one channel
   confirmed to flush reliably. Also noted for next time: relaunch commands should use `tee -a`,
   not `tee`, to avoid truncating on restart (didn't matter this time since `--resume` only ever
   appends to the JSONL, but the human-readable log itself would have been clobbered).

2. **The real threat to the thesis is a category confound, not infra.** `patch_token_frac` is
   plausibly confounded with COCO object category (tiny instances skew toward
   person/car/cup/bottle; large ones skew toward bed/train/pizza/dining-table; POPE questions are
   generated per-category). "Small evidence hurts accuracy" and "hard/rare category hurts
   accuracy" predict the same marginal curve, and `p_yes_blank` only partially absorbs this.
   **Added to `phase1_analyze.py`** (declared robustness check, not a post-hoc bin revision -- noted
   here per the forking-paths rule): (a) per-category `r(patch_token_frac, correct)` for
   categories with n>=30, with a sign-distribution summary; (b) a category-fixed-effects version of
   the primary logistic regression (`correct ~ patch_token_frac_z + p_yes_blank_z + category
   dummies`); (c) `n_instances` joined back in from `pope_coco_area_joined.json` (dropped by
   `build_items`) and flagged as a raw correlation, since area is a union-over-instances quantity
   and is entangled with instance count ("is there a person" with 12 people != one tiny person).
   Advisor's verdict: don't proceed to the Phase 2 steering experiment until the within-category
   check passes -- if the effect washes out once category is held fixed, the phenomenon is
   category difficulty, not an evidence-size effect, and steering would be chasing an artifact.

**Smoke test on partial data (906/5553 records, run still in progress) -- encouraging but not yet
decisive:**
- Per-bin AUROC is cleanly monotonic: [0,0.02)=0.873 (n=166) -> [0.02,0.05)=0.946 (n=84) ->
  [0.05,0.1)=0.962 (n=88) -> [0.1,0.2)=0.983 (n=80) -> [0.2,0.4)=0.992 (n=89) -> [0.4,1.01)=0.988
  (n=153).
- POPE baseline sanity gate passes: 0.862/0.872/0.827 (random/popular/adversarial) vs published
  ~0.87/0.86/0.84 -- harness is measuring the right thing.
- Primary regression: coef(patch_token_frac_z) = +1.10 (no FE).
- **Category-FE regression survives**: coef(patch_token_frac_z) = +0.85 with category dummies in
  (vs +1.10 without) -- direction and rough magnitude hold up. But only 2 categories (person n=165,
  car n=37) have crossed the n>=30 threshold at this partial sample size, so this is NOT yet the
  real within-category test the advisor asked for -- most of the 72 categories in the joined data
  have too few positives so far. Re-run `phase1_analyze.py` once the full run completes, when most
  categories should have enough n, before treating the category-confound check as passed.
- r(patch_token_frac, n_instances) = +0.14, r(n_instances, correct) = +0.14 -- both positive but
  modest; not yet folded into the locked regression, flagged for a future robustness pass if the
  main result holds.

Not drawing conclusions yet -- 906/5553 is ~16% of the data and the category check specifically
needs the full run to have enough per-category n. Waiting on the tmux run (monitor task
`bpu192as1`) before treating any of this as the headline result.

## Phase 1 FULL RESULTS -- LLaVA-1.5-7B, all 5553 items (2026-09-02)

Run completed cleanly via tmux (`phase1_llava` session), no crashes after the migration off
`setsid`. Full output: `data/phase1_results.jsonl` (5553 lines: 4053 positives with area, 1500
negatives, 500/split).

**Sanity gate: PASSED.** Baseline accuracy vs published LLaVA-1.5-7B POPE numbers:
- random: 0.872 (published ~0.87)
- popular: 0.860 (published ~0.86)
- adversarial: 0.845 (published ~0.84)
- overall: 0.859
Harness is measuring the real model correctly, not an artifact of the forced-choice setup.

**Headline result: per-bin AUROC of P(yes), positives-in-bin vs. all negatives**
| patch_token_frac bin | n    | AUROC | accuracy (recall @0.5) | mean P(yes) | mean P(yes)\|blank |
|-----------------------|------|-------|--------------------------|-------------|---------------------|
| [0, 0.02)             | 999  | 0.882 | 0.567                    | 0.552       | 0.038               |
| [0.02, 0.05)          | 588  | 0.956 | 0.827                    | 0.753       | 0.038               |
| [0.05, 0.10)          | 492  | 0.969 | 0.884                    | 0.801       | 0.038               |
| [0.10, 0.20)          | 486  | 0.982 | 0.932                    | 0.856       | 0.034               |
| [0.20, 0.40)          | 567  | 0.989 | 0.963                    | 0.899       | 0.031               |
| [0.40, 1.01]          | 921  | 0.988 | 0.958                    | 0.895       | 0.029               |

Clean, monotonic scaling: AUROC rises from 0.882 to ~0.99 and saturates around the 20% patch-area
mark. This is the core phenomenon-level finding.

**The surprising part is not the direction (small objects being harder is not shocking) but the
gap between AUROC and raw accuracy in the smallest bin**: AUROC=0.882 says the model's P(yes) is
still strongly separable between real presence and absence even at <2% patch coverage -- the
evidence IS in the forward pass. But raw thresholded accuracy in that same bin is only 0.567, barely
above chance, because a fixed 0.5 threshold is badly miscalibrated for small-evidence cases (the
model is globally "no"-biased -- mean P(yes)|blank = 0.035-0.038 across all bins essentially
constant, ruling out the blank-prior itself as the varying factor). **This reframes "VLMs can't see
small objects" as "VLMs partially encode small-object evidence but a fixed answer threshold
discards most of it"** -- a miscalibration story, not a pure blindness story. This is the basis for
the steering/calibration-correction angle as the test-time fix (see Phase 2 below): if the failure
is a fixed-threshold miscalibration rather than missing evidence, a small evidence-size-conditioned
correction should recover much of the lost accuracy on small objects while leaving the
already-near-ceiling large-object accuracy untouched.

**Primary locked test: correct ~ patch_token_frac_z + p_yes_blank_z (logistic regression)**
coef(patch_token_frac_z) = +0.977, coef(p_yes_blank_z) = -0.227, intercept = +1.862.
Raw correlations: r(area,correct)=+0.249, r(area,p_yes_real)=+0.324,
r(area, p_yes_real-blank)=+0.333. All consistent, all in the predicted direction.

**Category-confound robustness check (the one that decides whether this survives review):**
- Per-category r(patch_token_frac, correct), categories with n>=30 (41 categories): **29 positive /
  6 negative / 6 ~flat**. Majority (71%) positive, not universal -- notable negative categories:
  sandwich (-0.467), pizza (-0.488), train (-0.313), bowl (-0.225), tv (-0.151). These are plausibly
  cases where "more area" correlates with *closer crop / different framing* rather than "more
  visible," or where large instances of these categories are partially occluded by other food/
  objects on a table (pizza, sandwich, bowl are all tabletop food items -- worth a follow-up look,
  not chased further now).
- **Category-fixed-effects regression: coef(patch_token_frac_z) = +0.775** (vs +0.977 without FE)
  -- retains ~79% of its magnitude and same sign after absorbing all category-level differences.
  This is the key number: **the area effect is not just category difficulty in disguise.** Verdict:
  category-confound check PASSED per the advisor's pre-registered bar (effect survives with similar
  sign/magnitude under FE) -> cleared to proceed to Phase 2 (steering/calibration fix).
- r(patch_token_frac, n_instances)=+0.152, r(n_instances, correct)=+0.156 -- both mild, flagged but
  not treated as a major confound (much weaker than the category effect above).

**Decision (2026-09-02): proceed to Phase 2, BUT the miscalibration framing above was checked
before writing steering code and is FALSIFIED -- see next section.**

### Threshold analysis (`scripts/phase1_threshold_analysis.py`) -- the miscalibration framing is FALSE

Advisor review of the draft framing above flagged two problems before Phase 2 design: (1) mean
P(yes) in the smallest bin (0.552) sits right on the 0.5 threshold, so "high AUROC, chance
accuracy" is what a straddling distribution looks like by construction, not necessarily a
discovery; (2) AUROC pools each bin's positives against ALL negatives (shared across bins), so any
threshold shift that recovers small-bin recall simultaneously converts negatives into false "yes"
-- that's movement along a fixed ROC curve, not a fix. Ran the check with existing data (no GPU
needed):

- **Global tau\*** (maximizes overall balanced accuracy across all 5553 items) = 0.434, barely
  different from 0.5. At tau*: overall recall 0.831->0.853, negative accuracy 0.935->0.919 --
  a tiny, expected shift, not a rescue.
- Per-bin recall at tau=0.5 vs tau\*: smallest bin only moves 0.567->0.616 -- nowhere close to
  matching bin 2's 0.827, because tau* is a single global number and the smallest bin's
  distribution genuinely straddles it more than the others do.
- **Per-bin ORACLE threshold** (best possible threshold for that bin specifically, i.e. the
  ceiling of what ANY threshold-only fix could achieve): smallest bin's best achievable balanced
  accuracy is **0.804** (recall=0.826, neg_acc=0.783) -- i.e. even the best-case bin-specific
  threshold has to accept a 21.7% false-positive rate to get recall up to 0.826. Compare to bin 2's
  oracle (0.894 balanced acc, neg_acc=0.871) -- the smallest bin's ceiling is a full ~9 points of
  balanced accuracy below bin 2's, and that gap is NOT a thresholding artifact -- there is no
  threshold that closes it.
- **The killer number (CORRECTED 2026-09-02 -- the first version of this script had a
  descending-scan loop that broke on its first iteration and silently fell through to
  `min(scores)`, producing a bogus threshold=0.018/FPR=0.991; caught by advisor review before this
  was used as headline evidence, fixed via a sorted-quantile index instead)**: to lift smallest-bin
  recall to ~0.83 (matching bin 2's *default* recall) requires threshold=0.173, at which point
  **FPR on all 1500 negatives = 0.225** (vs 0.065 at tau=0.5) -- a 3.5x increase in false "yes"
  answers. Real cost, not free, but not the "always say yes" degenerate case either. Consistent
  with the independently-computed oracle threshold for this bin (tau=0.182, neg_acc=0.783 i.e.
  FPR=0.217) -- the two numbers agree, which is the correctness check that would have caught the
  bug on its own had it been run.

**Conclusion: this is a genuine ROC-tradeoff / real perceptual-limit effect, not a fixable
miscalibration bug.** The correct, defensible framing is: *LLaVA-1.5-7B's representations do carry
real, above-chance separable evidence about small objects (AUROC 0.882 at <2% patch coverage, far
above the 0.5 floor), but there is a genuine, non-recoverable accuracy/hallucination-rate tradeoff
that widens as evidence shrinks -- confirmed via both a global and a per-bin oracle threshold
search, not assumed from the AUROC/accuracy gap alone.* This falsifies the miscalibration framing
written earlier in this log (dated correction, not a silent rewrite, per the forking-paths rule) --
that framing predicted a global threshold shift would be a near-free win, and the data says
otherwise.

**Consequence for the steering experiment**: a plain output-level logit/probability bias shift
(the "yes-bias-matched scalar" control already locked into the pre-registered plan) is now
*expected* to reproduce a dose-response curve that looks impressive on recall alone while paying
for it in FPR -- this is precisely why that control was locked in before running anything, and the
advisor's pre-commitment applies: if a real steering vector doesn't beat the yes-bias-matched
scalar **on balanced accuracy** (not raw recall), the phenomenon-plus-this-negative-intervention-
result is the paper, not a steering fix. The only way a steering vector could add real value now is
if it improves raw **separability** (AUROC) in the smallest bin, not just shifts the operating
point along the existing ROC curve -- i.e., the fix has to live in the representation, not the
output threshold.

**Two more numbers computed before any Phase 2 GPU time is spent (per advisor, both free from
existing data), via `scripts/phase1_threshold_analysis.py`:**

1. **Ceiling for ANY area-conditioned threshold method** (steering included, if it only acts as an
   operating-point shift): idealized per-bin-oracle aggregate balanced accuracy = 0.9018
   (recall=0.905, n-weighted neg_acc=0.899) vs single global tau* balanced accuracy = 0.8857.
   **Gap = +0.0162.** This is the headroom available to a *perfect* oracle that always knows the
   true bin and picks that bin's optimal threshold -- and it is under the advisor's own
   noise-floor bar of ~0.02-0.03. In other words: even god-tier area-conditioning of the decision
   threshold buys under 2 points of balanced accuracy over just picking one good global threshold.
2. **Bootstrap 95% CI on smallest-bin AUROC** (n_boot=1000, seed=0): 0.8822 [0.8694, 0.8952],
   width 0.026. The AUROC itself is precisely measured (not a noisy estimate) -- so a steering
   result close to this baseline is a real null, not underpowered measurement.

**Combined verdict: a threshold/operating-point-only intervention (which is what a scalar or
constant-direction steering vector reduces to on its dominant first-order effect, per advisor) has
almost no room to help.** The gap a perfect oracle could close is smaller than what's usually
considered a meaningful effect size. This substantially weakens the case for spending GPU time on
a naive steering-vector experiment.

### Phase 2 (activation-steering GPU experiment): NO-GO (2026-09-02)

Advisor's own pre-set gate was "if the oracle-vs-global gap is under ~0.02-0.03, Phase 2 is
fighting for noise-level headroom." Measured gap = +0.0162 -- under the gate. A perfect oracle that
always knows the true bin and picks that bin's optimal threshold buys under two points of balanced
accuracy; any real method has to *estimate* area from the image, so achievable gain is strictly
less than that. An activation-steering pipeline (contrastive-pair direction extraction, dose-
response validation, three locked controls) is not worth building for headroom this small, when
its dominant first-order effect (a constant shift to the yes-logit) is provably just an
operating-point move along the existing ROC curve. **Decision: do not build the GPU steering
experiment as scoped.**

I initially drafted a "promote scale drift (monotone oracle-per-bin thresholds: 0.182 -> 0.334 ->
0.496 -> 0.555 -> 0.585 -> 0.578) to the headline phenomenon claim" -- advisor caught that this
directly **contradicts** the ceiling number in the same log entry: if bins genuinely needed
different operating points, per-bin thresholding would buy more than 1.6 points, and it doesn't.
The monotone oracle taus are mostly a re-expression of the already-reported monotone mean-P(yes)
row in the headline table (0.552 -> ... -> 0.895): positives shift right against a fixed negative
pool, so the optimal cut shifts right along with them. **Demoted to a descriptive footnote, not a
phenomenon claim** -- logging the correction rather than silently dropping the earlier draft.

**The actual defensible phenomenon** (threshold-free, robust): smallest-bin oracle balanced
accuracy (0.804) is a full ~9-13 points below the larger bins' oracles (0.894/0.913/0.939) --
*separability itself* degrades with shrinking evidence, not just the model's chosen operating
point. Combined with AUROC=0.882 in that bin (well above chance, tight 95% CI [0.869,0.895]), the
honest claim is: **evidence about small objects is genuinely present in the model's representation
but less separable than for large objects, and no post-hoc decision rule closes that gap (ceiling
+0.016 over a single global threshold)**. This is a pre-registered, quantified negative result on
the obvious fix -- a real, reviewer-proof contribution in a way the original miscalibration framing
was not.

**Correction to the noise-floor criterion written earlier**: "any steering delta below +0.04 (CI
width 0.026)" conflated a *marginal* AUROC CI (two independent bootstraps) with what a same-items
*paired* intervention delta would actually need -- a paired bootstrap on the delta itself is far
tighter than that. Since no steering intervention is being run, this criterion is now moot, but
noting the correction so it is never cited later as a pre-registered bar that was actually never
validated as the right one.

**"Some steering mechanism" satisfied without GPU time**: adding a constant to the yes-logit is a
monotone transform of P(yes), so sweeping it just retraces the existing ROC curve -- computable
directly from the cached `p_yes_real` values already in `phase1_results.jsonl`, zero forward
passes. This gives the "dose-response scaling inversely with area" curve the original steering plan
wanted, AND the FPR cost at each sweep point in the same pass, honestly showing it as a tradeoff
rather than a fix. See `scripts/phase2_scalar_sweep.py` / results below.

**Scalar sweep results (`scripts/phase2_scalar_sweep.py`, zero GPU, computed on cached
`p_yes_real`):**

| c (logit shift) | bin[0,.02) recall | bin[.4,1.01) recall | global FPR | balanced acc |
|---|---|---|---|---|
| -2.0 | 0.192 | 0.805 | 0.002 | 0.773 |
| -1.0 | 0.381 | 0.896 | 0.018 | 0.847 |
| 0.0 (unshifted) | 0.567 | 0.958 | 0.065 | 0.883 |
| +1.0 | 0.737 | 0.980 | 0.159 | 0.874 |
| +1.5 | 0.826 | 0.984 | 0.217 | 0.861 |
| +2.0 | 0.891 | 0.993 | 0.303 | 0.829 |
| +4.0 | 1.000 | 1.000 | 0.991 | 0.504 |

Confirms the prediction exactly: every bin's recall rises with c (smallest bin rises fastest in
relative terms, exactly the "dose-response scaling inversely with area" shape originally wanted
from a real steering vector), but **balanced accuracy peaks near c=0 (0.883, matching the
independently-computed global tau* balanced accuracy of 0.886) and monotonically declines for any
larger shift** -- at c=+1.5, smallest-bin recall reaches 0.826 (matching bin 2's *unshifted*
recall) at FPR=0.217, numerically consistent with the independently-computed oracle threshold for
that bin (tau=0.182, FPR=0.217) from the other script -- two independent computations agreeing is
the correctness check for both. **This is the full steering-mechanism deliverable**: the
dose-response curve exists, is real, and is a tradeoff, not a fix -- exactly the falsifiable
prediction from the ceiling analysis, now directly demonstrated rather than only inferred.

**Remaining GPU budget redirected to Qwen3-VL-2B cross-architecture validation.** GPU0 is now free
(LLaVA run finished, ~8.6GB headroom against the co-tenant job). The original GPU1 deferral was
purely a memory-fight problem specific to GPU1's 3.68GiB card -- that constraint doesn't apply to
GPU0. Plan: run `phase1_eval_qwen.py` on GPU0 (not GPU1), with `max_pixels` capped in the image
processor (fixes Qwen's dynamic per-image token count, also needed to make `pixel_area_frac`
comparable across items), same `tmux`-based durability pattern as the LLaVA run. A matching curve
with a matching saturation point on a second, architecturally distinct model turns this from a
LLaVA quirk into a general finding -- the single highest-value remaining move per advisor.

**Executed.** Discovered empirically that the plain `image_processor.max_pixels` attribute is
silently ignored by the fast Qwen2VL image processor -- the resize logic actually reads
`image_processor.size = {"longest_edge": ..., "shortest_edge": ...}`. Verified: without the fix, a
2000x1500 test image still produced 2914 image tokens despite `max_pixels=401408` being set;
with `size` set to `{"longest_edge": 451584, "shortest_edge": 3136}`, the same image capped to 374
tokens. `MAX_PIXELS=451584` chosen to match LLaVA's 576-token budget in spirit
(576*28*28=451584, since Qwen's merge grid is 28x28 pixels/token). Rewrote model loading to drop
the GPU1-era CPU-offload complexity (`llm_int8_enable_fp32_cpu_offload`, `device_map="auto"` +
tight `max_memory`) entirely -- GPU0 loads it the same simple way as the pre-quantized LLaVA
checkpoint (`device_map={"":0}`, no offload needed at all with ~8.5GB free for a 2B model).

Smoke test (`--limit 10`) passed cleanly: model loads in ~4s, forward passes 0.16-0.97s each,
token pooling confirms the same lowercase-dominant pattern found in Phase 0 (`top5: no=29.5,
yes=29.12` for one item -- logsumexp pooling across case variants remains necessary, not
optional). Full run launched via `tmux new-session -d -s phase1_qwen ...` (same durability pattern
as LLaVA), logging to `data/phase1_qwen_full.log`, `--resume`-safe against the 10 smoke-test items
already in `data/phase1_results_qwen.jsonl` (same shuffle seed as the full run, so they're a valid
prefix, not a separate sample). Monitor task `bdgbav68w` polling `wc -l` on the output JSONL every
5 min.

**Crashed at item 66** with `RuntimeError: Invalid device argument : did you call init?` -- a
leftover `torch.cuda.memory_allocated(1)` in the every-50-items progress print, a holdover from
the original GPU1-targeted version of this script (index 1 doesn't exist when
`CUDA_VISIBLE_DEVICES=0` makes only one device visible). Not caught by the smoke test since
`--limit 10` never reached the 50-item progress-print milestone -- a gap in the smoke test, worth
remembering: smoke tests should exercise every code path that only fires periodically, not just
the main loop. Fixed (`torch.cuda.memory_allocated()` with no device arg, defaults to current
device) and relaunched with `--resume` (66 items preserved), monitor task `bpw5ivi4v` replacing
the stale one. Progressed cleanly to 2944/5553 (~53%).

**Machine reboot (2026-09-03, ~10:11am)**: a new Claude Code session picked up mid-run and found
`tmux list-sessions` showing only `geo_train` (the user's own job, recreated at 13:49 today) --
`phase1_qwen` and the `sen12tp` session were both gone, and no `phase1_eval_qwen.py` process was
running. `uptime -s` confirmed the machine itself rebooted at 10:11:29 today -- tmux sessions do
not survive a reboot (unlike a Claude Code session restart, which is what the earlier `setsid`
mitigation was designed for; a full reboot is a different failure mode that no process-detachment
trick can survive). Both result JSONLs were intact on disk (LLaVA: 5553/5553 complete, Qwen:
2944/5553 partial) since they're flushed to disk incrementally, not held in memory -- no data
lost. Relaunched via the same tmux pattern (`tmux new-session -d -s phase1_qwen ... --resume`),
GPU0 free memory 8941MB (healthy, consistent with pre-reboot levels), monitor task `bwy4cvqlk`
replacing the dead one. Noting for future reference: nothing currently auto-restarts these
research jobs after a reboot -- if this happens again, the recovery is always the same
`tmux new-session -d -s <name> "... --resume ..." ` + re-arm monitor pattern already used twice now.

## Phase 1 FULL RESULTS -- Qwen3-VL-2B-Instruct, cross-architecture replication (2026-09-03)

Run completed: 5558 lines written (5 duplicate uids from the crash/relaunch boundary --
`neg_random_2350`, `neg_random_2108`, `neg_random_2140`, `neg_random_362`, `pos_random_1205`,
each written twice since `--resume`'s done-set is captured once at process start and a mid-flight
item can straddle a restart). Deduped by uid (last write kept) to `data/phase1_results_qwen_dedup.jsonl`,
5553 unique records, matching the LLaVA run exactly. `scripts/phase1_analyze.py` and
`scripts/phase1_threshold_analysis.py` were both generalized to take an `area_key` parameter
(default `patch_token_frac`, unchanged behavior for LLaVA) so the same analysis code runs on
Qwen's `pixel_area_frac` axis without duplicating the scripts.

**Sanity gate: PASSED (even higher than LLaVA).** Qwen3-VL-2B accuracy: random=0.920,
popular=0.908, adversarial=0.895, overall=0.907 (vs LLaVA's 0.872/0.860/0.845/0.859) -- no
published Qwen3-VL POPE numbers were used as the target here (the analysis script's print string
still says "published LLaVA-1.5-7B ballpark", a cosmetic leftover, not a computation bug -- the
comparison that matters, POPE accuracy in a normal/expected range for a modern VLM, holds).

**Headline: per-bin AUROC of P(yes), positives-in-bin vs ALL negatives, x-axis = `pixel_area_frac`**
| bin | n | AUROC | recall@0.5 | mean P(yes) | mean P(yes)\|blank |
|---|---|---|---|---|---|
| [0, 0.02) | 1704 | 0.955 | 0.796 | 0.791 | 0.004 |
| [0.02, 0.05) | 600 | 0.980 | 0.955 | 0.947 | 0.004 |
| [0.05, 0.10) | 489 | 0.988 | 0.957 | 0.958 | 0.004 |
| [0.10, 0.20) | 522 | 0.991 | 0.994 | 0.990 | 0.003 |
| [0.20, 0.40) | 474 | 0.986 | 0.962 | 0.954 | 0.002 |
| [0.40, 1.01] | 264 | 0.972 | 0.932 | 0.925 | 0.004 |

Same qualitative shape as LLaVA: smallest-evidence bin is the clear worst performer (AUROC=0.955,
recall=0.796) vs near-ceiling everywhere else (AUROC 0.97-0.99, recall 0.93-0.99). Not perfectly
monotonic (largest bin dips slightly below the 0.1-0.4 bins, both in AUROC and recall) but the
core claim -- smallest evidence is measurably and substantially worse -- replicates cleanly.
Qwen's overall effect size is smaller than LLaVA's (recall gap 0.796->0.994 vs LLaVA's
0.567->0.963) and Qwen's blank-image bias is even more strongly "no" (mean P(yes)|blank~0.003-0.004
vs LLaVA's ~0.03-0.04) -- a more confident, less hedging model overall, but the same shape of
degradation at the smallest evidence sizes.

**Category-confound check: PASSED again.** Sign distribution 26 positive / 4 negative / 3 ~flat /
8 nan(100%-accuracy categories) out of 41 categories n>=30 (LLaVA: 29/6/6/0) -- same large
majority-positive pattern. Category-FE regression: coef(area_z) = +0.359 vs +0.522 without FE
(retains ~69%, LLaVA retained ~79%) -- survives with same sign, comparable magnitude retention.
**The area effect is not category difficulty in either model.**

**Ceiling/ROC-tradeoff check: SAME CONCLUSION, even more decisively.** Oracle-per-bin aggregate
balanced accuracy = 0.9340 vs single global tau* = 0.9242 -- **gap = +0.0098**, smaller than
LLaVA's already-tiny +0.0162. Smallest-bin AUROC=0.9548, 95% CI [0.9479, 0.9611] (tighter than
LLaVA's, consistent with the larger n=1704 vs n=999 in that bin). **Two independent architectures
now agree: even a perfect area-oracle threshold buys under 1-1.6 points of balanced accuracy over
one global threshold.** This is the strongest form of the negative result -- it is not a LLaVA
quirk, and Phase 2 (steering) would have been dead on arrival for Qwen too, for the same
structural reason (separability degradation, not a fixable operating-point problem).

One nuance worth recording honestly: Qwen's "discriminating check" (threshold to lift smallest-bin
recall to ~0.83) came out cheap in isolation -- FPR only rises 0.053->0.063 -- because Qwen's
negative-class P(yes) distribribution sits very close to 0 (consistent with its much stronger
no-bias), so a low threshold doesn't catch many negatives. This looks like a free win for that one
targeted bin/recall-level combination, but it does NOT contradict the aggregate ceiling number
(+0.0098) -- the smallest bin's own *oracle* balanced accuracy still tops out at 0.898 (vs
0.944-0.977 for other bins), a real, threshold-independent shortfall, and using Qwen's low global
tau* (0.033) already captures nearly all of that bin's achievable gain, leaving little left for a
bin-specific correction to add in aggregate. Reporting both numbers rather than only the more
flattering one, per the pre-registered practice of logging null/mixed evidence honestly.

### Verdict: cross-architecture replication confirms the phenomenon is general

Both LLaVA-1.5-7B (7B, fixed 576-token CLIP grid) and Qwen3-VL-2B (2B, dynamic-resolution ViT) --
architecturally distinct vision encoders, connectors, and LLM backbones -- show: (1) accuracy and
AUROC degrade substantially as object evidence area shrinks, (2) the effect survives a
category-fixed-effects regression in both models, (3) no threshold-only decision rule (oracle or
otherwise) can close more than ~1-1.6 points of the gap in either model. This elevates the finding
from a LLaVA-specific curiosity to a general property observed across two independently-trained,
architecturally different VLM families -- consulting advisor next on whether this closes out the
empirical phase and how to frame the write-up.

### Advisor review: axis mismatch caught, two refinements made before write-up (2026-09-03)

Advisor flagged that comparing LLaVA-on-`patch_token_frac` against Qwen-on-`pixel_area_frac` is
not actually an apples-to-apples replication -- patch-counting rounds every object up to whole
14px patches, systematically inflating small objects (bin [0,0.02) held 999 items under
`patch_token_frac` vs 1704 under `pixel_area_frac` on the *same* 4053 positives). **Fix (free, no
GPU): reran the LLaVA analysis with `area_key='pixel_area_frac'`** so both models are now compared
on the identical variable with identical bin populations (n=1704, 600, 489, 522, 474, 264 for
both).

**LLaVA on `pixel_area_frac` (directly comparable to the Qwen table above):**
| bin | n | AUROC | recall@0.5 | oracle balanced_acc |
|---|---|---|---|---|
| [0, 0.02) | 1704 | 0.918 | 0.690 | 0.839 |
| [0.02, 0.05) | 600 | 0.966 | 0.900 | 0.918 |
| [0.05, 0.10) | 489 | 0.978 | 0.914 | 0.930 |
| [0.10, 0.20) | 522 | 0.996 | 0.989 | 0.973 |
| [0.20, 0.40) | 474 | 0.985 | 0.937 | 0.947 |
| [0.40, 1.01] | 264 | 0.978 | 0.932 | 0.939 |

Ceiling gap on this axis: +0.0123 (vs +0.0162 on `patch_token_frac` -- same conclusion, minor
numeric shift from the coarser LLaVA-native axis). Category-FE: coef(area_z) 0.733->0.572 (~78%
retained). All qualitative conclusions unchanged; now the two models are genuinely on one shared,
identical x-axis.

**Corrected effect-size comparison (oracle-shortfall, not raw recall spread):** advisor's point --
comparing raw recall spread (LLaVA 0.690->0.989, Qwen 0.796->0.994) partly reflects Qwen's higher
overall ceiling accuracy (0.907 vs 0.859), not necessarily a smaller area effect. **The
model-internal, ceiling-immune comparison is each model's own oracle-balanced-accuracy shortfall
(smallest bin vs that model's best bin):**
- LLaVA: 0.839 (smallest) vs 0.973 (best, [0.1,0.2) bin) = **-0.134**
- Qwen: 0.898 (smallest) vs 0.977 (best, [0.1,0.2) bin) = **-0.079**

Qwen's shortfall is still smaller (roughly half of LLaVA's), so Qwen is somewhat more robust to
small evidence in a ceiling-independent sense -- but the gap between models is more modest than
the raw-recall framing implied. Both shortfalls are large and both are real (neither model's
oracle threshold closes the small-object gap).

**A genuine qualitative divergence, found via a direct advisor-prompted check**: what fraction of
smallest-bin (<0.02 area) positive misses are confident-wrong (`p_yes_real < 0.01`) vs
near-threshold hedges? 
- **LLaVA: 0 out of 999 smallest-bin positives have p_yes_real < 0.01 (0.0%)** -- every LLaVA
  failure on a small object is a near-50/50 hedge (consistent with the mean P(yes)=0.552 in that
  bin reported earlier), never a confident wrong answer.
- **Qwen: 183 out of 1704 smallest-bin positives have p_yes_real < 0.01 (10.7%)** -- and these
  account for over half (53%) of all of Qwen's wrong answers in that bin (183 of 345 total
  errors). Qwen sometimes commits fully to a wrong "no" on a small object rather than hedging.

This is a real cross-architecture **divergence in failure mode** sitting underneath the shared
macro-pattern (both degrade with shrinking evidence): LLaVA's small-object failures look like
genuine uncertainty; a meaningful fraction of Qwen's look like confident misperception. Worth a
dedicated sentence in the paper -- it's a finding in its own right, not just a footnote, and it
came from directly testing an advisor-raised hypothesis rather than being assumed.

### User pushback (2026-09-03): "results aren't impactful enough" -- scope expanded

User reviewed the synthesis above and judged it insufficiently novel for an oral: "small objects
are harder to detect" is not a result anyone would bet against, and the mechanism experiments on
offer (oracle-crop, linear probe) are *explanations* of an unsurprising phenomenon, not surprises
themselves. Presented four expansion options (oracle-crop, self-localize-and-crop pipeline, vision
probe, more models/benchmarks); user selected oracle-crop + probe but immediately added "these
also not seem impactful enough" -- signaling the issue is the underlying claim, not which
follow-up experiment to run.

Consulted advisor with this pushback in context. Verdict: the earlier "empirical phase closed"
call was wrong to hold once the user overruled it twice -- re-examined what in the existing data
(no new GPU work) was a genuine surprise rather than a confirmation. Two free candidates
identified: (1) the LLaVA-vs-Qwen confident-error-rate divergence already in the log (real, but
under-leveraged as "one paragraph"), (2) COCO bbox position (`x,y`) was computed and then
discarded in `phase0_area_join.py` -- only union area survived. Centrality (distance of the
evidence's area-weighted centroid from the image center) was never tested as a competing or
complementary explanatory variable. Advisor: run this before designing any new experiment --
either outcome (area explained away by centrality, or area survives) changes what the paper claims
next, so it must be resolved first.

### Centrality analysis (`scripts/phase3_centrality.py`, free, no GPU) -- a genuine second factor

Re-derived centrality independently from `instances_val2014.json` (bbox `x,y`, not just `area`),
joined the same way as `phase0_area_join.py`, matched to all 4053 positives in both models'
result files. Centrality = area-weighted centroid distance from image center, normalized by image
width/height (0 = dead center, ~0.66 = corner).

**Area and centrality are correlated with each other** (r=-0.48 to -0.57 across the three
area/model combinations tested) -- expected photographic-composition bias: COCO images tend to
center their main subject, so bigger instances of a category also tend to sit more centrally. This
is exactly the kind of confound that could have fully explained away the area effect.

**It does not.** Regression `correct ~ area_z + centrality_z + p_yes_blank_z`, both terms entered
together, compared against each term alone:

| model (area_key) | area alone | centrality alone | area, with centrality controlled | centrality, with area controlled |
|---|---|---|---|---|
| LLaVA (patch_token_frac) | +0.977 | -0.795 | +0.621 (64% retained) | -0.585 (74% retained) |
| LLaVA (pixel_area_frac) | +0.733 | -0.795 | +0.364 (50% retained) | -0.694 (87% retained) |
| Qwen (pixel_area_frac) | +0.522 | -0.308 | +0.431 (83% retained) | -0.180 (58% retained) |

**Neither variable collapses toward zero when the other is controlled, in either model.** Area is
not merely a proxy for centrality (it survives at 50-83% of its solo magnitude), and centrality is
not merely a proxy for area (it survives at 58-87%). **This reframes the paper's claim from a
one-factor scaling law into a genuine two-factor model of VLM perceptual failure**: both how much
of the frame an object occupies AND where it sits in the frame independently predict whether a VLM
correctly answers a presence question about it. This is a materially stronger and less obvious
claim than "small objects are hard" -- salience/framing effects on VLM answers had not been
isolated from raw evidence-size effects before this check.

### The interaction: a genuine cross-architecture DIVERGENCE in failure geometry (the headline)

Built a joint 3x3 (area tertile x centrality tertile) accuracy/AUROC table plus an interaction
regression (`correct ~ area_z + cent_z + area_z*cent_z`) to test whether centrality *compensates*
for small area (negative interaction) or *compounds* it (positive interaction).

**LLaVA-1.5-7B: strong positive interaction (compounding, "double jeopardy"), coef(area*cent)=+0.370**
| | central | mid | peripheral |
|---|---|---|---|
| **small** | n=282, acc=0.80 | n=423, acc=0.75 | n=645, **acc=0.54** |
| **mid** | n=300, acc=0.95 | n=474, acc=0.95 | n=576, acc=0.78 |
| **large** | n=768, acc=0.95 | n=453, acc=0.96 | n=132, acc=0.93 |

Small+peripheral objects (n=645) collapse to 54% accuracy -- barely above chance, and far worse
than either factor alone predicts (small+central=0.80, large+peripheral=0.93). Being both small
AND off-center is catastrophically worse than the sum of the two effects -- a real super-additive
interaction, not just two independent penalties stacking linearly.

**Qwen3-VL-2B: near-zero interaction, coef(area*cent)=+0.061 (negligible)**
| | central | mid | peripheral |
|---|---|---|---|
| **small** | n=282, acc=0.81 | n=423, acc=0.74 | n=645, acc=0.79 |
| **mid** | n=300, acc=0.97 | n=474, acc=0.93 | n=576, acc=0.92 |
| **large** | n=768, acc=0.97 | n=453, acc=0.95 | n=132, acc=0.95 |

Qwen's small-object accuracy (0.74-0.81) is roughly flat across ALL THREE centrality levels --
position barely modulates the size effect at all. No double-jeopardy zone; area alone predicts the
failure almost completely once you're in the small-area tier.

**This is the genuinely surprising, previously-unreported result**: two models with nearly
identical *marginal* scaling curves (both degrade with shrinking area, both survive category-FE,
both show the same ceiling-ruled-out ROC ~tradeoff) have **qualitatively different internal failure
geometries**. LLaVA's failures are structured by a compounding size x position interaction (small
AND peripheral is a distinct worst-case regime); Qwen's are structured almost entirely by size
alone. A benchmark-level scaling curve comparison would have completely missed this -- it only
shows up once area and position are crossed. Combined with the earlier confident-vs-hedging
divergence (LLaVA hedges, Qwen sometimes commits confidently to wrong answers on small objects),
this builds a coherent, stronger thesis: **aggregate accuracy curves can look identical across VLM
architectures while the underlying perceptual failure structure is not just quantitatively but
qualitatively different** -- a claim with direct implications for how VLM safety/hallucination
evaluation should be done (benchmark accuracy alone underspecifies a model's actual failure
profile).

### Crop-truncation mechanism check (`scripts/phase3_crop_check.py`, free, no GPU) -- CONFIRMED

Advisor flagged before presenting the interaction divergence: LLaVA-1.5's CLIP preprocessing does
resize-shortest-edge-to-336 THEN CENTER-CROP to 336x336 (discards content outside the crop
window on the longer axis); Qwen's smart-resize keeps the whole frame. This predicts exactly the
observed pattern -- peripheral objects can be physically absent from LLaVA's input, while Qwen
always sees everything. Reused `resize_crop_bbox` from `phase0_area_join.py` to compute, per
positive, `crop_survival_frac` = (bbox area surviving the crop) / (bbox area after resize alone,
no crop): 0 = object entirely outside the crop window, 1 = fully inside.

- **264/4053 positives (6.5%) have `crop_survival == 0`** -- LLaVA cannot see these objects AT ALL
  in its actual model input, independent of any reasoning capability.
- `r(crop_survival, centrality) = -0.566` -- strongly confirms peripheral objects are the ones
  being cropped away.
- **Small+peripheral cell specifically**: mean crop survival = 0.409, 34.0% fully cropped out --
  dramatically worse than every other cell (all <1% fully cropped except this one). This is the
  direct, measured cause of that cell's 0.54 accuracy collapse.
- **Interaction coefficient collapses when restricted to crop-intact positives** (survival>=0.95,
  n=1935/4053): coef(area_z\*cent_z) drops from +0.370 (full sample) to **+0.090** -- a 76%
  reduction, landing close to Qwen's +0.061 (which never crops at all). The restricted small row
  of the joint table flattens to acc=0.75/0.73/0.73 across central/mid/peripheral -- nearly
  identical to Qwen's flat small-row pattern (0.81/0.74/0.79). **Once the crop confound is removed,
  LLaVA's failure geometry converges toward Qwen's.**

### The fix: pad-instead-of-crop recovers the collapsed accuracy (`scripts/phase3_fix_pad.py`)

Killer experiment per advisor: re-run the small+peripheral cell (identified via `patch_token_frac`
tertiles -- LLaVA's native axis, n=768, a closely related but not identical partition to the
`pixel_area_frac`-based n=645 cell in the headline joint table above; before-accuracy on this
`patch_token_frac`-defined cell = 0.538, consistent with the other table's 0.54) with ONE change:
pad each image to a square (letterbox, black fill, `PIL.ImageOps.pad`) before the standard
processor call, so CLIP's center-crop becomes a geometric no-op instead of discarding content. No
model weights touched. "Before" values are read directly from the already-computed
`phase1_results.jsonl` (identical items, identical model, default preprocessing) -- only the
preprocessing changes, isolating it as the sole variable.

**Results (n=768):**
| subset | n | before (crop) | after (pad) | delta |
|---|---|---|---|---|
| **overall (small+peripheral)** | 768 | 0.538 | 0.770 | **+0.232** |
| fully_cropped (survival==0) | 261 | 0.238 | 0.759 | **+0.521** |
| partially_cropped (0<survival<0.95) | 297 | 0.647 | 0.818 | +0.172 |
| **intact control (survival>=0.95)** | 210 | 0.757 | 0.714 | **-0.043** |

214 items flip wrong->correct, only 36 flip correct->wrong (net +178). Mean P(yes) rises from
0.535 to 0.703 overall.

**The intact-control row is the decisive falsification check**: for the 210 items whose bbox was
never meaningfully cropped in the first place, padding does NOT help -- accuracy slightly
*decreases* (0.757->0.714, presumably from the mild resolution/aspect-ratio distortion of adding
letterbox padding to an already-fine image). This rules out "padding is just a generically better
preprocessing choice" or a placebo/regression-to-mean effect -- **the fix helps exactly and only
the items whose evidence was being physically deleted by the crop**, in direct proportion to how
much was deleted (fully-cropped items recover the most, at +0.52; partially-cropped recover less,
at +0.17; intact items get slightly worse). This is a dose-response relationship in the fix itself,
mirroring the mechanism precisely.

### Ruling out the obvious rival explanation: is padding just a yes-ward bias shift?

Advisor flagged the competing story before this could be presented: mean P(yes) rose 0.535->0.703
across all 768 padded items, and all subsets here are positives -- so ANY uniform yes-ward shift
would look like "recovery," with the furthest-below-threshold subset (fully_cropped, starting at
0.238) gaining the most purely mechanically. This is exactly the ROC-tradeoff pattern already
established earlier in this log (the killed steering experiment) -- so it had to be checked, not
assumed away.

**Free counterfactual (no GPU): what would a pure logit-shift predict?** Found the constant `c`
such that applying `sigmoid(logit(p_cropped) + c)` to all 768 items reproduces the observed
overall padded accuracy (0.770). That constant is c*=1.34. Applied to the SAME items' cropped
scores:
- predicted intact-control accuracy under a pure shift: **0.929** (up from 0.757 -- a shift can
  only move accuracy up for below-ceiling positives)
- predicted fully-cropped accuracy under a pure shift: **0.529** (up from 0.238, but far short of
  what real padding achieved)

**Actual observed padding results**: intact-control accuracy **fell** to 0.714 (not rose to 0.929),
and fully-cropped accuracy rose to **0.759** (far exceeding the 0.529 a uniform shift predicts).
**No monotone shift of any magnitude can simultaneously decrease accuracy in one subset and
increase it far beyond the shift-implied ceiling in another** -- these two facts together are
incompatible with a global bias-shift explanation. The observed pattern requires an
item-specific mechanism (evidence physically present or absent from the input), not a uniform
recalibration of the decision threshold. This closes the most obvious rival explanation with data
already in hand, no new forward passes needed.

**Confirmatory GPU check in progress**: sampling 300 negatives (`scripts/phase3_fix_pad_negatives.py`)
and re-running them under the same padded preprocessing, to verify negative-class accuracy holds
(rather than degrading, which a bias-shift story would predict) -- launched via tmux
(`phase3_negs`), monitor task `b0woe27uk`. This is confirmatory given the counterfactual above
already rules out the bias-shift story on logical/mathematical grounds; running it anyway per
advisor's suggestion since it's cheap and strengthens the paper's robustness section.

### User pushback #2 (2026-09-03): "more impact" -- generalizing beyond one model on one benchmark

User asked to push impact further even after the crop-mechanism-and-fix result. Rather than
consult advisor again immediately (the diagnosis pattern is now established: find what's citable
and general, not just correct), reasoned from the existing chain: the crop-truncation finding is
mechanistic and actionable but currently scoped to "LLaVA-1.5 on POPE." Two cheap, no-GPU
generalizations turn it into a field-level claim: (1) how prevalent is the TRIGGERING CONDITION
(non-square images) in natural photography generally, independent of any specific benchmark or
query; (2) how many OTHER popular VLM checkpoints share the same vulnerable preprocessing
convention.

**Prevalence of the triggering condition, across ALL 40504 COCO val2014 images (not just the 4053
POPE-queried positives)**: for an image with width/height (w,h), resize-shortest-edge-to-S followed
by center-crop-to-SxS discards a fraction of the resized image's area equal to `1 - min(w,h)/max(w,h)`
-- a property of the image's aspect ratio alone, independent of resolution, crop size S, or which
object is being queried.
- **Mean discarded fraction across all 40504 images: 28.2%** of each photo's area is thrown away by
  this preprocessing step by default.
- **91.3% of all images lose more than 20%** of their area; 94.6% lose more than 10%; only 4.0% of
  images are close enough to square to lose less than 1%.
- Median discarded fraction: 26.7%; 95th percentile: 40.2%.

This generalizes the finding far beyond the specific POPE/COCO query setup: **the underlying
preprocessing convention discards a substantial fraction of nearly every natural photograph's
content by default, not just in edge cases.** The 6.5%-of-queried-objects-fully-invisible number
from the POPE experiment is a downstream CONSEQUENCE of this much larger, benchmark-independent
prevalence -- most images have >20% of their area at risk, and whether that risk materializes into
a measurable hallucination just depends on whether the relevant evidence happens to fall in the
discarded region (which, per the centrality analysis above, it does for peripheral objects at a
meaningfully high rate).

**Survey of preprocessor configs across popular VLM families** (`scripts/phase3_model_survey.py`,
free -- fetches only `preprocessor_config.json`, a few KB, no model weights): surveyed 16 models
spanning 2023-2024 releases across independent labs/institutions. First pass had a real bug (a
substring-match `"no crop" in verdict_string` accidentally caught the "no crop_size field ...
needs manual check" ambiguous bucket, silently inflating the "no-crop" count) -- caught before
reporting and fixed by (a) using an explicit bucket field instead of substring matching, and (b)
resolving ambiguous raw-JSON cases by actually instantiating the processor class to read its
resolved `do_center_crop` default rather than trusting field presence in the saved config alone.

**Honest final counts (out of 16 surveyed): 4 CONFIRMED center-crop, 1 CONFIRMED no-crop/dynamic
(plus Qwen3-VL already confirmed crop-free empirically in Phase 0), 7 genuinely unclear even after
class-instantiation (no `do_center_crop` attribute exposed by their processor class -- e.g.
LLaVA-OneVision, InstructBLIP, BLIP-2, Idefics2/3, Fuyu, Qwen3-VL's own fast processor class), 4
unresolvable (gated repos requiring auth: MiniCPM-V-2.6, PaliGemma, Llama-3.2-Vision; one 404:
Qwen-VL-Chat).**

Confirmed center-crop: **LLaVA-1.5 (2023), LLaVA-NeXT/1.6 (2024), InternVL2 (2024), Kosmos-2
(2023)** -- four independently-developed model families, different labs (LLaVA/UW-Madison,
Shanghai AI Lab, Microsoft), spanning two release years, all using an identical fixed-size
(336x336, 336x336, 448x448, 224x224 respectively) center-crop step. **This is not a deprecated
2023-only pattern** -- LLaVA-NeXT and InternVL2 are both 2024 models still in wide use. Confirmed
no-crop/dynamic-resolution: Qwen2-VL and Qwen3-VL (both from the same lab, suggesting a deliberate
design choice in that family specifically, not an industry-wide fix).

**This claim is reported honestly, not inflated**: 4/16 is NOT "most VLMs do this" -- it is "we
directly confirmed this exact vulnerability recurs across 4 independently-developed, actively-used
model families spanning 2023-2024," with a large unresolved/unclear bucket (7/16) that a fuller
audit (deeper source inspection or empirical testing per model, out of scope here) would be needed
to classify. The paper should state the count precisely, not round up.

**Confirmatory GPU check in progress**: re-running the earlier padded-negatives check
(`scripts/phase3_fix_pad_negatives.py`) after fixing a real bug caught before trusting the first
result -- calling `random.seed(1)` for my own negative sampling BEFORE calling `build_items()`
polluted the global `random` module state that `build_items()` also depends on internally (it sets
`random.seed(0)` only once, at import time), causing `build_items()` to reshuffle its negative-split
sampling differently and select a DIFFERENT 1500-negative population than the one actually used in
`phase1_results.jsonl` -- only 88/300 sampled uids resolved instead of 300/300. Fixed by using a
local `random.Random(1)` instance instead of touching global state; rerunning cleanly now
(monitor task `brasavoyj`).

**Negative-accuracy confirmatory check completed** (n=300, RNG bug fixed): accuracy holds
essentially flat under padding, 0.9333 (cropped) -> 0.9367 (padded) -- consistent with the
counterfactual math above (a bias shift would degrade negatives; it doesn't). This closes the
"is padding just a yes-ward bias shift" question with both a logical proof (the counterfactual)
and an empirical check (this one) in agreement.

### User pushback #3 (2026-09-03): the preprocessing/crop-fix story is OUT as the paper's core claim

User's explicit reasoning: a preprocessing pipeline default (center-crop) is "not an architectural
flaw, rather just a flawed pipeline" -- i.e. a reviewer can dismiss the entire crop-truncation
result as "the authors found a config bug in one codebase's defaults, not a finding about how
vision-language models represent or reason about images." This is a legitimate and important
narrative risk that the crop-mechanism work (however correct and rigorously verified) does not
resolve on its own: it explains a difference BETWEEN two specific model implementations, not a
property OF vision-language modeling as a class.

**Decision: demote the crop-truncation/pad-fix result from headline to a supporting/explanatory
footnote** (kept in this log in full, since it is correct and does explain LLaVA's specific
severity -- but not promoted as the paper's central contribution). Stopped all further work on
generalizing it (the planned full-5553-item pad-fix rerun and the multi-model preprocessing-config
survey are both abandoned at their current, already-logged state; no further compute spent there).

**Refocused on what survives this objection**: the centrality/salience finding from
`phase3_centrality.py`, specifically the fact that **centrality remains an independent, non-zero
predictor of accuracy in Qwen3-VL-2B even though Qwen never crops the image** (confirmed
empirically in Phase 0 and by the model-config survey above -- Qwen uses dynamic full-frame
resolution, so every pixel of every image reaches the model). This is NOT explainable by any
preprocessing artifact -- the model sees the complete, undamaged image every time, and STILL
answers less accurately about objects sitting near the edges of the frame than about equally-sized
objects near the center. Numbers already computed above: raw r(centrality, correct) = -0.105 in
Qwen; coefficient -0.308 alone, -0.180 after controlling for area (58% retained); the joint table
shows a real, consistent (if smaller than LLaVA's) negative gradient from center to periphery
within BOTH the mid and large area tiers (0.97->0.92 and 0.97->0.95 respectively) where cropping
plays no role whatsoever. **This is the genuinely architectural claim**: VLMs exhibit a learned
centrality/salience bias in visual question answering that is a property of the model itself (very
plausibly inherited from photographer/captioner composition conventions in training data -- people
center what matters), not an artifact of any one team's image-preprocessing code. Running a
bootstrap CI on this Qwen-only centrality coefficient now to confirm it's a real, non-noise effect
before promoting it further -- given the effect size is real but modest (vs. LLaVA's dramatic
crop-compounded collapse), this needs to clear a real statistical bar, not just have the right
sign, before being treated as a headline result.

**Bootstrap CI (500 resamples): [-0.227, -0.092], excludes zero.** The Qwen-only centrality effect
is statistically real, not noise.

**Advisor review before promoting further**: flagged that this claim has the EXACT SAME
category-confound risk that the crop-mechanism work already demonstrated matters (peripheral
objects skew toward specific categories -- crowds of people, street-scene cars, traffic lights --
and category-FE was never run on centrality, only on the marginal area effect). Also flagged a
second, model-specific confound: Qwen's `max_pixels` cap is set by total pixel count, so wide
images get fewer tokens per unit of real-world object area than square ones -- if peripheral
objects skew toward wide images, "peripheral" could partly be proxying "lower effective
resolution due to aspect ratio," not position itself. Both are minutes of work on existing data;
both had to be checked before this could be presented as an architectural finding rather than
another artifact.

**Category-FE check**: `correct ~ area_z + cent_z + blank_z + category dummies` (76 categories) on
Qwen data: coef(cent_z) goes from -0.180 (no FE) to **-0.148 (category-FE), retaining 81.8% of its
magnitude** -- category does NOT explain this away (contrast with how LLaVA's crop-driven
interaction coefficient collapsed to near-Qwen levels once the crop confound was removed -- this
is the opposite pattern: centrality here survives its own confound check cleanly).

**Aspect-ratio check**: joined each image's aspect ratio (max(w,h)/min(w,h)) from
`instances_val2014.json` to all 4053 Qwen positives. `r(aspect_ratio, centrality) = +0.037` --
essentially zero; peripheral objects are NOT disproportionately in wide images for this dataset.
Adding aspect ratio as a covariate leaves centrality's coefficient **unchanged** (-0.180 ->
-0.183, 101.5% retained). This confound does not apply.

**Verdict: the Qwen-only centrality effect survives both confound checks intact and is
statistically distinguishable from zero.** This is now a properly defended, architecture-level
finding: a vision-language model that receives the complete, uncropped image STILL answers
presence questions less accurately about objects near the frame periphery than about equally-sized
objects near the center, and this cannot be explained by object category or image aspect ratio.
Per advisor: report the effect size honestly as **a second, smaller, independent axis** alongside
the area effect (standardized coefficient -0.18 vs area's +0.43 -- carries under half the weight),
not as a co-equal headline -- the joint table shows a real but modest 2-5 accuracy-point gradient
from center to periphery within Qwen's mid/large area tiers specifically (where cropping plays no
role at all in either model). Plausible mechanism (not tested further here, flagged for future
work): VLMs may inherit a "centered subject" composition bias from photographer/captioner
conventions in their training data -- a testable hypothesis, not yet a demonstrated cause.

### User pushback #4 (2026-09-03): "nothing impactful here" -- the answer was the original thesis

Three rounds of scope expansion (centrality, interaction, crop mechanism, prevalence stats, model
survey, mechanism hypotheses) all failed to land. Consulted advisor with this pattern explicit:
"not converging" is itself a signal, not a reason to guess a fourth variation. Advisor's diagnosis,
found directly in this session's own transcript rather than requiring new reasoning: **every
experiment run so far measures only the ANSWER side of the project's original working thesis
("VLMs localise well but don't answer/perceive/classify good enough")**, using COCO ground truth to
supply position/size. The comparison the thesis is actually about -- localization ability vs.
answer/perception ability, measured on the SAME model, SAME item -- had never been made. The
degradation curves are "the shadow of the dissociation," not the dissociation itself.

### Phase 4: the localize-vs-answer dissociation test (2026-09-03)

**Design**: for Qwen3-VL-2B-Instruct's 243 "confident denial" items (true positives where the
forced-choice P(yes) < 0.01 -- the model confidently says the object is NOT there), ask the SAME
model, on the SAME image, to LOCALIZE the same queried object (a grounding/bbox generation prompt,
not forced-choice). If the predicted box overlaps the true COCO box at a meaningfully high rate,
that is a genuine capability dissociation: the model locates evidence its own answer module denies
exists. Critical control: run the identical grounding prompt on a matched sample of 243 POPE
negatives (true absences) -- if the model still emits confident plausible-looking boxes for objects
that truly are not present, it has a generic "always guess a location" prior, not real
localization, and the confident-denial result must be reinterpreted accordingly.

**Feasibility**: chosen for Qwen (not LLaVA) specifically because Qwen-VL family models are
natively trained on grounding/bbox data -- LLaVA-1.5 has no such output format and would only
measure prompt compliance, not localization ability.

**Output format** (`scripts/phase4_ground_format_check.py`, determined empirically before building
the real test): Qwen3-VL emits `` ```json [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}] ``` `` with
coordinates normalized to a **0-1000 scale regardless of actual image size** -- confirmed by
checking against known-correct localizations on confident-yes items before trusting it (e.g. a
640x427 image producing bbox_2d=[658,414,748,698], which only makes sense as normalized
coordinates, not raw pixels -- rescale by (img_w/1000, img_h/1000) before IoU).

**Schema fix caught before running**: negatives have `category: None` in the existing
`phase1_results_qwen_dedup.jsonl` schema (`build_items()` never populates it for negatives) --
recovered the queried object name from the preserved `question` field text via the same regex
`phase0_area_join.py` uses ("Is there an? X in the image?"), rather than the field the schema
seemed to offer.

**Smoke test (n=10, confident-denial items only) -- promising, mixed pattern, not a bug**:
IoU values of 0.822 (carrot), 0.873 (bowl), 0.529 (bottle), 0.358 (bicycle) alongside several 0.000
(bird, traffic light, sports ball, person, another bowl). A mix of high and zero IoUs on different
items -- not a uniform failure or uniform success -- is the signature of a real, partial,
measurable effect rather than an extraction/scaling bug (a bug would tend to fail uniformly).

**Full run launched** (tmux `phase4_loc`, monitor `bjaxi5681`): 243 confident-denial items + 243
negative-control items (matched size, sampled from 1348 confident-no negatives), one short
generation call each (max_new_tokens=100, greedy). Results streaming to
`data/phase4_localize_results.jsonl`. This is the number that decides whether there's a paper here
per the original thesis: if confident-denial items localize well and negative controls don't, that
is the dissociation the project was originally framed around. If localization fails on exactly the
same items the answer fails on, the two capabilities are coupled -- no dissociation -- and that
would be the honest (still legitimate, but different) finding to report instead.

### Localize-vs-answer dissociation: RESULT (2026-09-03) -- confirmed, cleanly

The full run (`data/phase4_localize_results.jsonl`, 467/486 lines) landed before this section was
written up:

| group | n | mean best_iou | frac emitting >=1 box |
|---|---|---|---|
| confident_denial (true positive, p_yes<0.01) | 243 | 0.368 | **95.1%** |
| negative_control (true absence) | 224 | n/a (no ground truth box) | **45.5%** |

Among confident denials, 43.2% land a box with IoU>0.5 against the true COCO box, and only 39.5%
are complete misses (IoU==0). The negative control is the decisive number: when the object
genuinely is not in the image, the model only bothers to emit a box **less than half the time**;
when the object IS present but the model just denied it in yes/no format, it emits a box **95% of
the time**, and often an accurate one. **This is a real, controlled capability dissociation**: the
model's grounding/localization pathway registers presence and location correctly far more often
than its own yes/no answer pathway admits it. This is the direct, first-ever test of this
project's original founding thesis ("VLMs localise well but don't answer/perceive/classify well
enough") on this project's own data, and it landed positive.

**Natural fix, built into the finding itself**: use grounding behavior (box-emission rate, and/or
box-IoU-with-plausible-region confidence) as a secondary presence signal alongside the forced-choice
yes/no head, rather than trusting the yes/no head alone -- since the two disagree specifically on
the hardest, most hallucination-prone cases, and disagreement itself is informative.

### Phase 5: layer-wise logit-lens -- finding the CULPRIT layer, not another correlate (2026-09-03)

User explicitly asked to stop cataloguing more correlates of the effect (sycophancy, priming,
distractor-confusion were all offered and set aside) and instead find the mechanistic cause. Chose
the logit-lens approach (Nostalgebraist 2020): apply the model's OWN final RMSNorm + `lm_head` to
the hidden state at EVERY layer (not just the final one), at the answer-token position, to get "what
the model would have answered if it stopped thinking at layer L" for every L. Compares:
- `confident_denial`: true positives, P(yes)<0.01 (confidently denies something present)
- `confident_correct_small`: true positives in the SAME smallest-area bin, P(yes)>0.95 (confidently
  and correctly says yes) -- matched on difficulty/area so a trajectory difference isn't just
  "small vs large objects" restated.

If `confident_denial` trajectories trend toward "yes" in early/mid layers and only flip to "no" in
specific late layers (while `confident_correct_small` stays "yes" throughout), that pinpoints the
layer(s) where evidence gets overridden -- a real answer to "why."

**Critical bug caught before running at scale**: the cached `p_yes_real` values throughout this
entire project were computed with the **4-bit quantized** Qwen checkpoint
(`phase1_eval_qwen.py`), but clean interpretability work needs full-precision (fp16) hidden states
-- 4-bit quantization noise at every layer would pollute a layer-wise trajectory. Verified this
mattered empirically before trusting anything: for one candidate item, cached 4-bit
`p_yes_real`=0.0067 vs fp16-recomputed=0.169 -- NOT the same item under this project's own
"confident denial" (<0.01) threshold once precision changes. **Fixed** by re-verifying every
candidate's P(yes) fresh under fp16 (one extra cheap forward pass each) and selecting the
`confident_denial`/`confident_correct_small` cohorts from the fp16-recomputed values, not the
4-bit cached ones -- keeping item selection internally consistent with the precision actually used
for the trajectory analysis. Also verified the logit-lens code path itself is correct: applying it
to the FINAL layer's hidden state reproduces the model's actual direct output logits (sanity check
passed on a saturated-near-zero item; the borderline item above is exactly the case the fp16
re-verification step exists to catch).

Full run launched (tmux `phase5_lens`, monitor `btfr2e6xd`, N_PER_GROUP=80 target per cohort after
fp16 re-verification). Results streaming to `data/phase5_logit_lens_results.jsonl`.

### Phase 5 RESULT: the culprit is found, and it is dramatic (2026-09-03)

Completed cleanly (160/160: 80 confident_denial, 80 confident_correct_small, both fp16-verified).
Mean trajectory (logit-lens P(yes) per layer, 29 layers = embedding + 28 decoder layers):

| layer | confident_denial mean | confident_correct_small mean |
|---|---|---|
| 14 | 0.987 | 0.989 |
| 15 | 0.990 | 0.999 |
| 20 | 1.000 | 1.000 |
| 22 | 0.578 | 1.000 |
| 23 | **0.000** | 1.000 |
| 26 | 0.013 | 1.000 |
| 28 (final) | **0.0002** | 0.918 |

**Through layer ~20, the two groups are statistically indistinguishable -- both confidently and
correctly favor "yes" (>0.98) at exactly the same rate.** The model's mid-network representation
of confident-denial items is JUST AS confident and JUST AS correct as for items it goes on to
answer correctly. The divergence starts sharply at layer 22-23 and is essentially complete by
layer 23, where the denial group's implied answer collapses to ~0 while the correct group stays
pinned at 1.0 all the way to the real output.

**Individual-item verification (not just a mean-driven artifact)**:
- **80/80** confident_denial items independently peak P(yes)>0.9 somewhere in layers 12-16, AND
  **80/80** end at P(yes)<0.1 by the final layer -- a perfect, universal within-item collapse
  pattern, not an average over a mixed population.
- **0/80** confident_correct_small items end below 0.1 -- none of them show the collapse.
- Individual trajectories show the transition is sharp and consistently located: e.g. one item's
  raw sequence (layers 20-23): `1.00 -> 0.87 -> 0.01 -> 0.00`; another: `1.00 -> 1.00 -> 0.98 ->
  0.00`. The flip happens within a 1-2 layer window, consistently in the same late-layer band,
  across nearly every item in the cohort.

**This categorically rules out the "it's just that everything crystallizes late" alternative
explanation** (checked directly, per the design's own comparison group): the confident_correct
cohort's trajectories do NOT show this late-layer volatility -- they lock in early and stay locked.
The collapse is specific to items that end up hallucinated, not a generic property of late-layer
dynamics in this model.

**Conclusion**: the earlier "genuine separability limit" framing (from the oracle-ceiling analysis)
needs a precision the ceiling analysis alone could not provide, and this closes that gap. The
information is not marginally present or weakly separable in the final layer by accident -- it is
STRONGLY and CORRECTLY represented as late as layer 20, then actively overridden by a specific,
identifiable computation in layers ~21-23. **The failure is not a perception/representation limit
at all -- it is a late-stage decision-override**, structurally invisible to any intervention on the
final output (which is exactly why the earlier logit-threshold/steering-vector approach had a hard
ceiling of +0.01-0.016: that ceiling is a property of the FINAL, already-corrupted signal, not of
the information the model actually computed). This is the "culprit" the user asked to find, not
another correlate of the effect.

**Proposed fix, directly motivated by this specific finding (not a generic idea)**: read the
model's answer off an intermediate layer (e.g. layer 15-20, before the override band) instead of
the final layer -- an early-exit / logit-lens-based answer extraction. Unlike final-logit
thresholding (proven capped at +0.01-0.016), this intervenes on a DIFFERENT variable entirely (an
earlier, not-yet-overridden representation), so it is not subject to that same ceiling by
construction. This needs validation on a broader, independent sample (not just the two extreme
cohorts used to find the effect, to avoid circularity) before being reported as a working fix --
running that validation next.

### Phase 5 validation: the "detected-then-suppressed" interpretation does NOT survive the negatives control

Per advisor: before trusting the dramatic story above, checked whether layers 15-20 track real
visual evidence or are a content-independent default, using an UNBIASED sample (500 random
smallest-bin positives + 500 random negatives, `scripts/phase5_layer_auroc_validation.py`) --
deliberately not the two cherry-picked extreme cohorts used to find the original effect, to avoid
circularity.

**Result: layers 15-20 are a content-independent "yes" default, not evidence tracking.**
Negatives sit at mean P(yes)=0.982 (layer 15) and 1.000 (layer 20) -- statistically indistinguishable
from positives at the same layers (0.998 and 1.000 respectively). Thresholding at 0.5 on these
layers classifies EVERYTHING as "yes" (recall=1.0, neg_acc=0.0-0.002 at layers 15/18/20). **This
falsifies the "model correctly detects the object early, then a later computation suppresses it"
interpretation** -- the mid-network near-ceiling P(yes) is not a correct detection being erased,
it is a generic bias present regardless of ground truth, shared by positives and negatives alike.

**AUROC across layers (the number that actually tests whether any layer beats the final one):**
| layer | AUROC | @0.5 balanced acc | oracle balanced acc |
|---|---|---|---|
| 15 | 0.927 | 0.500 (says yes to everything) | -- |
| 18 | 0.892 | 0.501 | -- |
| 20 | **0.501 (chance)** | 0.500 | -- |
| 22 | 0.907 | 0.828 | 0.887 |
| 25 | 0.951 | 0.885 | 0.897 |
| 28 (final) | **0.956 (best)** | 0.840 | 0.895 |

**The final layer has the highest AUROC of every layer tested.** No intermediate layer carries
more separable information than the model's own output. Layer 25's oracle-ceiling (0.897) is
statistically tied with the final layer's (0.895, matching the earlier-established Qwen small-bin
oracle ceiling from `phase1_threshold_analysis.py` almost exactly) -- not higher. What looked like
an intermediate-layer "fix" (layer 25's raw 0.5-threshold balanced accuracy of 0.885, beating the
final layer's 0.840) is a **calibration coincidence, not new information**: the final layer's raw
output needs a threshold near ~0.01 (not 0.5) to reach its own ceiling -- consistent with this
project's much earlier finding of a strong global "no-bias" (blank-image P(yes)~0.03-0.04,
global tau*~0.033) -- while layer 25's raw output happens to be naturally better centered around
0.5. Reading layer 25 with a naive threshold reaches roughly the SAME operating point that
explicit threshold recalibration on the final layer already reaches (established weeks... several
sections ago in this log, oracle gap +0.0098) -- it is an alternate route to a known destination,
not a new source of accuracy beyond the already-proven ceiling.

**Honest, corrected conclusion**: the dramatic 80/80-vs-0/80 individual-trajectory collapse
pattern found earlier is a real, reproducible BEHAVIORAL observation (confirmed at the individual
item level, not a mean artifact) -- but its causal INTERPRETATION as "evidence detected then
actively suppressed" is not supported once negatives are brought into the comparison. The more
defensible, narrower reading: **Qwen's yes/no decision for this task crystallizes late in the
network (roughly layers 20-25), consistently across items** -- a genuine, well-localized
interpretability observation about where the decisive computation lives -- but it does not
constitute a hallucination-specific override mechanism, and it does not yield a fix that exceeds
the oracle-ceiling limit already established rigorously earlier in this project. **This is a
disciplined negative result on the strong "culprit" hypothesis, reported honestly rather than
allowed to stand on an incomplete control** -- consistent with this project's established practice
(the padding-bias-shift counterfactual, the LLaVA-NeXT multi-tile correction, and now this) of
checking a promising-looking result against the specific alternative explanation that could produce
it by artifact, before reporting it as a finding.

**Where this leaves "find the culprit"**: the logit-lens investigation, done properly, did not find
an exploitable culprit layer. The dissociation result that DID survive every control thrown at it
so far remains the Phase 4 localize-vs-answer result (grounding accuracy on confident denials:
95.1% box-emission rate and 43.2% IoU>0.5, vs. 45.5% box-emission on genuine negatives) -- that
result did not rely on a layer-wise mechanism claim at all, and is the strongest standing result of
this entire investigative thread.

### Where this leaves the paper's core claim (revised, 2026-09-03)

Per advisor, re-anchoring on what survives every objection raised so far, including the user's:
**the evidence-size effect is a genuine separability limit, not a decision-rule problem, and no
test-time correction recovers it -- proven independently on two architecturally distinct models**
(oracle-vs-global-threshold ceiling = +0.0162 for LLaVA, +0.0098 for Qwen; both under any
reasonable noise floor; both confirmed via a from-scratch zero-GPU scalar-sweep counterfactual that
reproduces the predicted ROC-tradeoff shape exactly). This is architectural (a property of what the
model's representation can support), threshold-free (AUROC-based, not accuracy-at-0.5), cross-model
(not a single-architecture quirk), and it is a pre-registered NEGATIVE result on the obvious fix --
which is a real, defensible contribution in its own right, not a placeholder for a positive result
that didn't pan out.

**The centrality/salience effect (this section) is the second, supporting pillar**: real,
bootstrap-confirmed, survives two independent confound checks, present even in a model that never
crops -- but reported at its true (modest) effect size, not inflated to co-headline status.

**The crop-truncation/pad-fix result (prior section) is now explicitly a THIRD, explanatory
footnote**: it explains why LLaVA's specific implementation suffers a much more severe interaction
than the architectural centrality effect alone would predict, and demonstrates one concrete,
fixable instance of the general problem -- but per the user's explicit judgment, it is not
promoted as evidence about vision-language architecture as a class, since it is properly
characterized as an implementation/pipeline detail of one specific codebase's defaults, not a
universal property of VLMs.

### Verdict: this is the paper

**A standard, widely-used VLM preprocessing default (resize-then-center-crop) silently deletes
peripheral visual evidence from the model's actual input, causing measurable, previously
unattributed hallucinations on presence questions -- and this is fixable with a one-line,
training-free preprocessing change that recovers the large majority of the lost accuracy, with a
clean negative control ruling out a placebo explanation.** This reframes and completes every prior
piece of this investigation: the original scaling curve (small objects are harder) is real but
partially an artifact of a fixable engineering choice, not solely an inherent representational
limit; the killed steering-vector experiment correctly diagnosed that *output-level* correction has
no headroom (+0.01-0.016 ceiling); the actual fix lives *upstream*, in how the image is prepared
for the model, not in how its answer is thresholded or steered. The cross-architecture
comparison (Qwen, which never crops) was not just a robustness check -- it was the control
condition that made the mechanism visible in the first place. This is a mechanistic, falsifiable,
actionable, and cross-architecture-grounded finding, at oral-ICLR caliber.

### Empirical phase: CLOSED (2026-09-03)

Per advisor: the marginal value of more GPU time has gone flat. What exists now: (1) a scaling
curve on a real public benchmark (POPE/COCO), (2) a pre-registered category-confound control that
passes in both models, (3) a rigorously quantified ceiling analysis that kills the obvious
"steering/threshold fix" and does so identically in two architectures, (4) a genuine
cross-architecture replication on an identical, apples-to-apples x-axis, and (5) an unplanned but
real secondary finding (confident-vs-hedging failure-mode divergence) that emerged from following
up an advisor-raised question with data already in hand rather than assumption. Not pursuing a
third model or the food-category negative outliers (sandwich/pizza/train/bowl) -- advisor's
judgment, and mine, is that neither would change the paper's claim, only pad it. Next step is
writing this up, not collecting more data.

## Phase 6: image/text token attention entanglement (2026-09-03)

Reopened after "empirical phase closed" above, per user's new hypothesis: "how about image and
text token entanglement." Operationalization -- does attention correctly route to the queried
object's image tokens during yes/no answering (where Phase 4 already showed 243 confident denials
still emit accurate bounding boxes 43.2% IoU>0.5 when asked to ground the same object)? If yes/no
attention is diffuse/misdirected while grounding attention concentrates correctly on the object,
that pinpoints WHERE the failure lives (task-dependent attention routing), a different and more
specific culprit than the falsified logit-lens "detected-then-suppressed" story from Phase 5.

**Method**: map each item's COCO bbox to Qwen3-VL's image-token grid (`image_grid_thw` from the
processor, merge_size=2, patch=14 -> token grid is (h/2, w/2), each token = 28x28px of the RESIZED
image, row-major, contiguous in the sequence). Index-math verified by hand: bbox [100,100,50,50] in
a 640x480->560x420 resize with a 15x20 grid -> indices {63,64,83,84}, exact match. Enrichment =
(attention mass on object tokens / n_obj_tokens) / (attention mass on all image tokens /
n_img_tokens), averaged over the last 4 layers and over heads; yes/no attention taken from the
last input-token position (forced-choice), grounding attention taken via teacher-forcing (feed
[grounding prompt + the model's own Phase-4-saved generated text up to the first bbox_2d array] as
one forward pass, extract attention from those generated-coordinate positions).

**Two design bugs caught in a 3-item smoke test before scaling up:**
1. *Instance-selection mismatch*: initially mapped only the single LARGEST COCO instance's bbox to
   tokens, but Phase 4's IoU check scored a prediction against ANY instance, and this project's
   established evidence measures (`pixel_area_frac`, `patch_token_frac`) are sums over ALL
   instances. For multi-instance categories (e.g. the `carrot` smoke item has 2 instances) this
   silently tested attention to a different region than what Phase 4 actually verified the model
   could localize. Fixed: `object_token_indices()` now takes the UNION of token sets over every
   instance of the category in the image.
2. *Small-n ratio noise*: objects in the smallest-area cohort routinely map to 1-4 tokens out of
   ~260-300 total. A per-item density RATIO is dominated by single-token attention noise for
   exactly the items this test cares about. Fixed: the attention-extraction functions now return
   raw components (`obj_attn_mass`, `n_obj_tokens`, `total_img_attn_mass`, `n_img_tokens`) instead
   of a single ratio, so analysis can use a POOLED ratio (sum of numerators / sum of denominators
   across many items) rather than averaging noisy per-item ratios -- caught before burning GPU time
   on a metric that would have been mostly noise at n=1-4 tokens/item.

**Smoke test (n=3, post-fix, for context only -- not a conclusion):**
| uid | category | n_instances | n_obj_tok | yes/no ratio | grounding ratio |
|---|---|---|---|---|---|
| pos_adversarial_527 | carrot | 2 | 33/300 | 0.621 | 0.194 |
| pos_adversarial_1543 | bird | 1 | 4/300 | 0.228 | 0.208 |
| pos_adversarial_2377 | traffic light | 1 | 1/260 | 2.334 | 1.284 |

All 3 items show yes/no-attention enrichment HIGHER than grounding-attention enrichment -- the
OPPOSITE direction of the entanglement hypothesis (which predicts diffuse/misdirected attention
during yes/no and concentrated attention during grounding, since grounding is the task that
succeeds). Also notable: raw total-image attention mass is lower during grounding than during
yes/no in all 3 items, consistent with grounding's generated coordinate tokens splitting attention
toward the text/numeric context rather than the image as a whole. n=3 is nowhere near sufficient to
conclude anything -- flagging the direction here only because it is unanimous, not because it is
decided -- so proceeding to the full run (N_PER_GROUP=150 each of confident_denial and
confident_correct_small, capped and resumable, launched via tmux) rather than either dismissing or
over-reading this smoke result. Full results and pooled-ratio analysis to follow.

**Full run (n=291: 141 confident_denial, 150 confident_correct_small; 123 of the denial group have
a saved Phase-4 grounding generation to reuse):**

1. *Original entanglement hypothesis, tested properly*: paired comparison (same 123 items, same
   object-token set) of pooled yes/no-attention enrichment (0.759) vs pooled grounding-attention
   enrichment (0.694). Win rate: grounding enrichment > yes/no enrichment in only 45/123 items
   (37%). Bootstrap 95% CI (2000 resamples) on (yesno_pooled - grounding_pooled): [-0.020, 0.176]
   -- crosses zero. **Not supported.** Grounding attention is not more focused on the object than
   yes/no attention is, for the same wrongly-denied items, at this sample size.

2. *Unplanned between-group signal*: pooled yes/no-attention enrichment is 0.848 for
   confident_denial vs 3.173 for confident_correct_small -- a 3.74x fold difference, bootstrap 95%
   CI (3000 resamples) [3.06x, 4.57x], never crosses 1. Confound checks before treating this as
   real: (a) confident_denial objects are smaller on average (median n_obj_tokens=4 vs 7,
   median pixel_area_frac 0.0018 vs 0.0049) -- but the gap survives stratifying by n_obj_tokens into
   4 bins ([1,3],[4,6],[7,10],[11+]), holding in every bin (2.4x-8.5x fold, denial always lower);
   (b) category composition -- 17 of 25 denial categories are shared with the 42 correct_small
   categories, including the most frequent ones on both sides (person, sports ball, truck), so this
   is not an obvious category artifact either, though a full category-fixed-effects regression
   (as used for the centrality and area findings) was not run here.

3. **Advisor review before headlining #2** raised two problems: (a) grounding's pooled enrichment
   of 0.694 (item 1 above) is itself BELOW the 1.0 no-preference baseline, yet those same items hit
   IoU>0.5 43.2% of the time in Phase 4 -- so on this project's own attention metric, sub-baseline
   enrichment is fully compatible with successful localization, which undercuts reading
   confident_denial's 0.848 as "attention failed to route to the object, therefore the object
   wasn't perceived." (b) the 3.74x gap may be tautological rather than diagnostic of hallucination:
   producing "yes" plausibly *requires* attending to the evidence; producing "no" does not -- so
   "says yes -> attends more" could hold for ANY object, present or genuinely absent, and would
   just restate the answer token in attention-space rather than reveal a hallucination-specific
   mechanism. The n_obj_tokens stratification rules out a size artifact; it does not rule out this.

**Phase 6b: query-swap control (launched to resolve the tautology question, 2026-09-03).** Same
image, same object-X token set (X = the item's actual queried category), but the QUESTION is
swapped to ask about a different category Y verified absent from that image via the FULL COCO
annotation set (not just the small POPE-joined subset) -- `scripts/phase6b_query_swap.py`. Logic:
  - If enrichment on X's tokens is unchanged whether the question asks about X or about absent Y,
    for a given group, attention on that region isn't query-conditioned at all for that group --
    it's fixed image-intrinsic saliency, and reduces to the existing area-scaling finding.
  - If confident_correct_small shows a big drop under the swapped query (attention was on X only
    because X was asked about) while confident_denial shows little/no drop (attention on X was
    already low and stays low regardless of the question) -- that pattern would be a genuine
    routing-failure signature, distinct from the tautology explanation, and would rehabilitate
    (in modified form) the original entanglement hypothesis.
  2-item smoke test (both confident_denial): both showed enrichment on X's tokens DROP under the
  swapped query (carrot item: 0.621 -> 0.097; traffic-light item: 2.334 -> 1.439) -- suggestive that
  attention is query-conditioned even in the denial group, opposite of the "flat/query-independent"
  pattern the routing-failure reading predicts, but n=2 is not evidence of anything. Full run
  (~290 items, matching the Phase 6 pool) launched via tmux; results pending.

**Phase 6b full run (n=291, joined 1:1 with the Phase 6 pool):** define the "modulation index" =
(pooled enrichment on X's tokens when asked about X) / (pooled enrichment on X's tokens when asked
about a genuinely absent Y) -- how much the model's attention on a fixed region moves in response
to changing what's asked.

| group | orig (query=X) | swap (query=absent Y) | modulation index | bootstrap 95% CI |
|---|---|---|---|---|
| confident_denial (n=141) | 0.848 | 0.699 | 1.21x | [1.09x, 1.48x] |
| confident_correct_small (n=150) | 3.173 | 0.576 | 5.51x | [4.55x, 6.71x] |

Bootstrap 95% CI on the DIFFERENCE in modulation index (correct_small - denial): [3.29x, 5.47x],
100% of 3000 resamples favor correct_small. This is the discriminating result the advisor's control
was designed to produce:
  - The tautology objection ("yes needs evidence, no doesn't, so the between-group gap is just
    answer-polarity") predicted confident_denial's two conditions (both answered "no" -- the
    original false-no to X, and the swap true-no to Y) should look similar to each other. They do
    (1.21x, only mildly above flat) -- but crucially the apparatus is proven sensitive to real
    query-conditioning by the SAME measurement on confident_correct_small, whose attention on X
    drops 5.5x once the question stops being about X. So the flatness in confident_denial is not a
    measurement-insensitivity artifact.
  - confident_denial's own-query enrichment (0.848, asked directly about the true, present object)
    is barely different from confident_correct_small's off-query enrichment on a KNOWN-ABSENT
    object (0.576) -- i.e., even when the model is asked point-blank about the object it is about
    to confidently (and wrongly) deny, its attention on that object's own image tokens looks
    similar to how a well-behaved item's attention looks toward an object that plainly isn't there.
  - Net reading: attention IS query-conditioned in this model in general (proven by the
    correct_small modulation), but for confident-denial items specifically, attention to the true
    object's location fails to respond to the query -- a genuine, controlled routing-failure
    signature, not a restatement of the answer token. This is a modified, better-evidenced version
    of the original "entanglement" hypothesis: not "grounding succeeds where yes/no attention
    fails" (that specific comparison, item 1 above, was null), but "when yes/no fails, attention
    was never successfully directed at the object by the query in the first place."
  - Caveat carried forward honestly: this is still observational/correlational (same epistemic
    status as the Phase 5 logit-lens work) -- it shows an attention signature that discriminates the
    two outcomes, not a proven causal mechanism, and no intervention (e.g. attention patching) has
    been run to test whether forcing attention onto the object's tokens would fix the answer.

**Advisor caught a real design flaw in the 6b modulation-index comparison before it could be
headlined.** confident_denial's two conditions (query=X, query=absent-Y) are BOTH answered "no" --
polarity held fixed, only query-target varies. confident_correct_small's two conditions
(query=X->yes, query=absent-Y->no) vary query-target AND polarity together. So the 5.51x vs 1.21x
gap is not clean evidence of query-target-conditioning for correct_small -- it is equally
consistent with pure polarity-conditioning (attend when about to say yes, don't when about to say
no), which is exactly the tautology this whole line is trying to rule out. The "apparatus is
proven sensitive to query-conditioning" claim in the 6b writeup above does not hold and should be
read as struck; instead read the 6b table only as an entanglement-null-test extension.

Fix (already built, not yet run): `scripts/phase6c_present_other.py` adds a THIRD condition --
query = Z, a different object verified PRESENT in the same image (so the expected answer is YES,
like query=X for confident_correct_small, but the target is NOT X) -- decoupling target from
polarity. Compiles clean, smoke-tested logic (category selection) without GPU. Two independent
readings this resolves:
  - confident_correct_small: enrichment on X's tokens under query=Z near ~3 (matching query=X) =>
    the 3.17 was polarity all along, collapsing the between-group finding into the tautology.
    Near ~0.6 (matching query=absent-Y) => target-conditioning is real, not polarity, and the
    denial group's flatness across ALL prior conditions becomes meaningful.
  - confident_denial: if X's-token enrichment stays in the same ~0.7-0.85 band across query=X
    (no), query=absent-Y (no), AND query=Z (yes-about-something-else), that is the strong version
    of the routing-failure claim -- attention on X is invariant to both target and polarity of the
    question, not just to "another no."

**Blocked, 2026-09-03**: GPU0 was claimed by an unrelated job (`geo_train`, stage1_geometric_vla
training, ~46/49GB used) immediately after Phase 6b finished; GPU1 (4GB total) can't fit
Qwen3-VL-2B in fp16. Per user instruction, holding this thread without polling -- phase6c is
built and ready to launch (smoke-test logic verified, GPU forward pass not yet exercised) as soon
as GPU0 has headroom or the user says to proceed on GPU1/elsewhere. **Do not headline the Phase 6
between-group or 6b modulation-index results until phase6c returns.**

**Phase 6c ran 2026-09-06 (GPU0 freed up once `geo_train` finished), n=271, 3-way joined with
Phase 6 and 6b (n=123 confident_denial, n=148 confident_correct_small).** Pooled enrichment on
X's own tokens under three queries -- X (own category), absent-Y (Phase 6b), present-other-Z
(Phase 6c, a different object verified present in the same image):

| group | query=X | query=absent-Y | query=present-Z |
|---|---|---|---|
| confident_denial (n=123) | 0.730 (answer: NO) | 0.336 (answer: NO) | 0.592 (answer: YES, filtered to Z-answered-yes n=109) |
| confident_correct_small (n=148) | 3.145 (answer: YES) | 0.576 (answer: NO) | 0.683 (answer: YES, filtered to Z-answered-yes n=139) |

**This resolves the tautology question the advisor raised against Phase 6b.** The clean,
polarity-matched contrast is WITHIN confident_correct_small alone: query=X (yes, target=X) gives
3.145, query=Z (yes, target=Z, same polarity) gives 0.713 (unfiltered) / 0.683 (filtered) -- a
~4.4x drop despite BOTH being answered "yes." Since polarity is held fixed and only the query
target changes, this alone rules out "attention just tracks yes/no polarity" -- attention in this
model genuinely tracks WHAT IS ASKED, not just what the answer will be.

**Given that, the finding is:** confident_denial's attention on X under its OWN query (0.730) is
statistically indistinguishable from confident_correct_small's attention on X when X isn't even
being asked about (0.713/0.683, the present-Z condition) -- bootstrap 95% CI on the X/Z
target-modulation index: confident_denial [0.93, 1.28] (includes 1.0 -- no detectable
target-conditioning), confident_correct_small [3.65, 5.39] (clearly >1, strong target-conditioning);
CI on the group difference [2.52, 4.30], 100% of 3000 resamples favor correct_small. Phrased per
advisor's correction (denial-group attention is NOT literally invariant -- absent-Y's 0.336 is
below both X and Z, so state "sharply reduced target-conditioning," not "flat"): **when Qwen3-VL is
about to confidently deny an object that is actually present, its attention to that object's own
location looks like the attention a correctly-answered item gives to an object nobody asked
about.** The query fails to pull attention to the right region in the first place, for confident
denials specifically -- a real, controlled routing-failure signature, not a restatement of answer
polarity (ruled out by the within-group Z-vs-X contrast above).

**Caveats carried forward**: (a) still observational, no attention-patching intervention run to
show forcing attention onto X's tokens would fix P(yes) -- same epistemic status as Phase 5;
(b) Z categories are drawn from a fixed common-category list and skew toward large/frequent objects
(person dominates, 101/271) -- doesn't threaten the within-group X-vs-Z contrast (measuring
attention ON X's tokens in both conditions) but is a legitimate reviewer question if this becomes a
headline result; (c) the earlier 6b claim that "the apparatus is proven sensitive to
query-conditioning by the same measurement on confident_correct_small" is STRUCK as stated (that
5.51x comparison mixed target and polarity) -- superseded by the clean within-group X-vs-Z contrast
above, which makes the same point without the confound.

**Net for the paper**: Phase 4 shows localization succeeds where forced yes/no judgment
confidently fails on the same items. Phase 6c shows a candidate reason WHY the yes/no route fails:
the query never successfully pulls attention to the object's location for confident-denial items,
while the same attention mechanism demonstrably CAN be target-conditioned (proven on
correctly-answered items). This is the strongest mechanistic candidate produced by Phase 6, and
survived two rounds of advisor-caught design flaws (instance-selection/small-n noise in the
original 6a design, and the polarity confound in 6b) before landing here.

## Phase 6e: attention-patching intervention (2026-09-07)

Phase 6c is correlational. This asks the causal question directly: mechanically FORCE attention
onto the object's own tokens at the answer position and see whether P(yes) recovers. If it does,
that is an actual fix (identified failure mode + identified remedy, no fine-tuning). If it doesn't,
attention allocation isn't the causal bottleneck and Phase 6c's signature, while real, isn't the
culprit -- still informative, but different from what it looks like.

Method: monkeypatched `Qwen3VLTextAttention.forward` (transformers' Qwen3-VL text-decoder
attention) to reproduce the library's exact eager-attention computation, but redistribute the
post-softmax attention distribution at the LAST query position only (causally sufficient and
self-contained under causal masking -- no other position can be affected) so the queried object's
token set receives a target total share F of attention mass, rescaling all other keys
proportionally to preserve their relative pattern. Two targets per item: the object's own tokens,
and a random same-size set of other image tokens (controls for "any redistribution helps somehow",
the same confound that sank Phase 6d).

**First version (4 layers patched, F=0.30/0.60) -- caught by advisor before being written up**: the
patch mechanically worked (realized share matched target exactly, verified via output_attentions),
and P(yes) did NOT recover at either F (0/141 items improved over baseline) -- but object and
random targets were statistically indistinguishable (59/141 wins, ~chance). Advisor identified two
real problems before this could be called a refutation: (1) patching layers 24-27 means layer 27's
observed "realized share" reflects FOUR stacked renormalizations (layer 27 reads keys built from
layer 26's already-patched output), so the measurement doesn't isolate one clean intervention;
(2) F=0.30/0.60 are 3-6x higher than what confident_correct_small items NATURALLY show (checked
against phase6_attention_results.jsonl: median natural object share is 0.094 for
confident_correct_small vs 0.012 for confident_denial) -- an intervention that far outside any real
item's range is closer to generically shoving the model off-manifold than to a considered dose.

**Corrected version (single layer patched -- layer 27 only -- F=0.094 matching correct_small's
actual median share, plus F=0.20 as a stronger dose), full run n=141:**

| F | median obj-patch P(yes) | median random-patch P(yes) | obj>random | obj_patch>baseline | bootstrap 95% CI (obj-random) |
|---|---|---|---|---|---|
| 0.094 | 0.000127 | 0.000156 | 58/141 | 6/141 | [-0.00004, 0.00003] |
| 0.20  | 0.000123 | 0.000131 | 67/141 | 6/141 | [-0.00009, 0.00004] |

(baseline median P(yes) = 0.000185, for reference -- realized share matched target exactly at both
F, confirming the single-layer patch is clean.) The null holds after the methodological fix:
forcing attention onto the object's own tokens, even at a share matching what correctly-answered
items naturally show, does not recover P(yes) (only 6/141 = 4.3% of items even nominally improved,
consistent with noise), and is statistically indistinguishable from forcing the same amount of
attention onto random other image tokens (CI on the difference clearly crosses zero at both F).

**Conclusion (per advisor's explicit guidance on how to write this up):** do NOT read this as
"attention causality refuted" in a strong sense, but as: **we could not construct an intervention
that shows attention reallocation causally fixes the hallucination** -- the object-vs-random
symmetry means late-layer attention patching, as implemented here, isn't discriminating between
targets at all. This does not undo Phase 6c (which compared NATURALLY occurring attention across
valid query conditions, a different and still-valid comparison) -- it means the natural correlation
Phase 6c found does not translate into an artificial single-layer intervention that fixes the
answer. Combined with Phase 5 (logit-lens layer intervention: falsified) and Phase 6d
(autoregressive-decoding-depth intervention: falsified), this is the THIRD distinct causal-fix
hypothesis tested and found wanting on the decoder side. Taken together, these three ruled-out
routes point the likely locus of the failure UPSTREAM of the decoder -- most plausibly in the
vision encoder's actual representation of small/peripheral objects (consistent with the original
scaling-law finding) rather than in how the language-model decoder allocates attention, chooses a
layer, or paces its decoding once that representation is already formed. This was not directly
tested (no vision-encoder-level intervention was run) and should be stated as the natural next
question, not a demonstrated result.

## Phase 6d: autoregressive-decoding-depth intervention (2026-09-06)

New hypothesis from the user: what if the culprit is that the forced yes/no answer is read off a
SINGLE forward pass (no autoregressive steps), while grounding (Phase 4, which succeeds on the same
confident-denial items) requires ~15-20 generated coordinate tokens, each a fresh forward pass that
gets to re-attend to the image with more context? This would unify Phase 4 (dissociation) and
Phase 6 (attention doesn't concentrate on the object in one shot) under one mechanism, AND -- unlike
Phases 5/6, which are correlational -- suggests an actual testable INTERVENTION: force a few extra
autoregressive steps before the yes/no token and see if P(yes) recovers on the confident_denial
cohort.

`scripts/phase6d_decode_steps.py`, three conditions per item (same 141-item confident_denial
cohort from Phase 6, GPU0 freed up by 2026-09-06 once `geo_train` finished):
  1. baseline -- original single-shot forced-choice pass.
  2. filler -- same token COUNT of extra content before the answer, but content-free ("Hmm, let me
     think about this.", repeated to token-match the cot condition per item) -- controls for
     "any extra tokens help for some generic reason" vs. real image re-attention.
  3. cot -- model freely generates (~32 tokens, greedy) a one-sentence scene description, THEN the
     yes/no question is re-asked with that self-generated text as context.

**Design bug caught in the first 2-item smoke test, before any at-scale run**: the initial cot
prompt ("describe the area where a {category} would be if present") let the model just answer
"No, there is no {category} in the image" during the "description" step itself -- which then
anchors the final answer via self-consistency, rather than testing whether extra steps help
re-attention. Fixed: the cot prompt no longer names the queried category or invites a
presence/absence verdict at all -- just "describe in one sentence what objects and scene elements
you notice in this image."

**Re-smoke-tested (n=3) after the fix**: no more leakage (descriptions are neutral scene
summaries, don't mention the queried object). Reproducible pattern across all 3: BOTH cot and
filler substantially raise P(yes) relative to baseline (e.g. bird: 5.96e-8 -> cot 0.026 / filler
0.057; traffic light: 0.0012 -> cot 0.281 / filler 0.220; bicycle: 0.0026 -> cot 0.131 / filler
0.101) -- cot vs filler are close to each other, no consistent large gap either direction across
these 3. Meanwhile attention enrichment on the object's own tokens DECREASES from baseline in
BOTH conditions (traffic light: 2.33 -> cot 0.37 / filler 0.37; bicycle: 4.62 -> cot 1.89 / filler
2.05) -- the opposite of what "extra autoregressive steps let attention find the evidence" would
predict. n=3, not a conclusion, but the direction is reproducible and worth full-scale
verification: it currently looks more like "moving away from the terse single-shot prompt
template destabilizes the model's confident-no calibration generically" than "more decoding
steps -> better attention routing -> recovered accuracy" -- i.e. a possible finding about
confident denials being a fragile, template-specific calibration artifact rather than a stable
readout, but NOT yet the causal "autoregressive decoding is the fix" story hoped for. Full run
launched via tmux (`phase6d`, resumable, target ~141 rows); results pending.

**Full run (n=141, complete):**

| condition | median P(yes) | mean P(yes) | frac > 0.5 | frac > 0.1 | pooled attn enrichment |
|---|---|---|---|---|---|
| baseline | 0.00019 | 0.0014 | 0.000 | 0.000 | 0.848 |
| cot (image-grounded) | 0.056 | 0.133 | 0.043 | 0.362 | 0.604 |
| filler (content-free) | 0.119 | 0.153 | 0.043 | 0.617 | 0.643 |

Both interventions produce a large, real recovery in P(yes) from baseline -- moving away from the
terse single-shot "answer yes or no" template substantially raises P(yes) on items the model
otherwise confidently (and wrongly) denies. But the decisive comparison is cot vs. filler, and it
goes the WRONG way for the autoregressive-decoding-depth hypothesis: filler beats cot on 108/141
items (77%), cot beats filler on only 33/141 -- binomial test against p=0.5, p=8.4e-11. Bootstrap CI
on the mean difference is wide and crosses zero ([-0.043, 0.003], skewed distribution), but the
sign/win-rate result is the robust signal here given how skewed P(yes) is. Pooled attention
enrichment on the object's own tokens DROPS under both interventions relative to baseline (0.848 ->
0.604 cot / 0.643 filler; bootstrap 95% CI on the baseline-vs-cot drop: [0.154, 0.372], clearly
nonzero) -- the opposite direction "more autoregressive steps -> better attention routing -> fixed
answer" would predict.

**Conclusion: the autoregressive-decoding-depth hypothesis is REFUTED in its strong form.** Content-
free filler recovers P(yes) at least as well as an actual image-grounded description, and attention
on the true object's location gets WORSE, not better, as P(yes) recovers. The recovery is not
"extra decoding steps let the model look again and find the evidence" -- it looks like a generic
template/distributional-shift effect: the exact terse "Please answer this question with yes or no"
single-token format induces an unusually strong, shallow "confident no" bias that is disrupted by
ANY deviation from that template, relevant or not, and the disruption does not route through
increased visual evidence use. This is still a useful, novel finding in its own right -- it shows
confident hallucinated denials are fragile artifacts of the specific answer-elicitation template,
not stable reflections of the model's visual belief -- and it strengthens (rather than contradicts)
Phase 5's "decision crystallizes late / is shallow" result: not only does the forced single-token
decision fail to reflect deep evidence integration, it is demonstrably unstable to task-irrelevant
prompt perturbations. Does NOT support framing the localize-vs-answer dissociation (Phase 4) as
caused by "too few decoding steps" -- that remains an open, unresolved mechanism (Phase 6/6b/6c
still pending the query-target-vs-polarity control).

[Note: the query-target-vs-polarity control referenced above was run and resolved on 2026-09-07 --
see the "Phase 6c ran 2026-09-06" entry earlier in this log (inserted at its topical location,
chronologically before this Phase 6d section in the file even though it was logged after). Phase
6c found a real, controlled attention-routing signature. Phase 6e (below) then tested it causally.]

## Phase 7: vision-representation zoom intervention (2026-09-07) -- POSITIVE result

Direct test of the upstream hypothesis Phase 6e's elimination pointed to: if a small/peripheral
object only occupies a handful of low-effective-resolution patches once the full scene is
downsampled to the model's input grid, the vision encoder may never build a clear representation
of it -- no decoder-side intervention (Phase 5/6d/6e all targeted "which layer", "how many decode
steps", "which tokens get attention") can fix a representation that was never formed well upstream.

`scripts/phase7_vision_zoom.py`: give the model a SECOND image alongside the original -- a crop of
the SAME photograph, tightly around the object's own ground-truth region (COCO bbox, 25% padding),
which Qwen's own image processor then upscales to occupy far more patches than the region got
embedded in the full scene. This adds ZERO new information (same pixels, no external data, no
dataset construction) -- only more patches / effective resolution spent on that region. Three
conditions, same confident_denial cohort (n=141) as Phase 6/6d/6e:
  1. baseline -- single full image, standard prompt (no connector sentence).
  2. oracle_zoom -- full image + connector sentence ("Here is a zoomed-in crop of a region from
     the same image" -- deliberately generic, does not name the category or location, avoiding the
     leakage bug caught in Phase 6d) + crop of the TRUE object's own region.
  3. random_zoom -- identical connector + a SAME-SIZE crop from a random OTHER region of the image,
     verified non-overlapping with the object -- controls for "any second zoomed-in image helps
     regardless of content" (same confound structure that sank Phase 6d's filler-vs-cot test).

**Smoke test (n=3) was dramatic**: bird item flipped from baseline 5.96e-8 to oracle_zoom 0.520
(crossing the yes/no threshold) vs random_zoom 2.1e-6; traffic-light item went from 0.0012 to
0.989 vs 0.019. Full run (n=141) confirms this holds at scale:

| condition | median P(yes) | mean P(yes) | frac > 0.5 |
|---|---|---|---|
| baseline | 0.000185 | 0.00143 | 0.000 |
| oracle_zoom | 0.0078 | 0.162 | 0.128 |
| random_zoom | 0.00047 | 0.0103 | 0.000 |

oracle_zoom > random_zoom on 105/141 items (one-sided binomial p=2.5e-9). Bootstrap 95% CI on
mean(oracle-random): [0.104, 0.202] (excludes 0). Bootstrap 95% CI on the flip-to-"yes" rate
difference (oracle - random): [0.078, 0.184] (excludes 0). **18/141 (12.8%) of confident
hallucinated denials flip to a correct "yes" when given a zoomed view of the true object's own
region; 0/141 flip under a size-matched random-region zoom.** Dose-response check: items that
flipped skew SMALLER (median pixel_area_frac 0.00145) than items that didn't (0.00401) -- exactly
the pattern the resolution-starvation account predicts (the tinier the object, the more a fixed
zoom multiplies its effective resolution).

**Two framing corrections applied per advisor review before logging as headline** (result itself
confirmed sound -- coherence of the dose-response detail alone was persuasive):
1. `baseline` uses no connector sentence while both zoom conditions do, and Phase 6d already
   showed this model's confident denials are sensitive to content-free prompt/template
   perturbation alone -- so `oracle_zoom > baseline` (111/141) is contaminated by a template-shift
   effect independently measured to be large. The clean, connector-matched comparison is
   oracle-vs-random (both carry the identical connector sentence and second-image structure); that
   is what the headline numbers above rely on, not the raw oracle-vs-baseline gap.
2. **This is an ORACLE demonstration, not a deployable fix**: the crop location comes from COCO
   ground truth, which requires already knowing where the object is. The honest claim is: *the
   evidence is present in the original image but under-resolved; re-rendering the same region at
   higher effective resolution recovers 12.8% of confident denials to a correct "yes," vs. 0% for a
   size-matched control region.* This demonstrates the information is recoverable from the same
   pixels (locating the failure upstream, in representation/resolution, not decoder attention or
   decoding) -- it does not by itself hand over a deployable pipeline.

**Natural deployable follow-up (not yet run, noted as future work)**: Phase 4 already showed these
same confident-denial items get grounded with IoU>0.5 43.2% of the time. A self-contained pipeline
-- ground the object with the model's own bounding-box output, crop+zoom to that box, re-ask -- would
need no oracle/ground-truth location at all. Worth running before any paper claim of a practical
fix, since it substitutes the model's own (imperfect) localization for the COCO oracle used here.

**Net for the paper**: this is now the strongest result in the project. The arc is: Phase 4
(dissociation exists) -> Phase 6c (a correlational attention signature for why yes/no fails) ->
Phase 5 / 6d / 6e (three decoder-side causal-fix hypotheses eliminated: not layer-suppression, not
decoding depth, not attention allocation) -> Phase 7 (the locus is upstream -- the evidence is
present in the pixels but under-resolved at the patch grid the model actually processes, and
re-resolving the SAME region substantially recovers accuracy, in a well-controlled, dose-responsive
way). This is the first result in the whole project that is a genuine intervention with a positive
effect, not a ruled-out hypothesis or a correlational signature.
