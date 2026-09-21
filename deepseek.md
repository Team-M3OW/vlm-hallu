# deepseek.md — handoff from the DeepSeek session to Claude

**Author of this file:** the `deepseek-v4.1-flash` session (opencode), 2026-09-21.
**Recipient:** the Claude session that resumed after the weekly rate limit and is editing `paper/main.tex`.
**Purpose:** everything I ran, everything I changed, what is uncommitted, what is still open, and instructions.

> Read this before touching `paper/main.tex` or `FINDINGS.md` again: we were editing the same files
> concurrently between ~20:00 and ~23:15 IST on 2026-09-21. I have stopped paper edits after the appendix
> consolidation described in §6. Nothing I did has been committed by me.

---

## 1. The brief

The user asked (2026-09-20, 10:07 UTC) for the mechanistic-interpretability half of the paper:

1. interpretability results and their causality for most of our claims;
2. establish why **both** methods work, using logit lens / attention readouts / linear probes;
3. theoretical grounding of both methods by theorems and proofs, validated on actual data — "extensive
   experiments, not the bare minimum".

Phases 199–203 had already been run (read-out depth curve, depth weights, nuisance ablation, feature
ablation, patching + logit lens) before the rate limit. I continued from there.

## 2. Pre-registration

`PREREG_MECH_THEORY.md` (repo root) — written before any phase-204+ run. It fixes:
- the propositions with proofs (§3 there),
- the predictions and pass bars for phases 204–207,
- the decision rules (two-model rule, 8000-resample bootstrap, OOF GroupKFold(5) × seeds 700/701/702,
  end-task gating, exactly-zero contrasts are pipeline faults).

**Read it before changing any claim in this file's scope.** The predictions that failed are listed in §7.

## 3. What I ran, and what it showed

### Phase 204 — TWR/DWA formal grounding (CPU + GPU end-task), both models

Scripts: `scripts/phase204_theory.py`, `scripts/phase204_endtask.py`, `scripts/phase204_endtask_analyze.py`.
Data: `data/phase204_theory_{qwen3,qwen2}.json`, `data/phase204_endtask_{qwen3,qwen2}.jsonl`.
Logs: `logs/q_204_theory.log`, `logs/q_204e_chain.log`.

**204a — the linear model's assumptions, measured:**
- rank-one share of the item-mean maps: **0.855** (Qwen3) / **0.769** (Qwen2);
- cos(α, g), the orthogonality the old signedness argument needed: **+0.431 / +0.388** → premise **fails**;
- mean corr(m_l, t): −0.020 / −0.017.

**204c — the decisive 2×2 (free-sign vs non-negative × raw vs nuisance-ablated maps).**
Mean W=0.25 coverage, modal grid n=126, UNMASKED maps:

| arm | Qwen3 raw | Qwen2 raw | Qwen3 abl | Qwen2 abl |
|---|---|---|---|---|
| block mean | 0.083 | 0.292 | **0.438** | **0.490** |
| ridge (deployed DWA) | 0.578 | 0.588 | 0.539 | 0.518 |
| nnls_A (log-A ≥ 0) | 0.609 | 0.587 | 0.529 | 0.519 |
| nnls_R (rank ≥ 0) | 0.626 | 0.603 | 0.537 | 0.523 |
| **nnls_AR (both ≥ 0)** | **0.594** | **0.584** | 0.515 | 0.517 |
| constrained Σ w_l α_l = 0 | 0.619 | 0.589 | 0.533 | 0.516 |
| tied (one weight all layers) | 0.516 | 0.514 | 0.516 | 0.514 |
| shuffle (depth profile permuted) | 0.317 | 0.507 | 0.508 | 0.525 |
| map-space free-sign (28 wts) | 0.515 | 0.530 | 0.515 | 0.530 |
| map-space non-negative | 0.471 | 0.533 | 0.471 | 0.533 |

- **The pre-registered signedness prediction is REFUTED 0 of 2.** The non-negative fit keeps or beats
  DWA, zeroing **44/56 (Qwen3)** and **43/56 (Qwen2)** per-layer coefficients. Selected log-attention
  layers: **[4, 5, 17, 19]** (Qwen3) and **[19, 21]** (Qwen2).
- Map-space sign test: free-sign loses only **+0.044 / −0.002** to non-negative → "a mean can only add"
  is at most a small part of the correction.
- What *is* load-bearing: the depth **ordering** (shuffle costs 0.26/0.08 of the advantage; tied costs
  0.06/0.07) and the supervised fit. On ablated maps every fitted arm collapses to within 0.10/0.03 of
  the block mean.
- 204b prior-share diagnostic **failed to track coverage across arms (r=+0.11/−0.01)** — do not use it.

**204e — end task, deployed pipeline, n=191, bar uniform@600:**

| arm | Qwen3 | Qwen2 |
|---|---|---|
| uniform@600 | 63.9 | 58.1 |
| ridge (DWA) | 72.8 | 70.2 |
| nnls_AR | **71.7** | **68.1** |
| map-space free-sign | 70.2 | 63.9 |
| nnls − ridge | **−1.0 [−6.3,+4.2] n.s.** | **−2.1 [−5.8,+1.6] n.s.** |
| ridge − bar | +8.9 [+0.5,+17.3] | +12.0 [+4.7,+19.9] |
| nnls − bar | +7.9 [+0.0,+15.7] | +9.9 [+2.6,+17.8] |

**Consequence (already in the paper):** the correction is a *supervised depth weighting*, not signed
subtraction; the fitted signs are a ridge-collinearity artefact.

### Phase 205 — AVR/TSR's premise at its own operating point (GPU), both models

Script: `scripts/phase205_tsr_probe.py`; analysis `scripts/phase205_analyze.py`.
Data: `data/phase205_tsr_probe_{qwen3,qwen2}.jsonl`. Logs under `logs/q_205_*`.

Unpruned 900-token pass; donor = a different image, same question; patches at the layer output.

| patch, 900 tokens | Qwen3 KL (flips) | Qwen2 KL (flips) |
|---|---|---|
| dropped positions @L16 (810) | 0.0022 (1.0%) | 0.0038 (3.7%) |
| **kept** positions @L16 (90) | **0.0065 (3.1%)** | **0.0027 (0.5%)** |
| dropped @L8 | 0.3407 (20.9%) | 0.2622 (29.8%) |
| dropped @L24 | 0.0003 (0.5%) | 0.0006 (0.5%) |
| AVR (attention keep) vs base | KL 0.0025, agree 97.9% | KL 0.0037, agree 96.3% |
| AVR (random keep) vs base | KL 0.0075, agree 97.4% | KL 0.0047, agree 96.9% |

- The pre-registered control "kept values must be large" **failed** — because at L16 **all** visual
  values are inert, kept and pruned alike. That is a stronger premise, and it is why random keep works.
- Qwen2's arg-max agreement (96.3%) is a marginal miss of the pre-registered 97% bar — reported as such
  in `FINDINGS §54B`, do not round it up.

### Phase 206 — location linear probe (CPU), both models

Script: `scripts/phase206_probe_loc.py`; data `data/phase206_probe_loc_{qwen3,qwen2}.json`.

- Metric caveat: V*Bench targets are centred, so the constant mean-centre predictor covers **0.956**;
  probe placement lands below it (0.17–0.30). **Score the probe by correlation, not coverage.**
- Correlation rises through transport and peaks in the read-out band: **+0.635 at L18–19 (Qwen3)**,
  **+0.603 at L18 (Qwen2)**; shuffle control max |r| = 0.26 / 0.21, so do not read adjacent-layer
  differences below ~0.1.
- Interpretation: location is *accessible* at the answer position but not causally used — the third
  independent technique placing the band.

### Extensions of existing phases (all queued and completed by me)

- **Phase 203 → n=191 both strata** (`data/phase203_patchlens_{qwen3,qwen2}.jsonl` now 191 rows):
  Qwen3 C1 1.095/51.8%, C2 1.149/50.8%, img L≥16 **0.0022 / 0.9%**, paired KL(L16−L4) −1.090
  [−1.362,−0.853], lens onset L22, lens acc = base acc 57%. Qwen2: 0.611/49.7%, 0.601/48.2%,
  **0.0018 / 3.1%**, −0.607, onset L24, acc 51%. Both strata at the floor; img L8 large in all four
  cells. One anomaly under the analysis-time onset rule (Qwen2 relative L9) — a criterion fluctuation,
  not a curve shape.
- **Phase 176 Qwen2 answer-row mask** (`data/phase176_sens_qwen2.json`, n=40): KL **0.0343**,
  **0.0% flips** (Qwen3 0.0250 / 2.5%). All-rows mask remains 1.266/0.532 with 42%/45% flips. This
  closes the audit's biggest single-model gap.
- **Phase 199 Qwen2 layer sweep** (`data/phase199_layersweep_qwen2.jsonl`, n=191): spread
  **36.1–62.8**; L0–15 max 39.8 vs bar 58.1; L16–21 mean 55.4, best L21 62.8; L22–27 max 58.6.
  **DWA − best single layer = +7.3 [+1.0,+13.6] ✔** (Qwen3 +2.6 n.s.); block − best −3.7 n.s.;
  L14 − bar −19.9 ✔.
- **Phase 176c Qwen3-VL-8B masking** (`data/phase176c_transport_qwen3_8b.json`, 36 layers, n=40):
  suffix KL L16 0.413 (15% flips), **L20 0.052 (5%)**, **L24 0.009 (0%)**; prefix saturates at L16
  (1.740); late L24–35 mean 0.0025. The 0.57 law is now a masking result on the 36-layer model too
  (L20 = 0.556 × depth), matching the pruning-derived L21.

## 4. FINDINGS sections I added

`FINDINGS.md`: **§51B** (Qwen2 sweep), **§52B** (phase 203 at n=191), **§53** (phase 204 + pre-registered
verdicts table), **§54 / §54B** (phase 205 both models), **§55** (phase 206), **§56** (claim-to-evidence
map), **§57** (Qwen2 answer-row replication), **§58** (36-layer masking). Earlier in the session I also
wrote **§52** (phase 203 original, n=60).

## 5. Paper edits I made (`paper/main.tex`)

All of these were made with the audit of 2026-09-21 as the driver; each is a correction or a new
evidence-backed statement, not a style change:

1. **Table 1 (`tab:transport`)** — now two-model: answer-row `0.025 / 0.034`, flips `2.5% / 0%`;
   all-rows `1.266 / 0.532`, `42% / 45%`. Caption no longer needs a Qwen3 qualifier.
2. **Abstract + intro (ii)** — "flips no answers from L16" was false (12%/5% at L16, ~0 from L20);
   corrected. "0.57 on every architecture" now backed by masking on all three checkpoints.
3. **Abstract + intro (iii)** — native-resolution claim now says DWA is indistinguishable on both
   checkpoints, AVR on one; ratios 5.5×/7.2×.
4. **Figure 1 caption** — dashed line now described as the transport boundary with the measured flips.
5. **§transport** — 36-layer sentence now reports the masking result, not pruning only.
6. **§Policy I "Why re-weighting and not selection"** — the old sentence "the correction requires
   subtraction, and a mean can only add" is **gone**; the learned-head attribution is fixed (57%/71% are
   the GBT's, the deployed ridge ties it), and the corrected mechanism is stated with a reference to
   `app:theory`.
7. **Figure `fig_infer_ridge` caption** — the §49-forbidden "negative weight rather than patching the
   geometry" sentence replaced by the supervised-depth-weighting account.
8. **§Policy II** — "pruning free in 9 of 10" corrected to §38's wording (±0.5 in 7/10, two ahead, one
   real cost); the boundary-depth paragraph corrected (FastV is the better schedule on Qwen2; taper
   losses per model; the 0.14-depth run is voided and labelled as accidental evidence).
9. **§Scope law** — mechanism clause corrected: a W=0.25 window contains both boxes on 93.4% of
   relational items; the failure is integration, not coverage.
10. **Gate-max clause** — "above every published rule" is false on Qwen2 (LASER 61.3); fixed.
11. **Intro (i) + figure** — now two-model; new figure `paper/figs/fig_layer_accuracy2.pdf`
    (script `scripts/fig_layer_accuracy2.py`), replacing the single-panel `fig_layer_accuracy.pdf`
    reference in the paper.
12. **`app:depth` table caption + `tab:prefix`** — 36-layer masking rows added; caption updated.
13. **`app:claims`** — new claim-to-evidence map (14 rows), matching `FINDINGS §56`.
14. **`app:mech`** — added the location-probe sentence as the fourth independent technique.

## 6. The appendix consolidation (what I did last, at the user's request)

There were two overlapping appendices: your `app:theory` ("Propositions, and what validates them") and my
`app:formal` ("Formal grounding"). The user asked me to consolidate. I did:

- **Removed `app:formal` entirely.**
- **Kept your `app:theory` as canonical** and folded my formal content in:
  - your Proposition 3 (tail dominance) → **Theorem 1** (it has the only real proof);
  - my leakage decomposition → **Lemma 1**;
  - my non-negative-impossibility result → **Lemma 2**;
  - your Propositions 1, 2, 4 unchanged; my value-inertness statement → **Proposition 3**, with both
    models' 900-token patching numbers;
  - your measured paragraphs (100th percentile, Δ_B/σ_B, unsupervised correction, "what the fit adds",
    failed refinement, rejected accounts) are preserved verbatim, with two additions: the rank-one
    shares 0.86/0.77 and the map-space sign costs 0.044/0.002.
- Intro now correctly says "one theorem, four propositions and two lemmas".
- Added `\newtheorem{theorem}` and `\newtheorem{lemma}`; `\newtheorem{proposition}` already existed.
- All `\ref{app:formal}` → `\ref{app:theory}`.
- Added the probe sentence to `app:mech`.
- **Builds clean at 23 pages, no undefined references** (built with `tectonic`, see §8).

## 7. Pre-registered predictions that FAILED (keep them honest)

| prediction | outcome |
|---|---|
| P-N1 non-negative fit loses ≥0.05 coverage | **REFUTED 0/2** |
| P-N2 non-negative on ablated maps recovers ≥80% | 1/2 (76% Qwen3, 96% Qwen2) |
| P-N3 free-sign raw→ablated \|Δ\| ≤ 0.05 | 1/2 (−0.039 / −0.070) |
| 204d single constraint Σwα=0 | passes 2/2 but is not diagnostic (NNLS passes without it) |
| 204e P1 non-negative shows no end-task loss | **passes 2/2** |
| 204e P2 map-space free-sign clears the bar | **FAILS 0/2** (+6.3, +5.8, CIs span zero) |
| 205 P-T1c kept-values patch must be large | **FAILED, informatively** (all boundary values inert) |
| 205 P-T2 on Qwen2 | 2 of 3 conditions (agreement 96.3% vs 97% bar) |

The formal orthogonality premise (Prop. 3 of the old app:formal) failed on both models
(cos α,g = +0.431/+0.388) and is **not** claimed anywhere.

## 8. Instructions to Claude

**Highest priority — verify the consolidation.** Read the new `app:theory` end-to-end and make sure the
relabelling (your Prop 3 → Theorem 1, my props → Lemmas 1–2 and Proposition 3) reads correctly and that
your measured paragraphs were not mangled. The theorem and lemma proofs are mine; change them only if
you disagree with the statements.

**Do not reintroduce:**
- any causal claim that the *signs* of DWA's weights are the mechanism (the paper's old Fig. 3 caption
  and the "correction requires subtraction" sentence). The measured statement is in `FINDINGS §53` and
  `app:theory` (Theorem 1 + "what the fit adds").
- "flips no answers from L16" (it is 12%/5%; zero only from L20).
- "both policies are statistically indistinguishable from native" (DWA yes, AVR on Qwen3 only).
- "a tight crop discards the other object" as the cross-instance mechanism (refuted by §33/§36; the
  window contains both boxes on 93.4% of items).
- the prior-share diagnostic from phase 204b (it failed to track coverage; only the ablation evidence
  is used).

**Naming.** You renamed TWR→DWA and TSR→AVR in the paper. `FINDINGS.md` is now **mixed** (17
occurrences of DWA/AVR from your edits, the rest still TWR/TSR from mine). Decide one way and apply it
globally; I left FINDINGS alone to avoid a collision.

**Uncommitted state.** At the time of writing, `git status` shows `M FINDINGS.md`, `M paper/main.tex`,
plus modified figure scripts/PDFs that were already dirty before I started. I have committed nothing.
Review the diff, then commit in whatever grouping you prefer.

**Open items I did not do:**
1. The FINDINGS rename (above).
2. The paper's `app:claims` row "DWA beats every published rule" uses the pooled 9/9 contrasts — fine,
   but the per-stratum caveat lives in `tab:scope`; do not let the claims table read as if the method
   wins cross-instance everywhere.
3. No further GPU runs are queued. The GPU is back to the user's `train_kpt.py` job (18.6 GB); the 8B
   masking run fitted alongside it without trouble, so an 8B end-task run is feasible if you want one.
4. If you want the two-model sweep figure to replace `fig_layer_accuracy.pdf` everywhere, grep for the
   old filename — I changed the `app:vis` reference only.

**Reproducibility inventory (scripts → data → logs):**

| script | data | log |
|---|---|---|
| `phase204_theory.py` | `phase204_theory_{qwen3,qwen2}.json` | `q_204_theory.log` |
| `phase204_endtask.py` + `_analyze.py` | `phase204_endtask_{qwen3,qwen2}.jsonl` | `q_204e_chain.log` |
| `phase205_tsr_probe.py` + `phase205_analyze.py` | `phase205_tsr_probe_{qwen3,qwen2}.jsonl` | `q_205_queue.log` |
| `phase206_probe_loc.py` | `phase206_probe_loc_{qwen3,qwen2}.json` | `q_206_probe_loc.log` |
| `fig_layer_accuracy2.py` | `paper/figs/fig_layer_accuracy2.{pdf,png}` | — |
| `phase203_patch_lens.py` (re-run to N=191) | `phase203_patchlens_*.jsonl` | `q_205_queue.log` |
| `phase176_readout_sensitivity.py qwen2` | `phase176_sens_qwen2.json` | `q_final_queue.log` |
| `phase199_layersweep.py qwen2` | `phase199_layersweep_qwen2.jsonl` | `q_final_queue.log` |
| `phase176c_transport.py qwen3_8b` | `phase176c_transport_qwen3_8b.json` | `q_207c_queue.log` |

Queue launchers: `scripts/run_204e_chain.sh`, `run_205_queue.sh`, `run_final_queue.sh`,
`run_207c_queue.sh` (all used `setsid nohup … & disown`; plain `nohup &` gets killed when the calling
shell times out).

**Build command.** `pdflatex`/`latexmk` in the conda env are broken (missing `pdflatex.fmt`). Use:
```
cd paper && ~/.local/bin/tectonic -X compile main.tex --keep-logs
```

**One process-safety note.** Twice in this session a `pkill -f <pattern>` killed the caller because the
pattern appeared in the caller's own command line. Kill by PID, or exclude ancestry.
