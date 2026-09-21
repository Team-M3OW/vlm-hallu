# Pre-registration — mechanistic grounding and formal theory for TWR and TSR

Written 2026-09-21, before any phase-204+ run. Governs phases 204–207. Extends the requests of
2026-09-20 (interpretability + causality for the claims; why both methods work by logit lens /
attention readouts / linear probes; theoretical grounding by theorems + proofs validated on data).

The audit finding that motivates this file: *"TWR's negative weights cancel the nuisance"* is today
supported only on **coverage** (§50A/§50B, CPU, no end-task arm), and the pre-registered per-layer
nuisance-loading test **failed 1 of 2** (§49). *"TSR pruning is free because tokens are inert"* is
supported behaviourally at 900 tokens (§26C/§38) but at the **value level only at 300 tokens, n=60,
single stratum** (§52). Neither method has a formal statement. All three are addressed below.

Standing rules apply: two-model rule (a claim must hold on Qwen3-VL-2B and Qwen2-VL-7B or it is
reported rejected), paired item bootstrap 8000 resamples, learned components out-of-fold
GroupKFold(5) by item × 3 seeds (700/701/702), gate every claim on end-task accuracy or a
distributional test, report negatives with their diagnoses.

---

## 0. Notation and empirical objects

- `A_l^(i)(c)`: layer-`l` attention from the final prompt token to image cell `c` for item `i`,
  averaged over heads, `l = 0..27`, `c = 1..C`. Row-normalised per layer (`A_l / sum_c A_l`) as in
  phases 201/202; features are the deployed TWR features:
  `φ = [log A_0..log A_27, rank_0..rank_27, log nb, r_center, dist_edge, fx, fy, is_last_col,
  is_last_row]` (63 columns).
- `m_l = E_i A_l^(i)`: the **item-mean map** (the item-independent component). `m_l` is computed on
  training items only when a rule is fit.
- `B = {16..26}` (Qwen3) / `{15..26}` (Qwen2): the block-mean band.
- `t^(i)(c) ∈ {0,1}`: GT-box indicator; `y^(i)(c) = coverage of the W=0.25 window centred on c`
  (the ridge label, `COV_HIT=0.5`).
- Deployed TWR: OOF ridge with α=1, unpenalised intercept, on standardised `φ`.
- Nuisance-ablated maps (definition fixed in §49): `Ã_l = A_l − ⟨A_l, m_l⟩/⟨m_l,m_l⟩ m_l`, then
  shift-to-positive and renormalise (as in `phase201_ablate_nuisance.py`).

---

## 1. Formal statements (theorems)

### Proposition 1 (leakage decomposition; trivial but load-bearing)
For any score linear in the maps, `S_w = Σ_l w_l A_l`, with `A_l = m_l + U_l` and `E_i U_l = 0`,
`S_w = Σ_l w_l m_l + Σ_l w_l U_l`. The first term is item-independent: its arg-max and its ranking
of cells are the same for every item. Call `L(w) = Σ_l w_l m_l` the **leakage**.
The block mean's leakage is `L(1_B/|B|) = (1/|B|)Σ_{l∈B} m_l`. ∎

### Proposition 2 (non-negative filters cannot cancel a common prior)
If the item-mean maps are rank-one across depth, `m_l = α_l μ` with `α_l > 0` and `μ ≠ 0`, then
`L(w) = (αᵀw) μ`. Hence `L(w) = 0` iff `αᵀw = 0`. In particular no non-zero filter with `w ≥ 0`
can have zero leakage; since `α > 0` componentwise, `αᵀw > 0`. Cancelling a common prior requires
**mixed signs**. ∎

*Corollary (the NNLS prediction).* If TWR's advantage over the block mean is the cancellation of a
common prior, then constraining its log-attention weights to be non-negative must remove most of
that advantage, and must do so less on nuisance-ablated maps, where Proposition 1's leakage term has
already been removed by construction. This is the **2×2 decisive experiment** of phase 204.

### Proposition 3 (least-squares filters orthogonalise to the common prior)
Per item, model the layer profile at a cell as `x = α m + g t + ε`, with scalar prior loading `m`
and target indicator `t`, per-layer nuisance coefficients `α`, signal gains `g`; target `y = t`.
Assume `E[m] = E[t] = 0`, `E[m t] = 0`, `E[ε] = 0`, `Cov(ε) = σ²I`, and `αᵀg = 0`. Then the
population least-squares filter is `w* = Σ_xx^{-1} σ_t² g` with
`Σ_xx = σ_m² ααᵀ + σ_t² ggᵀ + σ²I`. Since `αᵀg = 0`, the resolvent does not mix the two directions:
`Σ_xx^{-1} g = c g`, hence `w* = c σ_t² g` and
`αᵀw* = c σ_t² αᵀg = 0`: the least-squares filter has **exactly zero leakage**. The equal-weight mean
`w_B = 1_B/|B|` has `αᵀ w_B > 0` whenever the α are positive, in general non-zero. ∎
(Ridge adds `λI` to `Σ_xx` and preserves the argument.)

*Assumptions are tested, not assumed:* rank-one structure of `{m_l}` (leading eigenvalue share of
the layer-mean covariance), `αᵀg` (cosine between the per-layer nuisance loadings and the per-layer
coverage gains), and `E[m t]` (correlation of the item-mean map with the GT indicator). If an
assumption fails on a model, that model's results are reported as qualified.

*Prediction (single-constraint test).* If Proposition 3's mechanism is what TWR implements, then
imposing the single linear constraint `Σ_l w_l α_l = 0` (in raw-feature scale) on an otherwise
identical OOF ridge should recover most of TWR's advantage over the block mean. Recovering the
advantage with a **one-dimensional** constraint is the strongest available evidence that the filter's
function is nuisance orthogonalisation, not 28 independent knobs.

### Proposition 4 (TSR identity is accounting, given pruning preservation)
Let `U_E` be the unpruned model at encode budget `E`, `P_D` the operation that makes visual token
set `D` invisible from layer `p+1` on, and `TSR = U_E ∘ P_D`. If `P_D` is output-preserving — the
option distribution is unchanged, for all items — then `acc(TSR) = acc(U_E)`, and with the
equal-compute bar `U_600`, `gain(TSR) = acc(U_900) − acc(U_600) = headroom`. ∎
*Lemma (budget).* With `E=900, p=16, |keep| = 0.10·E = 90, L=28`:
`token-layers = 900·17 + 90·11 = 16,290 ≤ 16,800 = 600·28`. ∎
The empirical content is the premise; §39's slope 1.01 / r=0.966 is its current validation, and
phase 205 tests the premise at the value level.

### Proposition 5 (selection irrelevance past the boundary)
Suppose replacing the hidden values of any subset of visual positions at layer `p` with arbitrary
donor values leaves the option distribution unchanged. Then for any two keep-sets `D₁, D₂` of the
same size, `U_E ∘ P_{D₁}` and `U_E ∘ P_{D₂}` induce the same distribution; every ranking rule
(attention, random, learned) is equivalent. ∎
The premise is exactly what §52's E1 tests at 300 tokens and phase 205 tests at 900.

---

## 2. Experiments and pre-registered predictions

### Phase 204 — TWR formal grounding (CPU; + end-task GPU confirmation)
Data: the modal-grid item sets of phases 201/202, both models, raw and nuisance-ablated maps.

- **204a (assumption tests).** Report: leading eigenvalue share of `{m_l}` (rank-one check);
  per-layer `α_l`; per-layer `g_l = cov(A_l, t) / ||m_l||·||t||` (descriptive); `αᵀ ĝ` and
  `corr(m_l, t)`.
- **204b (leakage of the deployed filter).** Prior-only score: evaluate each fitted rule's score
  function on the maps with `A_l := m_l` for all `l` (the "mean item"), and report
  `π = Var_c(s_prior) / E_i[Var_c(s^(i))]`, the share of score variance that is item-independent.
  *Prediction:* `π_TWR ≤ π_block / 3` on both models. *(Direction pre-registered; the factor 3 is
  the pass bar.)*
- **204c (2×2 NNLS).** Refit the deployed feature set with per-variable bounds: the 28 log-A
  coefficients `≥ 0`, all others free (`scipy.optimize.lsq_linear`, same folds/seeds/α=1,
  standardisation inside each training fold). Evaluate OOF coverage on raw and on ablated maps,
  against block mean and free-sign TWR on the same maps.
  *Predictions, both models required:*
  - P-N1: NNLS on raw maps loses `≥ 0.05` mean-cov against TWR; its coverage is within `0.05` of the
    block mean.
  - P-N2: NNLS on ablated maps recovers `≥ 0.80` of free-sign TWR's advantage over the block mean on
    those maps.
  - P-N3 (control): free-sign TWR's own raw→ablated movement is `|Δ| ≤ 0.05` (replicates §50A Q2).
- **204d (single-constraint ridge).** Fit with the linear constraint `Σ_l w_l^{raw} α_l = 0` by
  elimination in standardised space (`β_l = α_l/sd_l`), same protocol.
  *Prediction:* recovers `≥ 0.80` of TWR's advantage over the block mean on raw maps, both models.
- **204e (end-task confirmation).** Arms `nnls@600`, `constraint@600` in the phase-184 pipeline,
  same items as the published TWR/bar arms, both models.
  *Prediction:* NNLS is significantly below TWR, direction as coverage; constraint is within noise
  of TWR. *(If coverage passes but end-task does not, the claim is reported as coverage-only, per
  Appendix B's rule.)*

### Phase 205 — TSR formal grounding at its operating point (GPU)
`phase205_tsr_probe.py`, V*Bench n=191, both models, E=900, p=16, keep 10%.
Per item: one unpruned pass; TSR with attention keep; TSR with random keep; on the **unpruned** run,
hidden-state patching at the boundary (donor = a different image, same question):
patch the *dropped* cells at L16; patch the *kept* cells at L16; patch the *dropped* cells at L8
(control); patch the dropped cells at L24 (second control). KL over option distributions vs the
unpruned base, and arg-max flips.

- **P-T1 (premise at the operating point).** `KL(patch dropped @L16) ≤ 0.01` (the phase-203
  txt-L4 floor) on both models, against `KL(patch kept @L16) ≥ 0.10` and
  `KL(patch dropped @L8) ≥ 0.10` in the same run. This is §52's E1 moved to 900 tokens and given its
  own in-run contrast.
- **P-T2 (pruning preservation).** TSR's option distribution ≈ unpruned 900: mean KL `≤ 0.02`,
  arg-max agreement `≥ 97%`, both models; attention keep ≈ random keep (difference in KL `≤ 0.01`).
- **P-T3 (identity re-fit).** With unpruned@900 and @600 accuracies on these items, re-fit
  gain-vs-headroom; report slope and r; the existing 10-cell fit is re-checked, not replaced.

### Phase 206 — §52 extended to the full benchmark, both strata (GPU)
Re-run `phase203_patch_lens.py` to N=191 (the script appends; the existing 60 rows are skipped) for
both models. Report E1 and E3 per stratum. *Prediction:* both probes' verdicts survive; cross-
stratum E1 at L≥16 stays at the floor. This removes the single-stratum limitation of §52.

### Phase 207 — closing the two audit gaps that are interpretability claims (GPU)
- **207a.** Run the §176 answer-row-only mask on **Qwen2-VL-7B** (the leg that never landed):
  mask only the answer-emission token's row to the image, all layers, n=40. *Prediction:* KL and
  flip rate at the noise floor, as on Qwen3 (§20). This makes "the answer position does not read the
  image" a two-model claim, and is required for the paper's Table 1 caption to match its row.
- **207b.** Port the phase-199 read-out layer sweep to **Qwen2-VL-7B** (all 28 layers, n=191).
  *Prediction:* the same shape (flat inside the transport window, step at the boundary, peak in the
  read-out band, decay to the output); the Qwen3-specific ordering of L17 as the peak is not
  required. This makes the paper's headline (i) two-model.
- **207c.** Prefix/suffix **masking** on the 36-layer checkpoint (Qwen3-VL-8B), the §176c protocol,
  n=40. *Prediction:* suffix becomes harmless at `0.57·36 = L21` (matching the pruning-derived
  boundary), making the 0.57 law a masking result rather than a pruning result.

### Paper integration (after the runs)
- **One formal appendix** stating Propositions 1–5 with proofs and the validation tables.
- **A claims-to-mechanism audit table** (from the 2026-09-21 audit): each headline claim → its
  evidence class → models → section. The audit's confirmed wording faults are fixed in the same
  pass (in particular: "flips no answers" from L16 → L20; Table 1's mixed-source row;
  native-resolution "both" → TWR both / TSR Qwen3 only; "pruning free in 9 of 10" → §38's corrected
  wording; the scope-law mechanism sentence vs §33).
- The §49-forbidden "signed weights cancel the sink" sentence in Fig. 3's caption is replaced by
  whatever 204 establishes — or by the orthogonality statement if 204c/204d pass.

---

## 3. Decision rules

1. A proposition's validation passes only if its prediction holds on **both** core models.
2. 204c/204d gate on **OOF coverage** as the primary metric (comparable to §50A/B) and 204e on
   end-task; a coverage-only pass is reported as coverage-only.
3. No arm is adopted into the paper on a single seed; all learned rules use the 3-seed protocol.
4. If P-N1 fails (NNLS keeps the advantage), the signedness story is **rejected** and §50's open
   question is reported as still open — that outcome is as publishable as the positive one.
5. If P-T1 fails at L16 but holds at L24, TSR's free lunch is depth-limited and its justification
   moves to L24; the budget lemma is recomputed and reported either way.
6. Any exactly-zero contrast with a `[0,0]` CI is a pipeline fault, not a result.

## 4. Artifacts

`scripts/phase204_theory.py`, `scripts/phase204_endtask.py`, `scripts/phase205_tsr_probe.py`,
`logs/q_204_*`, `logs/q_205_*`, `FINDINGS §53–§56`, and the formal appendix in `paper/main.tex`.
