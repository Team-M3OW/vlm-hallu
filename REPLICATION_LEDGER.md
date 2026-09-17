# Replication ledger

**Standard adopted 2026-09-16:** every claim in the paper must hold on **more than one model**.
A claim that does not survive is **REJECTED**, not caveated. This file is the gate — a claim may
enter the paper only from the ✅ section.

Models: **Q3** = Qwen3-VL-2B · **Q2** = Qwen2-VL-7B · **OV** = LLaVA-OneVision-7B ·
**NX** = LLaVA-NeXT-7B

---

## ✅ SURVIVED (≥2 models) — may be claimed

| claim | models | evidence |
|---|---|---|
| **Averaging across depth dilutes the read-out** | **Q3, Q2** | fixing it is worth +6.2pp (Q3) / +8.4pp (Q2); averaging all layers → 36.6% / 21.5% |
| **Some layers are anti-correlated with the target, and averaging them in is what hurts** | **Q3, Q2** | Q3 final layer 0.529; Q2 early 0.620 / mid 0.515 — 15/28 layers worse than 0.45 |
| **★ A learned re-ranking head improves proposals — 3 OF 4, TWO FAMILIES** at the deployed W=0.15: Q3 +14.1 ✔, Q2 +8.9 ✔, **LLaVA-NeXT +7.3 [+2.1,+12.6] ✔**, LLaVA-OneVision −1.0 ✗ (the one model whose best single layer equals its block mean, §14N). Max-over-layers is only **1 of 4** at the same window (§14Y corrected) | **Q3, Q2, NX** | 39.3→52.9% (+13.6) / 35.1→44.0% (+8.9) — ⚠ vs the *deployed* argmax; see §14V for the stronger baseline |
| **★ MAX beats MEAN as the cross-layer aggregation rule** | **Q3, Q2** | max over block **+7.3 [+2.6,+12.6]** / **+6.8 [+3.1,+11.0]**; CLAA 4-layer window-max **+8.4 [+3.1,+13.6]** / **+12.0 [+6.8,+17.3]**. Training-free, one line (§14V) |
| **DCR beats the vanilla VLM** | **Q3, Q2** | **+12.0pp [+4.2,+19.9]** / **+10.5pp [+2.6,+18.3]** — both CIs clear |
| **DCR beats random placement** | **Q3, Q2** | +28.3pp / +20.9pp [+11.5,+30.4] |
| **The encoding cliff** | **Q3, Q2** | step at 0.15–0.25 merged tokens, oracle flat across it (§13C) |
| **The serialization sink** | **Q3, Q2, OV, NX** | 3.4–4.5× implicit, 2.0–2.3× on `image_newline` (§5) |
| **No published crop policy beats the budget axis at matched tokens** | **Q3, Q2** | §9A/§9B |
| **★★★ Layer-2 pruning is BELOW RANDOM — 2 MODELS × 3 BENCHMARKS** | **Q3, Q2 × V\*, POPE, MMBench** | layer-2 − random: V\* −3.7/−4.7, POPE −15.0/−11.0, MMBench −25.5/−5.5; late − layer-2 clears in all six cells (§14L(b), §14P, §15H) |
| **★★ DCR is the ONLY arm that clears the compute-matched bar — beating a training-free max-aggregated read-out on both models** | **Q3, Q2** | head−bar single-object +15.7 [+6.1,+25.2] / +11.3 [+1.7,+20.9] ✔; max_win4−bar +9.6 [+0.0,+19.1] / +6.1 [−3.5,+15.7] ✗; head−max_win4 +6.1 / +5.2, both lower bounds at zero (§14W(b)) |
| **★★★ DCR beats the compute-matched baseline on SINGLE-OBJECT questions — 2 MODELS × 2 BENCHMARKS**: V\*Bench +15.7 / +11.3; HR-Bench 4k **+7.5 [+2.2,+12.5] / +8.5 [+3.8,+13.2]**, n=400 each, W=0.15 unswept. Argmax clears in no cell; relational loses in every cell; pooled null in every cell (§15E, §15H) | **Q3, Q2 × V\*, HR** | **+15.7pp [+6.1,+25.2]** / **+11.3pp [+1.7,+20.9]**, n=115 each, W transferred not selected. Null on relational (−3.9 / +0.0) exactly as §6D predicts. ⚠ pooled does not clear; both numbers must be reported (§14T) |
| **Oracle placement wants a tighter window than learned placement does** | **Q3, Q2** | oracle monotone decreasing in W (90.1→65.4 / 91.1→80.6); head peaks at 0.25 on both (§14T) |

---

## ❌ REJECTED — single-model, failed replication. Do not claim.

| claim | status |
|---|---|
| **"The final layer is anti-correlated with the target"** | Q3 0.529, **Q2 0.416 (better than chance)**. Cut. |
| **"Signed contrast is the mechanism"** | On Q2 the learned combination = best single layer **exactly** (43.5% = 43.5%). Cut as a general claim; retained only as a Q3 observation if §2.3 survives review. |
| **"Signed contrast helps outside Qwen3-VL crop placement"** | **Second independent failure.** §14K: on Q2 the learned combination = best single layer exactly. §14L: on token pruning, linear vs plain block-mean is +2.6/−1.0/+1.0, all null. **Finished as a general claim.** |
| **"Rank-only multi-crop beats single-crop"** | Predicted +5.5pp from the coverage exchange rate; measured **−3.1pp**. The coverage model has no term for distractor cost. ⚠ prompting confound under test (phase 81b). |
| **"No single layer is a good localiser"** | False on Q2 — L21 alone beats the block mean by 8.4pp and all 5 folds pick it. Cut. |
| ~~**"DCR beats the compute-matched budget baseline"** (pooled)~~ | **Reworded, not un-rejected.** At the transferred W=0.25 it is Q3 **+7.9pp [−0.5,+16.2]**, Q2 **+6.8pp [−1.0,+14.7]** — two positive estimates with lower bounds inside a point of zero. That is **NOT DEMONSTRATED at n=191**, not refuted. The old numbers compared Q3 at a held-out W against Q2 at a window never swept on it (§14T). Pooled still may not be claimed. |
| **"DCR beats the argmax proposer it replaces"** | Q3 **+8.4pp [+2.6,+14.7]** ✔, Q2 **+4.7pp [−1.0,+11.0]** ✗. Direction consistent, magnitude halved, significance lost. Demoted to provisional; cannot be claimed pooled. |
| **"The learned head beats a properly aggregated training-free baseline"** | Q3 **+8.4pp [+3.1,+13.6]** ✔, Q2 **+3.1pp [−1.0,+7.3]** ✗ against CLAA's 4-layer window-max. **1 of 2 — rejected.** The head still beats max-over-block on both (+9.4 / +8.4). §14V |

---

## ⏳ PROVISIONAL — single-model, replication IN PROGRESS or REQUIRED

| claim | models | status |
|---|---|---|
| ~~DCR end-task gain vs vanilla~~ | **RESOLVED → ✅** | Survived on Q2 (+10.5pp). Moved above. |
| **DCR vs the argmax proposer** | Q3 only | Q2 gives +4.7pp with CI spanning zero (n=191). Needs more power or a third model to settle. |
| **Answer formation restored at L21 (§14I)** | Q3 | Needs the logit lens on Q2. The mechanism section rests on this. |
| ~~Zero-shot transfer to HR-Bench (§14E)~~ | — | **Superseded by §15E**: 72b's localiser stripped the options. With the full prompt the head beats the bar on single-region. |
| **Depth profile worth +7.3pp; sink indicators worth +0.0pp (§14C(b))** | Q3 | Ablation must be re-run on Q2. |
| **Evidence region is causally live (3.32×, −10.5pp) (§14H)** | Q3 | Masking experiment must be re-run on Q2. |
| **Seven internal interventions null (§10, §14H)** | Q3 | The *pattern* is single-model. At minimum the decisive arms (attention amplification, contrastive decoding) need Q2. |
| **Coverage as mediator / sign of the allocation effect (§6D)** | Q3, Q2 (partial) | Q2 replication was −9.4 / +41.8pp on V\*Bench. ⚠ The **W-by-category interaction on HR-Bench is 1 of 2** (−11.9 ✔ / −3.0 ✗, §15G). Claim coverage on V\*Bench only. |
| ~~Learned-sizer headroom +11.0pp (phase 78)~~ | — | **CLOSED.** The OOF sizer failed its pre-registered primary on Qwen2-VL (+3.7pp [−3.1,+10.5] vs the bar, **−3.1pp** vs the constant) and had already inverted across grids on Qwen3-VL. Dead on 2 models, 2 grids (§14U). |
| **Label-free locator finds the read-depth transition (§14S)** | Q3, Q2 | **Region ✔, exact layer ✗.** Divergence rises L13–L14; damage recovers L14–L16. L14 does not clear zero (+4.7 [−1.0,+10.5]); L16 does (+11.0 [+3.7,+18.8]). Claim only the region. |
| **★ Pruning at layer 2 is worse than random; late read-out loses nothing at 10% keep (§14L)** | Q3 | **phase 83 running on Q2.** Sharp prediction: Q2's early layers are worse (gt_pct 0.620 vs 0.456), so the penalty should be **LARGER**. A smaller penalty refutes the mechanism. **This is now the paper's strongest result — its replication matters most.** |

---

## 🛑 HARD SCOPE BOUNDARIES (measured, not assumed)

| boundary | evidence |
|---|---|
| **Relational questions — the first 2 MODEL × 2 BENCHMARK claim in the project** (a limit, not a gain) | V\*Bench −3.9 / +0.0pp; HR-Bench argmax@0.15 **−10.5 / −32.8pp** vs the bar (§14T, §15G) |
| **Existence questions — crop-based allocation is INVALID** | POPE: crop −56.7 / −77.3 / −63.0pp vs the bar; the crop answers "no" to everything because a 6%-area window is evidence of absence. The oracle arm is **circular** there (knowing the box = knowing presence), so POPE cannot even provide a ceiling (§15B) |
| **LLaVA-OneVision** — no layer disagreement to exploit | head −1.0pp; its best single layer equals its block mean (§14N, §14Y) |
| **MMBench** — 512px, no budget axis | not run; reported as NO AXIS, as in phases 25/28 |

## ⛔ INVALIDATED BY THE BARE-QUESTION LOCALISER (§15I) — do not cite

§5A/§5B (ph 33) · §8A 4K gating (ph 46) · §9B *our* arm only (ph 53; prior-art losses stand) · §9C all four 4K attempts (53–56) · §9D scale sweep (57) · §11B multi-crop at 4K (59) · §14E (72b). Replaced by §15E/§15H.

## 🚧 BLOCKED

| item | blocker |
|---|---|
| LLaVA-OneVision / LLaVA-NeXT as a third point | `image_newline` separators break the clean-grid assumption; phase 74's extractor skips rather than guesses. **Fix:** port phase 41's separator detection (matches the model's own `image_newline` parameter against merged embeddings), strip separators, rebuild the grid from the modal row length. |

---

## Rules that produced this ledger

1. **Fold-validate every selection.** In-sample "best layer" read 41.9% on Q3 and collapsed to
   36.1% out-of-fold — *below* the block mean it appeared to beat. Best-of-28 on n=191.
2. **Never present one model's numbers as both.** §2.1 briefly carried Q3's gt_pct table as if it
   covered Q2; Q2's profile is completely different (0.620/0.515/0.392 vs 0.456/0.413/0.409).
3. **The pre-registered rejection rule was applied as written.** Phase 80b fixed "head must beat
   argmax with CI clear of zero" before the run; it came out +4.7pp [−1.0,+11.0] and the claim was
   demoted, not re-argued. The internal control (+0.0pp [+0.0,+0.0], n=70) rules out a pipeline
   fault as the explanation.
4. **A replication that fails is a result, not a setback** — §14K's failure produced a better-specified
   general claim (anti-correlated layers exist on both; *which* ones differs) than the one it killed.
