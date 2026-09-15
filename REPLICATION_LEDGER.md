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
| **A learned re-ranking head improves proposals** | **Q3, Q2** | 39.3→52.9% (+13.6) / 35.1→44.0% (+8.9) |
| **DCR beats the vanilla VLM** | **Q3, Q2** | **+12.0pp [+4.2,+19.9]** / **+10.5pp [+2.6,+18.3]** — both CIs clear |
| **DCR beats random placement** | **Q3, Q2** | +28.3pp / +20.9pp [+11.5,+30.4] |
| **The encoding cliff** | **Q3, Q2** | step at 0.15–0.25 merged tokens, oracle flat across it (§13C) |
| **The serialization sink** | **Q3, Q2, OV, NX** | 3.4–4.5× implicit, 2.0–2.3× on `image_newline` (§5) |
| **No published crop policy beats the budget axis at matched tokens** | **Q3, Q2** | §9A/§9B |

---

## ❌ REJECTED — single-model, failed replication. Do not claim.

| claim | status |
|---|---|
| **"The final layer is anti-correlated with the target"** | Q3 0.529, **Q2 0.416 (better than chance)**. Cut. |
| **"Signed contrast is the mechanism"** | On Q2 the learned combination = best single layer **exactly** (43.5% = 43.5%). Cut as a general claim; retained only as a Q3 observation if §2.3 survives review. |
| **"Signed contrast helps outside Qwen3-VL crop placement"** | **Second independent failure.** §14K: on Q2 the learned combination = best single layer exactly. §14L: on token pruning, linear vs plain block-mean is +2.6/−1.0/+1.0, all null. **Finished as a general claim.** |
| **"Rank-only multi-crop beats single-crop"** | Predicted +5.5pp from the coverage exchange rate; measured **−3.1pp**. The coverage model has no term for distractor cost. ⚠ prompting confound under test (phase 81b). |
| **"No single layer is a good localiser"** | False on Q2 — L21 alone beats the block mean by 8.4pp and all 5 folds pick it. Cut. |
| **"DCR beats the compute-matched budget baseline"** | **Never survived on EITHER model.** Q3 +4.7pp [−3.7,+13.1], Q2 +3.1pp [−5.2,+11.0]. Only a V\*Bench single-region stratum ever cleared zero. **Remove from METHOD.md.** |
| **"DCR beats the argmax proposer it replaces"** | Q3 **+8.4pp [+2.6,+14.7]** ✔, Q2 **+4.7pp [−1.0,+11.0]** ✗. Direction consistent, magnitude halved, significance lost. Demoted to provisional; cannot be claimed pooled. |

---

## ⏳ PROVISIONAL — single-model, replication IN PROGRESS or REQUIRED

| claim | models | status |
|---|---|---|
| ~~DCR end-task gain vs vanilla~~ | **RESOLVED → ✅** | Survived on Q2 (+10.5pp). Moved above. |
| **DCR vs the argmax proposer** | Q3 only | Q2 gives +4.7pp with CI spanning zero (n=191). Needs more power or a third model to settle. |
| **Answer formation restored at L21 (§14I)** | Q3 | Needs the logit lens on Q2. The mechanism section rests on this. |
| **Zero-shot transfer to HR-Bench (§14E)** | Q3 | Needs a Q2 head transferred to HR-Bench. |
| **Depth profile worth +7.3pp; sink indicators worth +0.0pp (§14C(b))** | Q3 | Ablation must be re-run on Q2. |
| **Evidence region is causally live (3.32×, −10.5pp) (§14H)** | Q3 | Masking experiment must be re-run on Q2. |
| **Seven internal interventions null (§10, §14H)** | Q3 | The *pattern* is single-model. At minimum the decisive arms (attention amplification, contrastive decoding) need Q2. |
| **Coverage as mediator / sign of the allocation effect (§6D)** | Q3, Q2 (partial) | Q2 replication was −9.4 / +41.8pp; confirm it covers the same strata. |
| **Learned-sizer headroom +11.0pp (phase 78)** | Q3 | Future work; must not be claimed as general. |
| **★ Pruning at layer 2 is worse than random; late read-out loses nothing at 10% keep (§14L)** | Q3 | **phase 83 running on Q2.** Sharp prediction: Q2's early layers are worse (gt_pct 0.620 vs 0.456), so the penalty should be **LARGER**. A smaller penalty refutes the mechanism. **This is now the paper's strongest result — its replication matters most.** |

---

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
