# Paper plan v2 — after Phases 8–16

**Rewritten 2026-09-07.** Supersedes plan v1 (which was built on "readout beats resolution", killed
by Phase 9). `FINDINGS.md` is the record of results; this is the forward plan.

---

## 0. The claim, in one sentence

> **VLMs allocate visual tokens uniformly over space and independently of the query.** When the
> queried object receives a fractional share of 2–3 tokens, the evidence it leaves is real but
> ~1000× too faint to overturn a confident "no" — and **no inference-time intervention can recover
> it**, because you cannot redirect attention to capacity that was never allocated. Reallocating
> the **same** token budget by query recovers most of the gap.

**Why this is architectural, not preprocessing (constraint #2):** the flaw is the *ordering* —
token layout is fixed by image geometry before the model sees the question. Cropping and upscaling
are pipeline hacks that accidentally exploit this; the claim is about the allocator.

**Why it isn't the high-resolution literature:** "more tokens helps" is already asserted by
AnyRes / dynamic-resolution work and is **not** a finding. "*The same* budget, allocated by query,
beats uniform" is a different claim — and it is the one the four nulls actually support.

### 0.1 Hierarchy decision (make this now, or reviewers make it for us)
`FINDINGS.md` currently presents **two** headline-grade effects that are in tension:
§4A/§4C context interference (0% → 27.8% at *constant* resolution) and §4C token budget
(0% → 72%). **Decision: ALLOCATION is the claim; context interference is a secondary effect**
reported as a modulator. Rationale: the allocation effect is ~2.6× larger, it carries the
dose-response curve, and all four nulls speak to it. Interference gets a subsection, not the
abstract.

---

## 1. What is already established (see FINDINGS §4B–4E)

| result | status |
|---|---|
| RePOPE enrichment 7.2× — 58% of the classic confident-error cohort is label noise | ✅ strong, and a field-level methodological finding |
| Token-budget dose-response: 0.4 tok→0%, 1.8→28%, 4.1→50%, 32→72% | ✅ with information-free upsampling as the clincher |
| Four nulls for inference-time redirection: pointing 0.0%, redbox 5.6%, attention patching (6e), steering (16) | ✅ all controlled; Phase 16's `rand_last` caught a false headline |
| Category-specific encoding exists but is faint: 0.71 AUROC @ L16 | ✅ same-image, size-matched control |
| Context interference: 0% → 27.8% at constant resolution | ✅ → demoted to secondary per §0.1 |

**The gap:** the paper is currently *four nulls plus a known lever*. §2 is what turns it into a claim.

---

## 2. E1 — THE CENTERPIECE: budget-matched query-conditional allocation

**Question:** at a **fixed total visual-token budget**, does allocating tokens by query beat
allocating them uniformly?

**Arms** (same items, RePOPE-clean confident denials + clean negatives for FP):

| arm | layout | total tokens |
|---|---|---|
| `uniform@B` | whole image, downsampled to hit B | **B** |
| `alloc_query@B` | low-res full image + high-res crop of the **GT** region | **B** |
| `alloc_random@B` | identical layout, **wrong region** | **B** |
| `crop_only@B` | region crop alone, sized to hit B | **B** |

Sweep **B ∈ {150, 300 (native), 600}**.

### Non-negotiables (each would sink the claim on its own)
1. **Match on MEASURED tokens, not intended.** `image_grid_thw` is already logged per arm — matching
   is on the *realized* total per item per arm, and realized totals get reported. An arm landing at
   340 vs 300 is not budget-matched.
2. **The uniform arm must sit at the same B**, not at whatever the native image happened to get. If
   the allocated arm lands under native, **downsample uniform to match** — do not let it keep a
   budget advantage.
3. **`alloc_random` is the deciding control.** Same non-uniform layout, same realized budget, wrong
   region. Query-conditional beating random-region ⇒ *allocation*. Failing to ⇒ "non-uniform layouts
   happen to help", a much weaker paper. This is Phase 16's `rand_obj` lesson applied at the input.
4. **Oracle framing stated explicitly.** The region comes from COCO GT, exactly as Phase 7. That is
   legitimate for a *mechanism* claim and is what keeps us out of re-inventing V*/SEAL's localizer.
5. Negative arm (genuine absences) for every cell — no recovery number without its FP rate.

---

## 3. E2 — CUB-200-2011: is the subject bigger than a 1.6% tail?

**This is load-bearing, not just generalization.** On RePOPE-clean data the phenomenon is
**1.6% of positives** (54/3387) — thin, and a reviewer will say POPE is saturated (P(yes) AUROC
0.9725). If the confident-error tail is substantially larger on fine-grained discrimination, the
paper's subject stops being a rounding error.

CUB also **separates object size from detail resolution** — birds fill much of the frame, so a
token-budget effect there cannot be "the object is tiny".

**Build deliberately (nothing inherits):** CUB has **no absent-object negatives**. The FP arm has to
be constructed — ask about a **same-genus confusable species** (hard negative) *and* a **random
other species** (easy negative), reported separately. Downloaded to `data/cub/`; ships
`bounding_boxes.txt`, which the allocation arms need.

---

## 4. E3 — cross-architecture (AFTER E1 lands, not before)

Cached: `Qwen2-VL-7B-Instruct`, `InternVL2-8B`, `llava-onevision-qwen2-7b-ov-hf`. Also tests scale
(2B → 7/8B): if the tail persists at 8B it is not a small-model artifact.

**Sequencing matters:** replicate the *finished* claim. The claim has changed form three times
today; replicating a claim not yet in final form means re-running everything when it changes again.

---

## 5. Explicitly dropped

- **Benchmarking published steering methods** (RUDDER/DMAS/Revis) on our cohort. Faithful
  reimplementation is a large lift, each has hyperparameters that decide whether it "works", and
  underperformance reads as *our* bad implementation, not a hard cohort. Phase 16 already makes the
  mechanism claim using the family's shared primitive (mean-difference steering) — cite that.
- POPE label-noise as a headline (RePOPE published it; we use it as a tool and cite it).
- Answer-by-grounding / channel disagreement (Phase 8).
- Any framing of crop/zoom/upscale as a contributed fix (prior art; they are *diagnostics* here).

---

## 6. Standing constraints (unchanged)
Public benchmarks only · preprocessing is never the headline · check prior art before pitching ·
report negatives honestly · every recovery number carries its false-positive rate · verdicts read on
discrimination, never raw recovery.
