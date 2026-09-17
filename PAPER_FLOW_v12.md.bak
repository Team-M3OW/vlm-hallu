# Paper flow (v12) — Read attention at the right depth

> **What changed from v11.** v11 led with "averaging dilutes the read-out" and carried DCR as the
> method. Two things happened. (1) The strongest result is now **§14L**: pruning visual tokens by
> **layer-2** attention — FastV's default — is **worse than random**, while a late read-out at the
> same budget loses **nothing**. That is on someone else's task, against their baseline, with a fix
> that needs no training. (2) **Signed contrast was rejected** as a general mechanism after failing
> twice outside Qwen3-VL crop placement (§14K, §14L).
>
> So the method is no longer "our learned head." It is **read attention late, not early** — one
> principle, two instantiations, one mechanism. Backups: `PAPER_FLOW_v11.md.bak` and v5–v10.

---

## The claim

> Vision-language models decide *what to look at* — which tokens to keep, where to crop — by reading
> attention. **The field reads it at the wrong depth.** Question-conditioned localisation is
> *constructed* inside the language model and does not exist before it: the vision encoder is at
> chance, layer 2 is worse than random, and the answer only forms around L21. Reading late instead
> lets you **delete 90% of visual tokens at zero cost** where the standard choice loses **22
> points**, and raises crop placement by **+12.0 / +10.5pp** on two architectures.

---

# PART I — METHOD

## §1 — The principle

Rank visual tokens by attention **aggregated over late layers**, never by an early layer and never
by the vision encoder. One line. No training.

## §2 — ★★★ Instantiation A: token pruning (§14L)

V\*Bench, n=191, every arm pruning the **same** token count.

| keep | random | **layer-2 (FastV default)** | **late read-out** | no pruning |
|---|---|---|---|---|
| **10%** | 38.2% | **34.6%** | **56.0–58.6%** | 56.5% |
| 25% | 41.4% | 40.3% | 57.1–58.1% | 56.5% |
| 50% | 47.1% | 52.9% | 55.5–56.5% | 56.5% |

- **90% of visual tokens are deletable at zero cost**: −0.5pp [−5.8,+4.7] vs no pruning.
- **Layer-2 ranking loses 22.0pp [−29.8,−14.7] and scores below random selection.**
- Margin: **+24.1pp [+16.2,+31.9]** at 10% keep, **+16.8pp [+8.9,+24.6]** at 25%.
- **No learning required**: the trained signed head and a plain block mean are indistinguishable
  (+2.6 / −1.0 / +1.0, all null).

⏳ *Single model; phase 83 replicating on Qwen2-VL with a falsifiable prediction (its early layers
are worse, so the penalty should be larger).* ⚠ *FastV is reimplemented on a common backbone, not
run from its released checkpoint.*

## §3 — Instantiation B: crop placement (DCR, §14C/D/80)

A learned re-ranking head over the per-layer profile replaces the block-mean argmax.

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| evidence coverage | 39.3 → **52.9%** | 35.1 → **44.0%** |
| accuracy vs vanilla | **+12.0pp [+4.2,+19.9]** | **+10.5pp [+2.6,+18.3]** |
| vs random placement | +28.3pp | +20.9pp [+11.5,+30.4] |

⚠ **Stated up front: DCR does not beat spending the same budget uniformly** (+4.7pp [−3.7,+13.1] and
+3.1pp [−5.2,+11.0], CI spanning zero on both). Rank-only multi-crop was the attempt to fix that and
**failed** (−3.1pp, §81). This instantiation is the weaker of the two and is presented as such.

---

# PART II — FINDINGS

## §4 — Averaging dilutes the read-out (2 architectures)

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| deployed block mean | 39.3% | 35.1% |
| best repair | **45.5%** | **43.5%** |
| **gain** | **+6.2pp** | **+8.4pp** |
| mean of ALL layers | 36.6% | **21.5%** |

**Why:** a large fraction of layers are anti-correlated with the target — Qwen3's *final* layer
(gt_pct 0.529 > 0.500 chance), Qwen2's *early and mid* layers (0.620 / 0.515), **15/28 worse than
0.45**. Which layers differ; *that some are* replicates.

⚠ **Rejected:** "the final layer is anti-correlated" (Q3 only) · "signed contrast is the mechanism"
(failed twice outside Q3 crop placement) · "no single layer is a good localiser" (false on Q2).

## §5 — The encoding cliff (2 architectures)

Below ~0.15–0.25 merged tokens of target extent, accuracy is **19.0%** — *below the 25% chance
level* — while the same question on the resolved region is answered **96–100%** correctly.

## §6 — The allocation ledger: what does not work

Four attempts to **regress a continuous geometric quantity** from internals, all failed: budget from
a *perfect* size oracle (loses to flat uniform, 11/12 cells) · budget from any free signal (9.4%
exact-rung vs a 35.6% majority baseline) · window size from target size (best W identical in all
four size quartiles) · attention-space reallocation (16% of the crop). **Ranking cells works;
regressing scale does not.**

---

# PART III — MECHANISTIC INTERPRETABILITY

## §7 — ★★★ Localisation is built late, and everything earlier is blind

Three independent measurements, one cause:

| where you read | result |
|---|---|
| **vision encoder** (before the LM) | every arm **at or below the 2.3% chance rate** (§14G) |
| **layer 2** (FastV's choice) | pruning **worse than random** (§14L) |
| **L21** | the answer forms **abruptly**, +15.3pp in one layer (§14I) |

The vision tower cannot localise because **it never sees the question**. There is no
question-independent salience that suffices when the target is small and the question selects among
candidates.

## §8 — Why the crop works: restored answer formation (§14I)

Four arms, identical budget, only pixels differ. **Separation is ≈0.0pp through L20, then +15.3pp at
L21.** Failing arms jump at **L2** (the answer prior) and never improve — uniform's max over all 28
layers *equals* its final answer. Working arms jump at **L21**, and the step scales with evidence
supplied.

**Causal control** — same arm, split by whether its window delivered:

| | head | uniform | oracle |
|---|---|---|---|
| **covers** | **90.1%** | 60.4% | 98.0% |
| **misses** | 43.8% | 51.7% | 80.9% |

A **37.6pp swing**. ⏳ *Single model.*

## §9 — Why no internal fix exists

**Seven interventions, all null** — sink suppression, attention amplification, residual steering,
layer read-out, DoLa/DeCo, component ablation, and **contrastive decoding steered to the
ground-truth region** (ceiling +2.6pp [−0.5,+5.8]). The seventh is decisive: it aimed *perfectly*,
proved the target is causally live (masking it shifts logits **3.32×** more than a random region and
costs **10.5pp**), **and still could not convert it.**

> The model looks in the right place; the right place is load-bearing; it still cannot answer,
> because no operation on the residual stream can supply information the encoder never wrote.

---

## §10 — Limitations and the replication ledger

Every claim is tagged by model count in `REPLICATION_LEDGER.md`; **a claim enters the paper only
from the survived section.** Currently **6 survived** (≥2 models), **5 rejected**, **8 provisional**.

Retractions kept in the paper: the §12D double-normalisation bug · a "does not transfer" call read
from a category-ordered prefix · two mechanisms proposed for a failure that was not real · a capture
fraction measured against a restricted ceiling (65% → **27.6%**) · `peak` computed without the ring
mask (0.576 → **0.745**) · and three claims rejected by cross-architecture replication.
