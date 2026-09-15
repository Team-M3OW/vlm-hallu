# Paper flow (v11) — Attention read-outs destroy the signal they average

> **What changed from v10.** v10 was finding-led: "no policy beats the budget axis." That is still
> true and still in the paper, but it is no longer the claim. Between v10 and v11 we found the
> **defect** (depth averaging), built a **method** on it (+12.0pp), showed it **transfers zero-shot**,
> and established **why it works** (restored answer formation at L21). The paper is now
> mechanism-led with a method attached, which is the order the evidence supports.
>
> Backups: `PAPER_FLOW_v10.md.bak` and v5–v9.

---

## The claim

> Vision-language models decide *where to look* by averaging attention across a block of layers and
> taking the maximum. **That average is the defect** — measurably, on every architecture tested:
> repairing the read-out is worth **+6.2pp** of evidence localisation on Qwen3-VL-2B and **+8.4pp**
> on Qwen2-VL-7B, and averaging *all* layers is catastrophic on both (36.6% / 21.5%). Replacing it
> with a learned
> **signed** combination raises evidence localisation from **39.3% to 52.9%** and question-answering
> from **56.5% to 68.6%** (**+12.0pp [+4.2,+19.9]**), transfers **zero-shot** to a different
> benchmark at 2.7× the resolution, and works — as the logit lens shows — not by improving reasoning
> but by **restoring an answer-formation step at L21** that uniform encoding starves of input.

---

## §1 — The problem, and where the field actually loses

A VLM encodes an image into ~300 merged tokens. When the evidence is small it occupies a fraction of
one token and **nothing is encoded to read**: below ~0.15 merged tokens of target extent accuracy is
**19.0%** — *below the 25% chance level* — while the identical question asked of the resolved region
is answered **96–100%** correctly. Architecture-invariant: two models, one threshold (§13B/C).

The field's answer is to crop and look again. Every such method picks *where* from the model's
attention, and **that is where they lose**: the standard read-out lands on the evidence only
**39.3%** of the time, while on **88.5%** of items some cell in the same map would have worked.

⚠ Keep from v10: at matched realized budget no published crop/zoom policy beats simply spending the
budget (Zoom Eye **−15.3pp** at 4K). That result motivates §2 rather than competing with it.

## §2 — ★★★ THE DEFECT: the read-out averages away its own signal

### §2.1 The general claim (two architectures)

**The deployed block-mean read-out is suboptimal on every model tested**, and averaging the whole
stack is catastrophic:

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| deployed block mean | 39.3% | 35.1% |
| **best repair** | **45.5%** | **43.5%** |
| **gain from fixing the read-out** | **+6.2pp** | **+8.4pp** |
| mean of ALL layers | 36.6% | **21.5%** |

On Qwen3-VL, averaging the k individually-best layers degrades monotonically past k≈3:
**45.5% → 39.3% (k=11) → 36.6% (k=28)**.

**Why averaging hurts: a large fraction of layers are anti-correlated with the target.** Mean
gt_pct by depth (0.500 = chance, lower is better):

| | early (0–7) | mid (8–17) | late (18–27) | layers worse than 0.45 |
|---|---|---|---|---|
| Qwen3-VL-2B | 0.456 | 0.413 | 0.409 | 9/28 |
| **Qwen2-VL-7B** | **0.620** | **0.515** | 0.392 | **15/28** |

Qwen2-VL's early and mid layers are *actively worse than chance*, which is exactly why averaging all
28 collapses it to 21.5% — the mean is dominated by layers pointing away from the target. **That
some layers are anti-correlated replicates on both models; *which* ones does not.**

### §2.2 The remedy is architecture-dependent — reported, not smoothed over

| model | what fixes it | why |
|---|---|---|
| Qwen2-VL-7B | **pick L21** — one integer, no learning | all 5 folds choose it independently |
| Qwen3-VL-2B | **learned signed combination** | no single layer is stable out-of-fold |

⚠ **In-sample "best layer" is a trap.** On Qwen3-VL it reads 41.9% and collapses to **36.1%**
out-of-fold — *below* the 39.3% block mean it appeared to beat — because it is the best of 28
candidates on 191 items. Every layer recommendation in this paper is fold-validated.

### §2.3 Where signed contrast IS the mechanism (Qwen3-VL-2B)

The learned weights take **both signs** — L17/L19/L24/L16 positive, **L27/L26/L15/L11 negative** —
and the final layer's gt_pct is **0.529 against 0.500 chance**, i.e. anti-correlated, yet added at
weight +1 by the deployed read-out. A mean can only add, so it cancels layers pointing at
distractors.

⚠ **Two parts of this replicate and one does not.** That anti-correlated layers exist, and that
averaging them in is what hurts, holds on both (§2.1). But on Qwen2-VL the *final* layer is fine
(0.416) — the bad layers are early/mid — and the learned combination reaches *exactly* the best
single layer (43.5%), so contrast recovers nothing there. Signed contrast is
therefore presented as **the mechanism on one model**, not as a property of VLM read-outs. What
generalises is §2.1.

## §3 — The method: Depth-Contrast Re-ranking

Glance (one pass, B₀=300, keep the per-layer attention already computed) → re-rank each cell with a
small learned head over its 28-layer profile, neighbourhood and position → answer from a W=0.15
window at the argmax. Step 2 costs microseconds on CPU and **no extra forward pass**.
Gate: on relational questions, skip and answer from the glance.

## §4 — ★★★ It converts

V\*Bench, n=191, Qwen3-VL-2B, tokens measured, proposals out-of-fold.

| | acc | passes | vs vanilla |
|---|---|---|---|
| vanilla uniform@300 | 56.5% | 1 | — |
| uniform@600 (spend it instead) | 63.9% | 2 | +7.4 |
| **DCR** | **68.6%** | 2 | **+12.0 [+4.2,+19.9]** |
| DCR gated | 72.8% | 1.61 | +16.2 [+10.5,+22.5] ⚠ |
| single-region: DCR − uniform@600 | | | **+13.0 [+2.6,+23.5]** |

Predicted **+7.2pp** before the run from §6D's coverage strata; observed **+8.4pp** over the
incumbent. **Internal control exact:** on the 79 items where both proposers chose the same cell,
**+0.0pp [+0.0,+0.0]**.

## §5 — It transfers zero-shot

HR-Bench 4k, n=800, **nothing refitted**, a benchmark that ships **no boxes** so the head could not
have been fitted there: **+4.9pp [+1.8,+8.1]** per-row, **+6.5pp** CircularEval (26.5→33.0%),
**+8.5pp [+4.0,+13.0]** on single-region. Internal control exact again (n=160).

⚠ The **allocator** still loses to the budget axis at 4K (−12.4pp). Component transfers; method does
not. Both are reported.

## §6 — ★★★ WHY: restored answer formation at L21

Logit lens, four arms, identical budget, only pixels differ. Final layer from `model.logits`, never
`norm(h[-1])` — that bug cost us a retraction (§12D) and both paths are now asserted to disagree.

**Separation is ≈0.0pp through L20, then +15.3pp at L21** and stays. The failing arms (uniform,
argmax) jump at **L2** — the answer prior — and never improve: uniform's max over all 28 layers is
**56.3% = its final answer**. The working arms jump at **L21**, and the step scales with evidence
supplied (+28.4 DCR, +44.2 oracle).

**The causal control** — same arm, split only by whether its window delivered:

| | head | uniform | oracle | Δ |
|---|---|---|---|---|
| **covers** | **90.1%** | 60.4% | 98.0% | **+29.7 [+19.8,+39.6]** |
| **misses** | 43.8% | 51.7% | 80.9% | **−7.9 [−18.0,+2.2]** |

> DCR does not reason better. It moves items across a threshold — out of the regime where the answer
> exists at no layer, into the regime where L21 has something to convert.

## §7 — Why nothing internal can do this

**Seven interventions, all null**: sink suppression, attention amplification, residual steering,
layer read-out, DoLa/DeCo, component ablation, and **contrastive decoding steered to the
ground-truth region** (ceiling +2.6pp [−0.5,+5.8], −5.5pp vs its own bar).

The seventh is decisive because it aimed *perfectly* and proved the target is causally live —
masking the evidence region shifts logits **3.32×** more than a random region and costs **10.5pp** —
**and still could not convert it.** And localisation itself is built by the LM: every vision-tower
arm is at or below the 2.3% chance rate (§14G).

> The model looks in the right place; the right place is load-bearing; it still cannot answer,
> because no operation on the residual stream can supply information the encoder never wrote.

## §8 — Boundary, limitations, negatives kept

- **Relational questions get worse** (−10.5pp ungated). One window cannot cover a multi-region
  evidence set — §3's coverage account, replicated on both benchmarks.
- ⚠ **The gate is not validated as transferring**: its keyword rule matches V\*Bench's own
  `category` on **99.0%** of items and is at chance concordance on HR-Bench. **Ungated +12.0pp is
  the defensible pooled claim.**
- **At 4K allocation still loses to budget.** **21.5pp** of headroom remains to oracle placement;
  DCR captures **27.6%** of the proposal ceiling.
- **The adaptive-budget family is closed at both levels** — with a perfect size oracle (§14A) and
  for every free pass-1 statistic (§14B).
- **W=0.25 beats the deployed W=0.15** (+3.1pp) but is selected in-sample; a learned sizer has
  **+11.0pp** of stable headroom with **no known predictor** (target size is flat across quartiles).
- **Retractions kept in the paper**: two mechanisms for a transfer failure that was not real; a
  "does not transfer" call from a category-ordered prefix; a capture fraction measured against a
  restricted rather than true ceiling (65% → **27.6%**); `peak` computed without the ring mask
  (0.576 → **0.745**); and the §12D double-normalisation bug.
- **One backbone** for end-task numbers. The read-out defect replicates on two architectures
  (§2.1); its *mechanism* does not (§2.3). LLaVA-OneVision is the open third point — its
  `image_newline` separators break the grid assumption, and phase 41's separator detection is the
  fix.

## §9 — Related work

On the resource axis: pruning holds LLM-side tokens fixed; LLMind/Gizdov hold pixels; ViRGo holds
seconds; crop/zoom hold nothing. **This work fixes the read-out they all share.** Ram's
accessibility / localization / causal-use hierarchy supplies the diagnostic vocabulary; §14G
qualifies its vision-side selector, which is at chance where targets are small and query-specific.
