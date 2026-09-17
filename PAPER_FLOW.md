# Paper flow (v13) — findings-led

> **What changed from v12.** v12 was method-led and titled around reading attention at the wrong
> depth. Three things forced a rewrite. (1) That framing is a predecessor's — TwigVLM, VEA, Lu et al.
> and CLAA all establish that early-layer attention is unreliable; we cite it rather than claim it.
> (2) The strongest result is no longer the method: **87% / 84% of errors are fixed by spending the
> same budget on the evidence region**, and §15A measures the same boundary *inside the
> representation with no intervention at all*. (3) The method is narrower than v12 assumed — one
> benchmark, one stratum, and its lead over a properly-built baseline is **5.3pp, not 17pp**.
> Backup: `PAPER_FLOW_v12.md.bak`.

---

## The claim

> Vision-language models fail on small objects because the evidence is **never encoded**, not because
> attention is misdirected. On V\*Bench, **87% / 84%** of two models' errors are fixed by re-spending
> the *same* token budget on the evidence region. Below ~0.25 merged tokens of target extent the
> visual tokens covering the target are **at chance** for predicting the model's own correctness,
> while above it they reach 0.656 — a threshold visible in the representation itself. A family of
> published methods that redirects emphasis cannot cross this boundary; only adding resolution can,
> and we give a method that does, with its two hard limits measured rather than conceded.

---

## §1 — The finding (leads)

| | Qwen3-VL-2B | Qwen2-VL-7B |
|---|---|---|
| pass-1 wrong | 43.5% | 49.2% |
| **of those, fixed by the same budget on the evidence region** | **86.7%** (94.0% at W=0.15) | **84.0%** (89.4%) |

The oracle-crop control is what makes this a claim about *encoding* rather than difficulty: it is
**flat across the cliff** at 95.8% / 100.0% on both models. Items below the threshold are not harder;
they are invisible. Median confident-denial object: **0.4 merged tokens** (POPE cohort).

**Figure 1** is the accuracy-vs-tokens-on-target curve with the oracle arm flat across it.

## §2 — The same boundary, without any intervention  ★ the new evidence

Probing hidden states at five positions × 28 layers, out-of-fold (§15A):

| best AUROC for "will this answer be wrong" | below the cliff | above |
|---|---|---|
| final prompt position | 0.725 | 0.693 |
| **visual tokens covering the target** | **0.513 — chance** | **0.656** |

The evidence tokens predict correctness only above the threshold. The final position is unaffected,
because it encodes the model's own uncertainty whether or not it saw anything. **Two independent
routes to one threshold — one causal, one representational.**

Also here: Orgad et al.'s (ICLR'25) headline — probe the *exact answer tokens* — **does not port**
(evidence tokens are 0.127 / 0.085 *worse* than the final position), with the size-matched
random-region control passing so the comparison is interpretable. And a visual token's own content
peaks at **L1** and decays to chance by L27.
⚠ *Qwen2-VL replication queued; this section is single-model until it lands.*

## §3 — Why nothing downstream fixes it

Seven internal interventions, each well-controlled: steering · sink suppression · residual injection ·
DoLa/DeCo · component ablation · attention amplification · contrastive decoding aimed with the GT box.

The two that close it:
- **Aim perfectly and you get 16%.** Oracle-targeted amplification +5.8pp against the oracle crop's
  +36.2pp at the same budget. The oracle amplification set is **one cell of 294** — there is nothing
  to amplify.
- **The region is causally live and still unusable.** Masking it moves the output **3.32×** more than
  a random region and costs 10.5pp; contrastive decoding steered there has a **ceiling of +2.6pp**.

**This bounds a family.** VEA highlights evidence pixels; Lu et al. rescale attention heads; KLAL
supervises attention in training. All report gains, none measures target size, and none runs an
oracle control. §1–§3 say where that family's ceiling is. KLAL is the honest exception and is cited
as such: it works, and it works by *changing the weights* — its own Table 5 shows target-token
embedding norms rising 6–19%.

## §4 — Mechanism

- **Answer formation at L21**, 2 models. Flat for twenty layers, then +15.3pp. The failing arms jump
  at L2 — a prior forming — and their 28-layer maximum **equals** their final answer.
- **The causal split**: the same method, divided by whether its window contained the evidence —
  **+29.7 / −7.9** and **+23.8 / +0.0**.
- **Coverage decides the sign**: dose–response crossing zero at 25% coverage; half of all windows miss.
- **Attention is question-blind until ~50% of depth**; and the band it switches on in is the same one
  CoRe, Lu et al. and VEA independently identify — four papers, two modalities.
- **Localisation is built by the LM**: the vision tower is at or below chance (1.6 / 1.0 / 0.5% vs 2.3%).

## §5 — Method, and its two hard limits

Rank cells by the 28-layer attention profile, crop once at W=0.25, one extra pass.

| vs equal compute, single-object questions | |
|---|---|
| Qwen3-VL-2B | **+15.7pp [+6.1,+25.2]** ✔ |
| Qwen2-VL-7B | **+11.3pp [+1.7,+20.9]** ✔ |

The **only** arm that clears the bar, on both models. Pooled does not clear (+7.9 / +6.8) and both
numbers are reported.

**Two boundaries, each with a mechanism, both found by our own controls:**
1. **Relational questions** — the evidence set is a union **7.9× larger in area**; one window cannot
   cover it. −3.9 / +0.0pp.
2. **Existence questions — invalid, not merely unhelpful.** On POPE the crop arm is **−57 to −77pp**,
   because a 6%-area window is genuine evidence of *absence* for "is there an X in the image". And
   POPE cannot even supply a ceiling: knowing the GT box *is* knowing the object is present, so the
   oracle arm is **circular** there.

## §6 — Read-outs: what we measure against  ★ must precede §5's numbers in the table

The deployed read-out averages indiscriminately over **heads and layers**, and both averages lose
signal. Fixing each is one line:

| | gain over the deployed argmax | models |
|---|---|---|
| **max across layers, not mean** (ported from CLAA) | **+7.3 / +6.8pp** | 2 |
| **norm-weighted attention α·‖v‖** (ported from Kobayashi) | +4.7pp | 1 |
| both | **+10.5pp** | 1 |
| **top 10% of heads by visual-attention ratio** (ported from Lu et al.) | **+11.5pp**, label-free | 1 |

**What does not transfer, and why that is a finding:** attention rollout **−33.5pp** and
gradient×attention **−30.4pp**. *Fixes that target the sink transfer; fixes that model information
mixing destroy the signal* — consistent with §2's finding that a token's own content peaks at L1.

**CoRe's contrastive head criterion does not replicate**: +3.1pp [−1.0,+7.3] over the absolute one,
and it needs boxes while the absolute one needs none.

> **The baseline in every table is the strong one.** Against the deployed argmax the learned head
> leads by 17pp; against a max-aggregated, head-selected read-out it leads by **5.3pp**. Reporting the
> deployed baseline alone would be the reviewer's first objection.

## §7 — Negatives and protocol

The window sizer (pre-registered, failed on 2 models × 2 grids) · three gates · multi-crop
(−3.1 vs a predicted +5.5) · the pre-generation detector (1 of 2, ordering reversed) · the head
architecture search (MLP, listwise, cross-attention over layers — nothing beat the tree on both) ·
adding the §6 fixes as head features (nothing clears).

And the protocol that produced them: pre-registered decision rules applied as written; oracle controls;
size-matched random-region controls; out-of-fold selection grouped by item; a measured **~1pp noise
floor**; and four retractions found by our own checks (double-normalisation, ROI-selection artefact,
wrong-image join, capture-fraction mispairing).

---

## Scope, stated in the paper

**Method:** questions presupposing their target exists · small targets · V\*Bench · the Qwen family
end-task (LLaVA tested on proposals only, 3 of 4).
**Findings §1–§4:** two models throughout, except §2 and the intervention nulls (single-model,
replication queued).

## Open before submission

1. **§2 on Qwen2-VL** — queued. It is the paper's best opening and is single-model today.
2. **A prior-art survey of the cliff itself.** Never done, and it is now load-bearing for the whole
   paper. Before any writing, not after.
3. Second model for the phase 51 / 77 intervention numbers.
