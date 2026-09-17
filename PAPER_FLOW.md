# Paper flow (v14) — method-led, findings as the explanation

> **What changed from v13.** v13 demoted the method to §5 on the grounds that it is "one benchmark,
> one stratum". Two things were wrong with that. (1) **A double standard**: v13 demanded
> multi-benchmark, multi-family, pooled significance from the method while accepting single-model
> evidence in its own findings sections. (2) **The venue objection was stated in its weak form.** We
> did not merely evaluate on one benchmark — we *measured why the alternatives cannot test this*:
> POPE's oracle arm is circular, MMBench has no budget axis, HR-Bench's budget axis is unusually
> productive at 4K. "The only valid venue, with the others shown invalid" is a different claim from
> "we used one dataset". Backups: `PAPER_FLOW_v13.md.bak`, `PAPER_FLOW_v12.md.bak`.

---

## The claim

> Vision-language models fail on small objects because the evidence is **never encoded**. We give a
> method that fixes it by re-spending the same token budget — the **only** arm in our study that beats
> spending that budget on a larger image, on two architectures — and then show *why* it works, *where*
> it stops, and why the published family of attention-reweighting methods cannot do the same.

---

## §1 — The method

Rank cells by their **28-layer attention profile** (65 features, out-of-fold, grouped by item), crop
once at **W=0.25**, one extra forward pass. No fine-tuning, no image editing, no extra model.

**Why the deployed read-out leaves this on the table:** its argmax covers the evidence 40.5% of the
time, oracle selection among *the same map's* top-5 covers 61.5%, and the target cell already sits in
the **top 3.4%**. The ranking contains the answer; the max throws it away to a sink.

| vs **equal compute**, single-object questions | |
|---|---|
| Qwen3-VL-2B | **+15.7pp [+6.1,+25.2]** ✔ |
| Qwen2-VL-7B | **+11.3pp [+1.7,+20.9]** ✔ |

vs the unmodified model **+15.0 / +14.1pp**; vs random placement **+28.3 / +26.7pp**; zero-shot to
HR-Bench with nothing refitted, **+4.9pp** coverage and **+6.5pp** CircularEval.

⚠ **Pooled over both question types it does not clear** (+7.9 / +6.8). Both numbers in the table.

## §2 — What it is measured against  ★ before any headline number

The deployed read-out averages indiscriminately over **heads and layers**. Both averages lose signal,
and fixing each is one line — so the baseline in every table is the fixed one, not the deployed one:

| | over deployed argmax | models |
|---|---|---|
| max across layers, not mean (ported from **CLAA**) | **+7.3 / +6.8pp** | 2 |
| norm-weighted α·‖v‖ (ported from **Kobayashi**) | +4.7pp | 1 |
| both | **+10.5pp** | 1 |
| top 10% of heads by visual-attention ratio (ported from **Lu et al.**) | **+11.5pp**, label-free | 1 |

| locator | coverage |
|---|---|
| deployed block-mean argmax | 43.5% |
| best training-free (heads + max) | 55.0% |
| norm-weighted × max | 56.5% |
| CoRe-OOF heads + max *(needs boxes)* | 58.1% |
| **our head** | **63.4%** |

> Against the deployed baseline the head leads by 17pp. **Against a properly-built one it leads by
> 5.3pp**, and it is still the only arm that converts to a win at equal compute (§4).

**What does not transfer, and why that is a result:** rollout **−33.5pp**, gradient×attention
**−30.4pp**. *Sink-targeting fixes transfer; mixing-targeting fixes destroy the signal.*
**CoRe's contrastive criterion does not replicate**: +3.1pp [−1.0,+7.3], and it needs boxes.
**Given all of these as features the head gains nothing** (+1.6 [−1.0,+4.2]) — it already has them.

## §3 — Why it works: the evidence is not encoded

| | Qwen3-VL | Qwen2-VL |
|---|---|---|
| pass-1 wrong | 43.5% | 49.2% |
| **fixed by the same budget on the evidence region** | **86.7%** | **84.0%** |

The oracle arm is **flat across the cliff** (95.8 / 100.0%), so items below it are not harder — they
are invisible. Median confident-denial object: **0.4 merged tokens**.

⚠ **An intervention-free version of this was attempted and REJECTED.** Probing the visual tokens that
cover the target gave 0.513 (chance) below the cliff and 0.656 above on Qwen3-VL — but 0.623 vs 0.633,
no gap, on Qwen2-VL (§15A). **The cliff rests on the intervention evidence alone.** What survives from
that experiment is a 2-of-2 negative: Orgad et al.'s "probe the exact answer tokens" does **not** port
to vision — evidence tokens are worse than the final position on both models.

**Seven internal interventions fail**, closed by the two that matter: aim attention perfectly and you
recover **16%** of what the crop recovers, because the oracle amplification set is **one cell of 294**;
and the region is causally live (**3.32×**, −10.5pp masked) with a contrastive-decoding ceiling of
**+2.6pp**.

> **This bounds a published family.** VEA highlights evidence pixels, Lu et al. rescale heads, KLAL
> supervises attention in training. All report gains; none measures target size; none runs an oracle
> control. KLAL is the honest exception and works by *changing the weights* — its own Table 5 shows
> target-token norms rising 6–19%, which is our claim from the training side.

## §4 — Where it stops: two boundaries, each with a mechanism

1. **Relational questions.** The evidence set is a **union 7.9× larger in area**; one window cannot
   cover it. −3.9 / +0.0pp. Predicted by the coverage account *before* the runs.
2. **Existence questions — invalid, not merely unhelpful.** POPE: **−57 to −77pp**, because a 6%-area
   window is genuine evidence of *absence* for "is there an X in the image". And POPE cannot supply a
   ceiling: knowing the GT box **is** knowing the object is present, so its oracle arm is **circular**.
3. **LLaVA-OneVision.** The one model whose best single layer equals its block mean — no layer
   disagreement for any re-ranker to exploit. The head is 3 of 4 on proposals at W=0.15.

**These are why V\*Bench is the venue**: not the only one tried, the only one that can test this.

## §5 — Mechanism

Answer formation at **L21** (2 models), flat for twenty layers then +15.3pp; the failing arms peak at
L2 and their 28-layer maximum **equals** their final answer. The causal split by whether the window
delivered: **+29.7 / −7.9** and **+23.8 / +0.0**. Coverage decides the sign, crossing zero at 25%.
Attention is question-blind until ~50% of depth — the same band **CoRe, Lu et al. and VEA**
independently identify, four papers across two modalities. Localisation is built by the LM: the vision
tower is at or below chance.

## §6 — Negatives and protocol

The window sizer (pre-registered, dead on 2 models × 2 grids) · three gates · multi-crop (−3.1 against
a predicted +5.5) · the pre-generation detector (1 of 2, ordering reversed) · the head architecture
search (MLP, listwise, cross-attention over layers — nothing beat the tree on both models).

Protocol: pre-registered decision rules applied as written · oracle and size-matched random controls ·
out-of-fold selection grouped by item · a measured **~1pp noise floor** · four retractions found by our
own checks.

---

## Scope, stated plainly

**Method:** questions presupposing their target exists · small targets · V\*Bench (the only valid
venue, §4) · Qwen family end-task, LLaVA on proposals only (3 of 4).
**§3/§5:** two models, except §15A and five of the seven nulls (single-model, replication queued).

## Open before submission

1. **§15A on Qwen2-VL** — queued; it is §3's second pillar and single-model today.
2. **Prior-art survey of the cliff.** Never done and now load-bearing. Before writing, not after.
3. Second model for the phase 51 / 77 intervention numbers.
