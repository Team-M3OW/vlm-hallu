# Phase 70 / 70b — A learned re-ranking head on the attention map

**Classification: METHOD** (first positive internal, non-cropping result)

## 1. Research question

The proposer already knows where the evidence is and then throws the knowledge away. §11A measured
it: the ground-truth cell sits in the **top 3.4%** of cells, yet taking the **argmax** covers the
evidence only **39.3%** of the time — while on **88.5%** of items *some* cell in that map would
have covered it.

So: can a small learned read-out over the model's own attention tensor recover that lost ceiling —
with no extra forward pass, no crop, and no pixel preprocessing?

## 2. Finding, in simple English

Yes, and by a lot. A small head that looks at the attention map raises top-1 coverage from **39.3%
to 52.9%**, which closes **27.6% of the real headroom** (the true ceiling is 88.5%). Attention features alone get +11.5 of
the +13.6pp, so it is not reading position.

The controls matter more than the headline. V\*Bench targets are not scattered uniformly, so a head
that learned nothing but "look near the middle" could fake this. It doesn't: a head given **only**
position scores **3.7%**, and a head trained on **shuffled** labels scores **2.6%** — both at the
rate a randomly chosen cell achieves, given how small these targets are.

**Then the ablation contradicted us.** We expected the head to be fixing the serialization sink —
§5's finding that attention piles up 2.0–4.5× on the trailing row boundary across four
architectures. It isn't. Handing the head explicit last-column and last-row indicators is worth
**exactly zero**, and deleting them costs exactly zero. What carries the result is something else:
the **28-layer depth profile**, worth **+7.3pp** by itself. The deployed read-out averages layers
16–26 into one map and then takes a max, and that average is what destroys the signal — different
layers disagree about which cell matters, and the disagreement is the information.

That is a correction to our own paper. The sink is real as a *description* of where attention mass
goes; it is **not** the proposer's defect. §10A already found that suppressing the sink at inference
buys nothing, and this now says the same thing from the other direction. The §5→§6 link in
PAPER_FLOW asserted a connection nobody had tested.

One trivial explanation had to be excluded: maybe the head is just smoothing a noisy map. It is not
— blurring destroys the signal rather than improving it (39.3% → 17.8% at 3×3, 0.5% at 7×7). The
neighbourhood term is learned local *contrast*.

## 3. Numbers that changed

| arm | covers evidence | vs incumbent |
|---|---|---|
| deployed argmax (incumbent) | 39.3% | — |
| **learned head (attention + geometry)** | **52.9%** | **+13.6pp** |
| attention only, no geometry | 50.8% | +11.5 |
| *geometry only (centre-prior CONTROL)* | **3.7%** | −35.6 |
| *shuffled labels (CONTROL)* | **2.6%** | −36.6 |
| oracle among top-5 / 10 / 20 (restricted refs) | 60.2 / 64.4 / 72.8% | |
| **TRUE ceiling (any cell covers)** | **88.5%** | |

Ablation (70b):

| features | covers | Δ |
|---|---|---|
| deployed map only (value + 3×3 + position) | 45.0% | +5.8 |
| + sink indicators | 44.5% | +5.2 |
| **+ depth profile (28 layers)** | **51.8%** | **+12.6** |
| + within-layer ranks [full head] | 52.9% | +13.6 |
| full head **minus** sink indicators | **52.9%** | **+13.6** |

Training-free blur baselines: 3×3 → **17.8%**, 5×5 → 2.1%, 7×7 → 0.5%.

## 4. Keep in paper: **9/10**

The only positive method result that satisfies every constraint at once — internal, plug-and-play,
no extra pass, no crop, no preprocessing — and it arrives with its controls already run and its
mechanism already ablated. It docks a point because the **end-task gain is unmeasured**: this
improves proposal quality, and a better proposal only converts to accuracy if something downstream
acts on it.

## 5. Experiment, stepwise

1. **No GPU.** Every attention map is read from `phase30c_attn_maps_all.jsonl` — 191 items × 28
   layers × ~295 cells, already on disk.
2. **Per-cell label.** Coverage = the fraction of the GT box falling inside a W=0.15 window centred
   on that cell; "covers" means ≥ 0.5. Identical geometry to the deployed allocator, including the
   same edge-clamping, so the label matches what the method would actually crop.
3. **Per-cell features, all from the pass that already happened.** The 28-layer normalised attention
   profile; the cell's within-layer rank at each layer (ordering as well as magnitude); the deployed
   block-16-26 value and its 3×3 neighbourhood mean; and geometry — row/col fraction, distance to
   centre, distance to nearest border, plus explicit **last-column / last-row** indicators so the
   head *can* learn the sink if the sink is what matters.
4. **Out-of-fold with GROUPED folds.** GroupKFold(5) × 3 seeds, grouping by item, so an item's 295
   cells are never split between train and test. Training rows are negative-subsampled (every
   covering cell plus 30 random non-covering cells per item); **evaluation is always on all cells.**
5. **Score like the deployed proposer.** Apply the same outer-ring mask, take the argmax of the
   head's score, and record whether that cell covers. Same function, different ranking.
6. **Run the controls before believing anything.** Geometry-only (catches a centre prior),
   shuffled-labels (catches a leaking harness), and oracle-among-top-k (the ceiling of any function
   of this map).
7. **Ablate by feature group (70b)** against two predictions fixed in advance — P1 "the block mean
   destroys the depth profile", P2 "the sink corrupts the max". P1 confirmed at +7.3pp; **P2
   refuted at +0.0pp.**
8. **Exclude the trivial account.** Re-run top-1 selection on a blurred version of the same map with
   no learning at all; if smoothing explained the gain, it would show here. It collapses instead.

## Next step this opens

The end-task measurement: run the allocator with the head's proposal instead of the argmax and score
accuracy at matched realized tokens. §6D's coverage→accuracy curve predicts the 39.3%→52.9% shift
should pay, but that is a prediction and it needs the GPU run to become a result.
