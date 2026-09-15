# Phase 22 — Does attention-guided pruning drop small targets? (prediction refuted)

## 1. Research question
A published result says attention-guided token pruning is beaten by random selection and "fails
catastrophically on fine-grained localisation", without explaining why. We predicted our Phase 6c
supplied the mechanism: attention does not mark small targets, so a selector cannot keep them.

## 2. Finding and contribution (plain English)
**The opposite is true.** Attention-guided retention keeps small targets **+20 to +35pp above
chance**, and is *most* above chance exactly where we predicted it would be worst. Our explanation
for the published result does not hold, and we report it as a refutation.

The subtlety that misled us is worth its own line: Phase 6c measured attention **mass**, which is
sub-proportional (0.49–0.95× enrichment). Pruning selects on **rank**. A target's few tokens can rank
inside the top-k while carrying less than proportional mass, because mass is dominated by sinks
elsewhere. We conflated the two.

## 3. Numbers that changed
Retention minus keep-rate (chance = 0):

| stratum | r=0.10 | r=0.25 | r=0.50 |
|---|---|---|---|
| <0.5 tok | **+20.2** [+12.9,+27.9] | **+34.8** [+27.3,+42.5] | **+33.3** [+28.7,+37.9] |
| 2–8 tok | +10.6 | +23.9 | +26.6 |
| >32 tok | −1.1 | +3.1 | +6.0 |

## 4. Keep in paper: 6/10
Keep the **mass-vs-rank distinction** — it is a reusable methodological point and it forced a
retraction of Phase 6c's stronger reading. The refuted prediction itself is one paragraph.

## 5. Experiment, step by step
1. Pre-register the test: retention of the target's tokens **below** the keep-rate r, which is chance
   by construction for random/uniform selection.
2. Take layer-2 attention from the final prompt position on the Phase 21 stratified sample (n=600).
3. Keep the top-r fraction of image tokens; measure what fraction of the target's tokens survive.
4. Subtract r to get retention-above-chance.
5. Report per stratum, and separately measure attention **mass** enrichment to expose the
   mass-vs-rank confusion.
