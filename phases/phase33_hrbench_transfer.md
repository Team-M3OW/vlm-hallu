# Phase 33 (33/33b) — Does the gate transfer to 4K images?

## 1. Research question
The V\*Bench gate's main threat is that its relational-keyword rule agrees with V\*Bench's own
category annotation on 99% of items. HR-Bench 4k breaks that (53.5% concordance). Does the gate
transfer?

## 2. Finding and contribution (plain English)
Partly, and the failure is informative. The confidence route transfers (+3.6pp per-row, +8.0pp on
CircularEval); the **text rule fails** (−7.5pp) and drags the combined gate down.

The control that resolved the ambiguity is Phase 33b: a **random-centre window of the same size**.
attn − rand = **+13.6pp**, so the localiser genuinely transfers to 4K — allocation, not localisation,
is what fails there.

This phase also contains the project's worst silent bug: 26.5% of rows were answered against the
**wrong image**, because 159 unique question strings across 200 instances meant grouping on
(question, category) merged different images. The tell was "5.7 cycles per instance" when the design
has 4.

## 3. Numbers that changed
- uniform@300 52.5% · uniform@600 **59.2%** · conf_route 56.1% · TEXT+CONF 53.4% ·
  always attn@0.15 43.8%.
- **attn − rand = +13.6pp** [+10.0, +17.2].
- Keyword-gate concordance with `category`: V\*Bench **99.0%**, HR-Bench **53.5%**.

## 4. Keep in paper: 8/10
Keep the transfer result, the rand control, and the wrong-image bug (as a methods warning: always
key instances on image **content hash**, never on text fields).

## 5. Experiment, step by step
1. Key instances on `(question, category, md5(image bytes))` and **assert** exactly 4 rows per
   instance. This is the fix for the silent wrong-image bug.
2. Refit nothing: W=0.15, the layer block, the ring mask and B₀ all come from V\*Bench.
3. Run uniform@300, uniform@600 (the compute-matched bar), attn@0.15 and attn@0.25.
4. Report both per-row accuracy and **CircularEval** (an instance counts only if all 4 option
   permutations are correct).
5. Add the decisive control (33b): a random centre drawn **once per instance**, so the 4 cycle rows
   share a crop exactly as in the attention arm.
