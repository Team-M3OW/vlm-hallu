# Phase 76 — Can the vision encoder localise on its own?

**Classification: NEGATIVE (closes an architecture direction) + FINDING**

## 1. Research question

The method needs two forward passes: look, then crop and look again. That second pass is the cost
that has sunk every allocation method we measured. The vision tower runs *before* the language
model — so if its attention already knew where to look, resolution could be allocated at encode time
and the whole thing would cost one language-model pass instead of two.

## 2. Finding, in simple English

**It knows nothing. Not weakly — nothing at all.**

Every way of reading the vision tower's 24 layers lands at or *below* the rate you get by picking a
random cell. Best single vision layer: 1.6%. A learned combination across all 24: 1.0%. Averaging
them all: 0.5%. Random guessing: 2.3%. On the very same images, with the identical definition of
success, reading the *language model's* attention gets 39.3%, and our head gets 52.9%.

So the "where to look" signal is not sitting in the image waiting to be read. **The language model
builds it from the question.** Which, stated plainly, is obvious in hindsight: the vision encoder
never sees the question, so it has no way to know which of several plausible objects matters.

This kills the cheap version of foveation. Any encoder that allocates resolution unevenly needs a
question-conditioned signal, and that needs a language-model pass. A *cheap glance* — a low-resolution
first pass — is still viable and much cheaper than a full one, but it is not single-pass and the
cost model has to say so.

**It also sharpens the mechanism from phase 73.** The vision tower's learned weights come out with
the same shape as the language model's — twelve positive, twelve negative, final layer negative —
and buy nothing at all. So the story is not "signed combinations are magic." **Depth contrast
amplifies a signal that is present; it does not create one.**

**And it qualifies a result in ram's paper.** His Vision-CLS selector *is* vision-tower attention,
and he reports it falling inside the bird's box 80.8% of the time on CUB. CUB birds are large and
centred, so any salience finds them. On V\*Bench, where the target is small and the question picks
among candidates, the identical kind of selector is at chance. Both are right; the scope differs.

## 3. Numbers that changed

| proposer | top-1 coverage |
|---|---|
| random cell (chance, 200 draws/item) | **2.3%** |
| best single vision layer (L17) | 1.6% |
| learned signed combination, 24 vision layers, OOF | 1.0% |
| mean of all 24 vision layers | 0.5% |
| language-model read-out (argmax) | **39.3%** |
| learned language-model head | **52.9%** |

Vision weights: 12 positive / 12 negative, final layer −0.382 — same shape, no gain.

## 4. Keep in paper: **8/10**

A negative that closes an expensive direction in twenty minutes, plus a clean positive statement
about where localisation comes from. The "constructed by the LM, not read off the image" sentence is
worth more than the failed gate.

## 5. Experiment, stepwise

1. **Find where the attention actually lives.** Qwen3-VL's vision block computes
   `hidden_states + self.attn(...)` and returns a single tensor, so `output_attentions` never
   reaches it and a forward hook captures nothing. The weights exist only inside the module-level
   attention function.
2. **Patch that function and reduce inside the patch.** Raw vision weights are ~1200×1200×16 per
   layer, about 2 GB across 24 layers — they cannot be stored, so collapse to per-patch attention
   received (mean over heads, then queries) before leaving the patch.
3. **Pool to the grid the language model sees.** The vision tower works on the pre-merge patch grid;
   average each merge×merge block so the map is directly comparable to the LM read-out.
4. **Reuse the phase 73 analysis unchanged**, so the vision and language results cannot drift apart.
5. **Estimate chance with many draws per item.** One draw per item gave 0.0%, which made a
   multiplicative gate threshold vacuous and printed "PARTIAL" on data that was a clean failure.
   200 draws gives 2.3%, matching the random control measured elsewhere.
6. **Fix the verdict rule before reading it,** so the gate cannot be argued into passing after
   seeing the numbers.
