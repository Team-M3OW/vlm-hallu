# Phase 72 — Does the re-ranker transfer to a benchmark it has never seen?

**Classification: MIXED — positive for the component, negative for the allocator**

## 1. Research question

The head is trained on V\*Bench's own ground-truth boxes. The obvious objection is that it learned
V\*Bench. HR-Bench 4k removes that objection completely: different images, 4032×4032 instead of
~1500px, different question style, and **no bounding boxes at all**, so the head could not have been
fitted there even in principle.

## 2. Finding, in simple English

**Two results that point in opposite directions, and the paper has to carry both.**

The re-ranking itself transfers. With nothing refitted — not the window, not the layer block, not
the mask — it improves proposals on this new benchmark by **+4.9pp**, and by **+6.5pp** on
HR-Bench's own strict metric where an item counts only if all four shuffled versions of the options
are answered correctly. On single-object questions it is **+8.5pp**. That is real generalisation
across benchmark, resolution and question distribution.

But the *allocator* still loses. Cropping anywhere — well or badly — scores far below simply
encoding the image at twice the budget (−12.4pp). At 4032px, spending tokens beats placing them, and
a better proposal narrows that gap without closing it. That was already this project's finding for
4K images, and a good re-ranker does not overturn it.

**Two things I got wrong during this run, both recorded.** I called it "does not transfer" at 244
rows, from a prefix that happened to be 65% relational — the exact stratum where the head does
nothing. HR-Bench is ordered by category, so a prefix is a biased sample; the balanced 800-row
result flipped the sign. Then, explaining a failure that turned out not to exist, I proposed two
mechanisms and our own data refuted both: "the map is flat at 4K" (it is not — peak concentration is
0.0228 vs V\*Bench's 0.0269, and both are ~7× a uniform map) and "below the cliff there is nothing
to re-rank" (the gain is *largest* below the cliff).

## 3. Numbers that changed

| | per-row | CircularEval |
|---|---|---|
| uniform@300 | 52.8% | 37.5% |
| uniform@600 (bar) | 59.9% | 46.0% |
| argmax@0.15 | 42.6% | 26.5% |
| **head@0.15** | **47.5%** | **33.0%** |
| rand@0.15 | 38.9% | 20.0% |

head − argmax **+4.9pp [+1.8,+8.1]** ✔ · head − rand **+8.6pp [+4.6,+12.8]** ✔ ·
head − bar **−12.4pp [−16.2,−8.4]** ✗
`single` head − argmax **+8.5pp [+4.0,+13.0]** ✔ · `cross` +1.3pp [−3.2,+5.8] ✗
Internal control: identical proposal **+0.0pp [+0.0,+0.0]** (n=160).

## 4. Keep in paper: **9/10**

It is the evidence that the mechanism is not a V\*Bench artifact, and it is also the honest
statement of where the method stops. A paper with only the first half would be overclaiming; with
only the second, underclaiming.

## 5. Experiment, stepwise

1. **Train on all of V\*Bench and freeze (72a).** No folds, because the test set is a different
   benchmark — there is no item overlap to leak through. Save the model plus a spec recording the
   training grid range.
2. **Check feature portability rather than assuming it.** Every feature is grid-agnostic; the spec
   records 286–315 cells per item in training, and HR-Bench came in at 289–312 — inside the range,
   so the head is interpolating, not extrapolating.
3. **Verify the head is not collapsing before trusting any accuracy.** On the first 20 instances it
   produced 19 distinct cells with spread matching the incumbent's — so it is not defaulting to the
   centre, which is the main way a cross-benchmark transfer produces a meaningless number.
4. **Key instances by image hash.** HR-Bench has 159 unique question strings across 200 instances;
   keying on (question, category) alone once paired 212/800 rows with the wrong image, silently.
5. **Five arms, no oracle.** HR-Bench ships no boxes, so no coverage and no oracle-capture fraction
   is computable, and none is quoted.
6. **Report CircularEval as well as per-row.** It is the benchmark's own metric and strictly harder.
7. **Do not read the run early.** Fixed after the fact: the analyzer now refuses to present pooled
   numbers when only one category is present.
