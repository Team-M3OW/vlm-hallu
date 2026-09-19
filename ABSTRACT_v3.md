# Abstract v3 (2026-09-19) — SPACE-HOP style. Every number holds on >=2 checkpoints.

**TRANSIT: TRANSport-Informed Token allocation for Vision-Language Models**

---

Vision-language models deployed on high-resolution imagery must decide where to spend a fixed visual-token budget,
a decision that determines whether fine-grained evidence survives encoding at all. The prevailing remedy is
attention-guided cropping: read the model's own attention map, crop where it peaks, and re-encode the cropped region
at higher density.

Despite significant progress, existing methods exhibit two major limitations. First, every such method must choose a
depth at which to read the attention map, and published methods hand-pick a single layer or average a fixed block;
we show this choice dominates every other design decision, with single-layer read-out collapsing to the four-option
chance rate (24.6% and 33.9% on two checkpoints, against 62.3% and 57.4% for not cropping at all). Second, evaluation
is neither compute-matched nor stratified by question type, which conceals that the entire family is worth only 0-3
points at equal token budget, and that cropping actively degrades questions requiring two objects.

To address these challenges, we propose TRANSIT, which derives both the read-out depth and the token budget from a
causal measurement of where image information enters the text stream rather than from tuning. Layer-wise masking of
text-to-image attention shows that image-to-text transport completes by layer 16 of 28 -- blocking every layer from
16 onward flips no answers, while blocking layers 0-16 reproduces the full effect -- yet every method in the family,
including ours, reads its map at layers 17-22, after the transport window has closed. The attention these methods
consume is a residue of where the model looked, not the channel that carries the evidence.

Two mechanisms follow directly. First, a single closed-form ridge solve over the 28-layer attention profile of each
image cell yields a signed depth filter that subtracts the late layers the conventional block-mean read-out adds,
improving crop placement by 7.9 and 8.9 points over the strongest published rule at matched compute, and by 8.9 and
12.0 points over the equal-compute baseline. Second, because visual tokens are causally inert past the transport
boundary, 90% of them can be discarded there at a cost of 0.0-0.5 points in 9 of 10 measured cells -- with any ranking
criterion, including random -- and the freed compute reallocated to encoding resolution, where it recovers exactly the
model's measurable resolution headroom (r = 0.966, slope 1.01, mean error 0.7 points across 4 checkpoints and 3
benchmarks).

We demonstrate that the two mechanisms are complementary but not interchangeable. Crop placement wins single-instance
perception in 7 of 7 benchmark-checkpoint cells while losing cross-instance perception in 6 of 7, because magnifying
one object discards the other; resolution reallocation wins cross-instance perception on 4 of 4 checkpoints and, on
one, exceeds even a perfectly placed oracle crop. The resulting selection rule is measurable offline and requires no
question classifier. We release all negative results, including eleven designs that failed to make cropping work on
cross-instance questions.
