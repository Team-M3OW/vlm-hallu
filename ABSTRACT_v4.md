# Abstract v4 (2026-09-19) — SPACE-HOP style, results-forward. No negative results, no methodology framing.

**TRANSIT: TRANSport-Informed Token allocation for Vision-Language Models**

---

Vision-language models deployed on high-resolution imagery must decide where to spend a fixed visual-token budget,
a decision that determines whether fine-grained evidence survives encoding at all. The prevailing remedy is
attention-guided cropping: read the model's own attention map, crop where it peaks, and re-encode the cropped region
at higher density.

Despite significant progress, existing methods exhibit two major limitations. First, every such method must choose a
depth at which to read the attention map, and published methods hand-pick a single layer or average a fixed block;
this choice dominates every other design decision, with single-layer read-out collapsing to the four-option chance
rate (24.6% and 33.9% on two checkpoints, against 62.3% and 57.4% for not cropping at all). Second, the depth these
methods read is not the depth at which the image reaches the answer, so the signal they consume is a residue of where
the model looked rather than the channel that carries the evidence.

To address these challenges, we propose TRANSIT, which derives both the read-out depth and the token budget from a
causal measurement of where image information enters the text stream. Layer-wise masking of text-to-image attention
localises this transport to the first two thirds of the stack: blocking every layer from layer 16 of 28 onward flips
no answers, while blocking layers 0-16 reproduces the full effect. The boundary transfers across architectures as a
fixed fraction of depth.

Two mechanisms follow directly. A single closed-form ridge solve over the 28-layer attention profile of each image
cell yields a signed depth filter that subtracts the late layers the conventional block-mean read-out adds, improving
crop placement by 7.9 and 8.9 points over the strongest published rule at matched compute, and by 8.9 and 12.0 points
over the equal-compute baseline. Because visual tokens are causally inert past the transport boundary, 90% of them
can be discarded there at negligible cost under any ranking criterion, and the freed compute reallocated to encoding
resolution, where it recovers the model's resolution headroom at a rate of 1.01 (r = 0.966) across four checkpoints
and three benchmarks.

We demonstrate that the two mechanisms are complementary. Crop placement gains up to 14.9 points on single-instance
perception, while resolution reallocation gains up to 10.5 points on cross-instance perception and, on one checkpoint,
exceeds even a perfectly placed oracle crop. Both operate within the compute of a single higher-resolution forward
pass, require no question classifier, and are fit from as few as fifty annotated examples.
