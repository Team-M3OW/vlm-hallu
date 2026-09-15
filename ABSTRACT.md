# Abstract (draft, 2026-09-08)

Vision-language models allocate their visual token budget uniformly across an image, and they do so
*before* reading the question. We show that this ordering — not image resolution, decoding, or
language priors — explains a class of confident perceptual failures.

Studying objects that a VLM confidently denies while they are plainly present, we find the median
such object occupies **less than one merged visual token**. The object is not unencoded: a linear
probe recovers its category from its own tokens at **0.71 AUROC** against a same-image,
size-matched, other-category control. But the signal is too faint to use. Explicit coordinates in
the prompt recover **0.0%** of failures, a bounding box drawn on the image **5.6%**, and both
attention reweighting and activation steering are null against norm-matched random directions.
Information-free bicubic upsampling — which adds tokens but no evidence — recovers **50%**. You
cannot redirect attention to capacity that was never allocated.

Our main result isolates the cause. At **matched realized token budget**, allocating tokens by
query beats uniform allocation by **30.4 points** over a size-matched *random-placement* control on
V\*Bench (n=191), and by 24–35 points on POPE; query-placed allocation at **292 tokens outperforms
uniform allocation at 1176 tokens**. On items uniform allocation gets wrong, query placement
recovers **86–93%** while random placement recovers 16–19%. The effect has a domain: it is
significant below ~2 tokens on target and vanishes above ~4.

We bound the claim rather than generalize it. On CUB-200-2011 the failure is opposite in polarity —
70.7% false-accept on same-genus species with 53.8 tokens on object — and part-targeted allocation
does not help. We also report a methodological hazard: cohorts selected on model confidence, the
standard construction in the hallucination interpretability literature, are **7.2× enriched for
benchmark label errors**, leaving 58% of the canonical cohort mislabeled or ambiguous.

*Limitations: results are for a single model family (Qwen3-VL-2B) and use oracle target regions
from benchmark annotations; a learned query-conditional proposer is untested.*
