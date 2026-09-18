# Abstract (draft, 2026-09-19) — built only from claims that replicate on >= 2 checkpoints

**Working title:** *Reading Attention at the Wrong Depth: A Budget-Matched Audit of Attention-Guided Cropping in VLMs*
**Alternative:** *Depth Re-ranking: Where to Read a Vision-Language Model's Attention, and What It Costs to Read It Wrong*

---

Vision-language models routinely localise the evidence a question needs while failing to perceive it: at a fixed
visual-token budget the attention map points at the right region, yet the answer is wrong because the evidence was
merged away during encoding. The standard remedy is attention-guided cropping — read the model's own attention, crop
where it peaks, re-encode the crop. Every such method must choose **a depth at which to read that map**, and published
methods hand-pick a single layer or average a block. We show this choice dominates everything else about the method:
cropping at the peak of one hand-picked mid layer scores **24.6% / 33.9%** on two checkpoints — the four-option chance
rate, and far below not cropping at all (62.3% / 57.4%).

We introduce **depth re-ranking**: one closed-form ridge solve over the 28-layer attention profile of each image cell,
fit on as few as 50 boxed examples, whose weights form a **signed depth filter** — it subtracts the late layers that
the conventional block-mean read-out adds. It beats the strongest published placement rule at matched compute on both
checkpoints (**+7.9 and +8.9 points**) and the equal-compute baseline by **+8.9 and +12.0**; on a third benchmark at
n=800 it gains **+10.2** on single-instance questions.

We then evaluate the whole family under three conditions it is not usually tested under — **matched token budgets**,
**stratification by question type**, and a **causal measurement of the read-out depth**. Under matched budgets the
published methods are worth **0–3 points**, and the conventional read-out is *below* baseline on one checkpoint.
Stratified, cropping is not uniformly helpful but **sign-separated**: across 14 cells (3 benchmarks x 4 checkpoints x
2 question types) it helps single-instance perception in 7/7 and is non-positive on cross-instance perception in 7/7,
costing up to 8.8 points — because a tight crop magnifies one object and discards the other. Causally, masking text
tokens' attention to the image layer by layer shows image-to-text transport **completes by layer 16 of 28** (blocking
everything from layer 16 onward flips no answers), while every method in the family — including ours — reads its map
at layers 17-22, **after the window has closed**. The attention these methods read is a residue of where the model
looked, not the channel that carries the evidence; this explains why depth re-weighting works, why depth *selection*
never does, and why placement rules separate only modestly. One corollary follows immediately: 90% of visual tokens can
be dropped at that boundary at **zero accuracy cost** on both checkpoints and any keep ratio, saving ~35% of prefill.

We release all negative results, including nine designs that failed to make cropping work on cross-instance questions.
