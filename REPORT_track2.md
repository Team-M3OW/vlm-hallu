# Track 2 — a more sophisticated head than the GBT: KEEP THE TREE (phase 130)

Script `scripts/phase130_cnn_head.py`; outputs `data/phase130_{model}_W{W}.json`. OOF, GroupKFold(5)
grouped by item, folds identical to phase 70, 30 negatives/item, ring mask, top-1 coverage at 0.5.
GBT reproduced on all four models within a subsampling draw.

Arms: deployed argmax · gbt3 (incumbent, 3 seeds) · gbt9 · gbt3_flip (h-flip augmentation of
map+box+geometry) · cnn_mse / cnn_pair (Conv3×3(C→16)–ReLU–Conv3×3–ReLU–Conv1×1 over per-layer map +
rank + 7 geometry channels; Adam 3e-3, wd 1e-3, early stop on 20% of *training* items, flip aug,
3 seeds; MSE vs pairwise RankNet) · ens (within-item rank-average gbt3 + better CNN).

## W=0.15, Δ vs gbt3 [95% CI]
| arm | Qwen3-VL | Qwen2-VL | LLaVA-NeXT | LLaVA-OneVision |
|---|---|---|---|---|
| gbt3 | **50.8** | **44.0** | **20.4** | **28.8** |
| gbt9 | +1.6 [−1.6,+4.7] | +0.0 [−3.1,+3.1] | +0.0 [−4.2,+4.2] | −1.6 [−5.8,+2.1] |
| gbt3_flip | +0.0 [−3.7,+3.7] | +0.5 [−3.1,+4.2] | −2.1 [−6.3,+1.6] | −1.6 [−6.3,+2.6] |
| cnn_mse | **−6.8 [−12.6,−1.0]** ✗ | **−8.9 [−15.7,−2.1]** ✗ | +1.0 [−5.2,+7.3] | −2.6 [−8.9,+3.7] |
| cnn_pair | −1.0 [−6.8,+4.7] | −4.7 [−11.5,+2.1] | +0.0 [−6.3,+6.3] | −1.6 [−7.9,+4.7] |
| ens | +0.0 [−4.7,+4.7] | −2.1 [−7.3,+3.1] | +1.6 [−4.2,+7.3] | +3.1 [−2.1,+8.4] |

W=0.25 (Qwen): gbt3 59.7 / 56.5; gbt9 +2.1 [−0.5,+5.2] / −1.0; cnn_mse +0.5 / **−9.4 [−15.7,−3.1]**;
cnn_pair −0.5 / −6.3 [−12.6,+0.0]; ens +1.6 / **−5.2 [−10.5,−0.5]**.

## Verdict
**Nothing clears on both models.** The spatial CNN is significantly *worse* on Qwen2 at both windows
and no better on Qwen3; ensembles are 1 of 2 at best; 9 seeds ≤ +2.1 inside the noise floor; flip is
null. Fourth independent confirmation (phases 45, 102, 112, 130) that **capacity is not the limit at
n=191** — six configurations across four architectures. Secondary: for neural heads the objective
ordering is listwise < MSE < pairwise, but the best only ties the tree.

Honest framing for the paper: *a small feature-engineered ranker whose gains are data-limited, chosen
because more expressive heads overfit at this n.* The one lever that could change it is more boxed
training items.
