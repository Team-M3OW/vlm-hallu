# Phase 13 — Sweep every non-crop intervention at once

## 1. Research question
The user ruled out image preprocessing as a contribution. So: among interventions that do **not**
crop, which actually recover confident denials — raising resolution, removing scene context, or
pointing the model at the object?

## 2. Finding and contribution (plain English)
Three results, one of them genuinely surprising.

**First, the number that reframed the project: the median confident-denial object occupies 0.4
merged visual tokens — less than one token.** These objects are *sub-token*.

**Second, two additive mechanisms.** Raising resolution with the scene fully retained gives a clean
dose–response (0.4 → 1.8 → 4.1 → 32 tokens maps to 0% → 27.8% → 50% → 72.2%). Removing the scene at
*identical* token budget also helps (0% → 27.8%). Resolution is the larger of the two.

**Third, and the surprise: pointing does nothing.** Drawing a red box on the object recovers 5.6%;
telling the model the coordinates in words recovers **0.0%**.

> You cannot point a VLM at what it has not encoded — you can only spend more tokens on it, or
> remove what competes with it.

## 3. Numbers that changed
| arm | tokens on object | recovery | FP | disc |
|---|---|---|---|---|
| baseline | 0.4 | 0.0% | 0.0% | +0.0 |
| `black_outside` | 0.4 | 27.8% | 2.2% | +25.6 |
| `upscale2x` | 1.8 | 27.8% | 1.4% | +26.3 |
| **`upscale3x`** | 4.1 | **50.0%** | 0.7% | +49.3 |
| `redbox` | 0.4 | 5.6% | 0.7% | +4.8 |
| `prompt_coords` | 0.4 | **0.0%** | 0.0% | +0.0 |
| `crop_alone` | 32.0 | 72.2% | 0.7% | +71.5 |

## 4. Keep in paper: 8/10
The "pointing does nothing" result and the 0.4-token number are both strong. **Carry the
correction:** the original explanation ("nothing there to attend to") was *refuted* by Phase 22 —
the target **is** attended, it just doesn't carry the detail. §13 later states this properly.

Also note the practical corollary: `upscale3x` needs **no bounding box** and recovers ~70% of
crop's benefit — a real threat to the crop literature's premise.

## 5. Experiment, step by step
1. Build 11 arms in three families: scene-removal at constant resolution (`black/gray/blur/dim
   outside`), resolution-raising with the scene kept (`upscale2x/3x`), and pointing (`redbox`,
   `prompt_focus`, `prompt_coords`), plus `crop_alone`.
2. **Gate first:** verify the upscale arms genuinely raised realized token count on 141/141 items
   (median 300 → 1200 → 2700). A previous bug had them silently not doing so.
3. Compute tokens-on-object per arm from the logged `image_grid_thw` and the COCO box.
4. Score recovery and FP for every arm; report discrimination.
5. Restrict to RePOPE-clean (n=54) for the trustworthy numbers.
