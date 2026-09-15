# Phase 78 — Is there anything for a learned allocation function to learn?

**Classification: FINDING + open direction**

## 1. Research question

The method learns *where* to crop but the window size is fixed at 0.15, inherited from a sweep that
used the **old** proposer. Window size is a genuine trade-off and it is exactly the cliff: a wider
window is more likely to *contain* the target and less likely to *resolve* it, because
`tokens_on_target = (target_area / W²) × B₀`. Is a fixed size leaving anything on the table?

## 2. Finding, in simple English

**Three things, and the first one is free money.**

The deployed window is the wrong size. At the new proposer's placement, **W = 0.25 scores 71.7%
against W = 0.15's 68.6%** — three free points that we were losing because the window had been tuned
for a worse proposer. That is counter-intuitive: a *better* aim wants a *wider* window, not a
tighter one, because the proposal still misses about half the time and a wider window is more
forgiving of a near-miss.

Second, there is real room for a learned sizer. Picking the best size per item would reach 86.4%
against the best single size's 71.7%. That number is optimistic — it is chosen with the answer key,
on the same items — so we ran the check that killed a similar-looking result earlier in the project:
require the item to be correct at a size *and* at an adjacent size, so isolated lucky hits do not
count. The ceiling only drops to 82.7%, still **+11.0pp**. Unlike the budget-controller ceiling,
which collapsed entirely when the same test was applied, this headroom survives.

Third, and this is the problem: **target size does not predict the right window size.** Split items
into four quartiles by how big the target actually is — spanning a 100× range of area — and the best
window is 0.15 in every single quartile. So the obvious signal is useless, exactly as it was when we
tried to predict token budget from image features. Whatever decides the right window is not object
scale.

## 3. Numbers that changed

| W | head placement | oracle placement | magnification |
|---|---|---|---|
| 0.15 (deployed) | 68.6% | 90.1% | 6.7× |
| **0.25** | **71.7%** | 88.0% | 4.0× |
| 0.35 | 69.1% | 83.8% | 2.9× |
| 0.50 | 65.4% | 79.6% | 2.0× |
| 0.70 | 63.4% | 65.4% | 1.4× |

best fixed W = **0.25 → 71.7%** · oracle per-item W **86.4%** · **stable** oracle per-item W
**82.7%** → learnable headroom **+11.0pp**. Non-monotonic flips across W: 11.5%.

Best W by target-size quartile (median area 0.00015 → 0.01606): **0.15 in all four**.

## 4. Keep in paper: **7/10**

The W = 0.25 correction belongs in the method (with the caveat below). The +11.0pp learnable
headroom belongs in future work, honestly labelled as an upper bound with no known predictor.

⚠ **W = 0.25 is selected in-sample**, on the same 191 items it is scored on. The pre-registered
W = 0.15 is what the headline claim should use (+12.0pp); W = 0.25 needs held-out selection before
it can be quoted as +15.2pp.

## 5. Experiment, stepwise

1. **Sweep W at the head's placement and at oracle placement**, five sizes, accuracy not coverage.
   Coverage is a vacuous objective here — W = 1.0 "covers" 100% by construction — so only accuracy
   can price the trade-off.
2. **Keep the budget matched.** Every crop refits to B₀ = 300 regardless of W, so the arms differ in
   magnification and nothing else; all landed within 10% of target.
3. **Compute the oracle-per-item-W ceiling** — the entire budget available to any learned sizer,
   however clever.
4. **Then attack that ceiling as noise**, because an identical-looking +5.2pp ceiling earlier in the
   project collapsed to zero under this test. Two checks: what independence at the observed rates
   would predict, and a stability requirement (correct at a size *and* an adjacent size). The
   headroom survives both.
5. **Ask whether the obvious predictor works** before proposing to learn one: stratify by true target
   area and read off the best W per stratum. It is flat, so target scale is not the signal.
