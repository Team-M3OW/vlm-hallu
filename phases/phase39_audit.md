# Phase 39 — Audit: four things that decide how §3 and §5 may be stated

## 1. Research question
Before publishing the mechanism and the method, check four things that could each invalidate a claim.

## 2. Finding and contribution (plain English)
Three of the four went against us, which is why the phase was worth running.

**(A) An arm mismatch in the capture fraction.** We had divided the TEXT+CONF arm's gain by the
*conf-only* arm's ceiling, getting "83%". Matched, it is **88%** — and the mismatched pairing gives
95%, which would have **inverted** the conclusion from "detection is the bottleneck" to "the proposal
is".

**(B) The exogenous-coverage instrument is inconclusive**, not supporting: only 11/191 random windows
overlap the box at all.

**(C) The interior optimum is retracted.** Bootstrapped, every W contrast crosses zero.

**(D) The zero-coverage failures are FAR misses**, not near ones — which redirected the next
experiment away from bigger windows.

## 3. Numbers that changed
- Capture fraction **83% → 88%** (matched arm and ceiling).
- W sweep: +4.7 / +6.3 / +5.8 / +2.6, **all CIs cross zero**; peak vs neighbours n.s.
- Random-placement instrument fires on **11/191** items; hit bin CI [−72.7, +0.0].
- Missed windows: median gap **0.231 of image width**; only **6.2%** within 0.05; W=0.35 rescues 20%.

## 4. Keep in paper: 6/10
Not a result section, but the corrections belong in the record and two of them (A and C) changed
published-looking numbers.

## 5. Experiment, step by step
1. Recompute each arm's capture fraction against **its own** ceiling — same routing rules, routing on
   true coverage.
2. Replay phase32's seeded RNG to recover the random centres; pre-register the test as **one-sided**
   (a desynced replay cannot manufacture a dose–response, so a positive is valid and a null is
   ambiguous).
3. Bootstrap the W sweep, including paired peak-vs-neighbour contrasts.
4. Measure the distance from the window **edge** to the box on the zero-coverage items.
