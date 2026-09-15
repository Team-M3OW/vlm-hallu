# Phase 55 — Is the gate proposer-agnostic?

## 1. Research question
Our contribution is arguably *when to spend*, not *which proposer to use*. Does wrapping the free
pass-1 gate around **any** proposer improve it — including the baselines?

## 2. Finding and contribution (plain English)
**No, not in general** — and the reason is structural and I should have anticipated it. `peak` is the
attention map's own maximum, so it measures **that proposer's** reliability, not proposal quality
generally. It helps our proposer at both scales and each baseline at exactly one.

What it did produce is the largest single improvement in the project: gating turns our 4K result from
catastrophic (−17.3pp) to break-even (−0.2pp) **while spending 30% fewer tokens**.

And a real result about the literature, even though it is not universal: gating recovers **+5.7pp for
Zoom Eye at 4K while cutting its tokens by 53%** — tree search wastes most of its 9 passes on items
where no crop helps.

## 3. Numbers that changed
| proposer | V\*Bench gain from gating | HR-Bench gain | verdict |
|---|---|---|---|
| **attention peak (ours)** | **+7.9pp** | **+17.0pp** | both |
| grounding | +6.9pp | −2.1pp | V\* only |
| tree search (Zoom Eye) | −1.1pp | +5.7pp | HR only |

## 4. Keep in paper: 6/10
Keep the +17.0pp result and the Zoom Eye cost saving. **Drop the "proposer-agnostic" framing** — the
data does not support it.

## 5. Experiment, step by step
1. Define the wrapper: if `peak < τ` answer from pass 1; else run proposer P and answer from whichever
   of {uniform, P} is more confident.
2. Apply it to three proposers on two benchmarks, entirely offline from existing logs.
3. Use the **transferred** τ throughout.
4. Score each arm against the uniform sweep measured in its own run on its own items.
5. Report both the margin gain and the token saving.
