# Phase 28 — Does the exchange rate hold across architectures?

## 1. Research question
Is the placement effect a Qwen phenomenon, and how much budget axis does each architecture even have?

## 2. Finding and contribution (plain English)
Placement beats random placement on **all four** architectures. But the more useful finding is that
**dynamic range is an architectural property** spanning three orders of magnitude — and on
LLaVA-NeXT there is essentially **no axis to move along**, so we report "NO AXIS" rather than a
misleadingly small bound.

## 3. Numbers that changed
Dynamic range (floor → ceiling, measured): Qwen2-VL **4→7776 (1944×)** · Qwen3-VL **64→7957 (124×)** ·
OneVision **1261→7329 (5.8×)** · LLaVA-NeXT **1416→2144 (1.5×)**.

Placement effect (oracle crop vs random crop at B₀): Qwen2-VL **+59.2pp** · OneVision **+49.2pp** ·
LLaVA-NeXT **+42.4pp**. Exchange rates ≥25.9× / ≥3.9× / ≥1.2× (the last = no axis).

## 4. Keep in paper: 8/10
Establishes the scope of the claim honestly, including where it does **not** apply. The "NO AXIS"
reporting decision is worth stating explicitly.

## 5. Experiment, step by step
1. For each architecture, find the token floor and ceiling by climbing a resize ladder and
   **measuring** the realized count at each rung.
2. Stop the ladder on two equal counts (the ceiling), and guard against unreachable targets — an
   earlier version chased a 35568-token target on a 2144-capped model and OOM-killed the host.
3. Run oracle crop and random crop at the floor budget.
4. Report dynamic range alongside every exchange rate, and refuse to quote a bound where the range
   is ~1.
