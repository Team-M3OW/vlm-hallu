# Phase 29 — Is our V\*Bench read-out inflating the numbers?

## 1. Research question
Our V\*Bench accuracies are higher than some published numbers. Is the logit read-out unfairly
generous compared to generation-based scoring?

## 2. Finding and contribution (plain English)
No. The read-out is not the source of the gap — the **oracle arm** is. We report an oracle-placed
crop, which prior work does not have, and that is where the difference comes from. An honest check
that prevented a reviewer from finding it first.

## 3. Numbers that changed
Read-out comparison shows no inflation; the gap to prior work is attributable to the oracle arm.

## 4. Keep in paper: 5/10
One paragraph in Methods. It is the kind of check that buys credibility cheaply.

## 5. Experiment, step by step
1. Score V\*Bench with the logit read-out and with generation-based matching.
2. Compare on identical items.
3. Attribute any remaining gap to arm composition rather than scoring.
