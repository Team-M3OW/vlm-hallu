# Phase 59 — Does multi-crop transfer to 4K?

## 1. Research question
Is Phase 58 a V\*Bench result or the method? Test on HR-Bench `single` — the qualifying regime —
where every previous version of our method failed.

## 2. Finding and contribution (plain English)
Two results pointing opposite ways, and both matter.

**The innovation transfers**: multi-crop beats single-crop by **+8.0pp** at 4K, significant, at equal
passes and tokens. That replicates V\*Bench's +7.9pp almost exactly at 2.7× the image scale on a
different benchmark. Handing the model k candidates instead of one is a **scale-general** improvement.

**The method's win does not**: multi4 is **−7.3pp** against the budget axis. On this subset uniform
runs 53.8 → 64.5 → 72.0%, i.e. **+18.2pp for 4× tokens** — a steeper return than any crop policy we
or the literature produced. So the V\*Bench win is scale-bound.

The important correction this forced: the boundary is not primarily about **proposal precision**, it
is about the **budget axis being unusually productive at 4K**. That is a property of the baseline, not
a flaw we can engineer around — which is why four separate proposal fixes all failed.

## 3. Numbers that changed
| arm | passes | tokens | acc | bar | margin |
|---|---|---|---|---|---|
| top1@0.15 | 2 | 580 | 49.8% | 64.5% | −14.8 |
| **multi4** | 2 | 611 | 57.8% | 65.1% | **−7.3** [−12.3,−2.6] |
| gated multi4 | 1.4 | 418 | 63.5% | 59.5% | **+4.0** [−0.7,+8.8] n.s. |

**multi4 − top1 = +8.0pp [+3.0,+13.0] SIG** (and multi3 − top1 = +8.0pp [+4.0,+12.2]).

## 4. Keep in paper: 8/10
Gives the method a two-part claim with both parts significant: the **innovation** generalises (+8pp
at both scales), and the **scope** is bounded (wins at ~1500px, loses at 4032px).

## 5. Experiment, step by step
1. Restrict to HR-Bench `single` — the regime where the coverage condition is satisfied.
2. Join the uniform bar arms from Phase 53 on identical rows, with 20 asserted consistency checks.
3. Compute the bar **on this subset**, not on all items — an all-items bar understates what uniform
   achieves on easier items.
4. Run top1, multi2/3/4 and a 2× budget variant at matched total tokens.
5. Report per-row and CircularEval.
