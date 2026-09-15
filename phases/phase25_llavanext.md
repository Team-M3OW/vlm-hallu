# Phase 25 — Query placement vs native AnyRes on LLaVA-NeXT, done properly

## 1. Research question
Phase 23's comparison was voided by a budget confound. Redone with a strict gate: does query
placement beat native AnyRes at matched realized tokens on LLaVA-NeXT?

## 2. Finding and contribution (plain English)
The useful output is a **correction and a bound**. The original gate was median-based and too
lenient; tightened, several versions of the table turned out void. And LLaVA-NeXT has almost no
budget axis to move along — which becomes a first-class finding in Phase 28.

## 3. Numbers that changed
- Gate corrected from median-based to a stricter per-item criterion; **three void versions preceded
  the reportable table**.
- LLaVA-NeXT's dynamic range is later measured at **1.5×** (1416 → 2144 tokens).

## 4. Keep in paper: 3/10
Fold into the architecture-scope paragraph. The "three void versions" note belongs in the methods
discipline section.

## 5. Experiment, step by step
1. Match realized tokens between AnyRes and query-placed allocation per item, not on medians.
2. Void any item whose arms differ by more than the gate.
3. Report what remains, and report how many versions were voided before it.
