"""Analyzer for phase 183 (DPR -> VRH reinforcement). Pre-registered contrasts only."""
import json, sys
import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
rng = np.random.default_rng(1830)


def ci(d, B=8000):
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean()*100, float(np.percentile(b, 2.5))*100, float(np.percentile(b, 97.5))*100


for which, mdl in [("qwen3", "Qwen3-VL-2B"), ("qwen2", "Qwen2-VL-7B")]:
    try:
        rows = [json.loads(l) for l in open(f"{D}/phase183_reinforce_{which}.jsonl")]
    except FileNotFoundError:
        print(f"\n{mdl}: not finished"); continue
    A = lambda k: np.array([int(np.argmax(r["probs"][k]) == r["label"]) for r in rows], float)
    base = A("base")
    cat = np.array([r["category"] for r in rows])
    cov = np.array([r["head_cov"] >= 0.5 for r in rows])
    print(f"\n{'='*90}\n{mdl}  n={len(rows)}   bar = uniform@300 (the intervention is free)")
    print(f"  base (uniform@300) {100*base.mean():5.1f}%   uniform@600 (reference) "
          f"{100*A('uniform@600').mean():5.1f}%")
    print(f"  {'arm':>16} {'acc':>7} {'Δ vs base':>24}   note")
    notes = {"vrh_dpr_l2": "PRIMARY (pre-registered)", "vrh_rand_l2": "CONTROL, must be null",
             "all_dpr_l2": "does head restriction matter?", "vrh_oracle_l2": "CEILING, perfect router",
             "vrh_dpr_l1": "sweep (secondary)", "vrh_dpr_l4": "sweep (secondary)"}
    for arm in ["vrh_dpr_l1", "vrh_dpr_l2", "vrh_dpr_l4", "all_dpr_l2", "vrh_rand_l2",
                "vrh_oracle_l2"]:
        a = A(arm); m, lo, hi = ci(a - base)
        print(f"  {arm:>16} {100*a.mean():6.1f}% {m:+7.1f} [{lo:+6.1f},{hi:+6.1f}]"
              f"{' ✔' if lo > 0 else '  '}   {notes.get(arm,'')}")
    p = A("vrh_dpr_l2"); o = A("vrh_oracle_l2"); r_ = A("vrh_rand_l2"); al = A("all_dpr_l2")
    for nm, d in [("PRIMARY − control (vrh_dpr_l2 − vrh_rand_l2)", p - r_),
                  ("head restriction (vrh_dpr_l2 − all_dpr_l2)", p - al),
                  ("localisation gap (oracle − dpr)", o - p)]:
        m, lo, hi = ci(d)
        print(f"  {nm:<46} {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{' ✔' if lo > 0 else ''}")
    print("  --- strata (vrh_dpr_l2 − base) ---")
    for nm, msk in [("single-object", cat == "direct_attributes"),
                    ("relational", cat == "relative_position"),
                    ("DPR window covers", cov), ("DPR window misses", ~cov)]:
        if msk.sum() < 8: continue
        m, lo, hi = ci((p - base)[msk])
        print(f"    {nm:20} n={msk.sum():3d}  {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{' ✔' if lo > 0 else ''}")
