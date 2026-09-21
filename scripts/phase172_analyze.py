"""Analyzer for the context graft (phase 172). Pre-registered contrasts only."""
import json, sys
import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
rng = np.random.default_rng(172)


def ci(d, B=8000):
    n = len(d)
    b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean()*100, float(np.percentile(b, 2.5))*100, float(np.percentile(b, 97.5))*100


for which, mdl in [("qwen3", "Qwen3-VL-2B"), ("qwen2", "Qwen2-VL-7B")]:
    rows = [json.loads(l) for l in open(f"{D}/phase172_context_{which}.jsonl")]
    cat = np.array([r["category"] for r in rows])
    A = lambda k: np.array([int(np.argmax(r["probs"][k]) == r["label"]) for r in rows], float)
    T = lambda k: np.mean([r["realized_tokens"][k] for r in rows])
    bar, head = A("uniform@600"), A("head@0.25")
    print(f"\n{mdl}  n={len(rows)}")
    print(f"  tokens: bar {T('uniform@600'):.0f} | head {300+T('head@0.25'):.0f} "
          f"| ctx64 {300+T('ctx64'):.0f} | ctx128 {300+T('ctx128'):.0f}  (localise@300 included)")
    print(f"  {'stratum':>18} {'bar':>6} {'head':>6} {'ctx64':>6} {'ctx128':>7} | "
          f"{'PRIMARY ctx-bar':>22}  {'GUARD ctx-head':>22}")
    for s in ["relative_position", "direct_attributes", "ALL"]:
        m = np.ones(len(rows), bool) if s == "ALL" else cat == s
        line = f"  {s:>18} {100*bar[m].mean():5.1f}% {100*head[m].mean():5.1f}%"
        best = {}
        for arm in ["ctx64", "ctx128"]:
            a = A(arm); best[arm] = a
            line += f" {100*a[m].mean():5.1f}%" + ("" if arm == "ctx64" else " ")
        out = ""
        for arm in ["ctx64", "ctx128"]:
            p = ci((best[arm]-bar)[m]); g = ci((best[arm]-head)[m])
            out += (f"\n      {arm:>6}  vs bar {p[0]:+5.1f} [{p[1]:+5.1f},{p[2]:+5.1f}]"
                    f"{'✔' if p[1] > 0 else ' '}   vs head {g[0]:+5.1f} [{g[1]:+5.1f},{g[2]:+5.1f}]"
                    f"{'  GUARD BREACH' if g[2] < 0 else ''}")
        print(line + out)
