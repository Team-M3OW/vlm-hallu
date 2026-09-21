"""Analyzer for phase 188 (layer-wise pruning ranked by the DPR head). Pre-registered contrasts only."""
import json
import numpy as np

D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"
KEEP = [0.10, 0.25]
rng = np.random.default_rng(1880)


def ci(d, B=8000):
    n = len(d); b = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean()*100, float(np.percentile(b, 2.5))*100, float(np.percentile(b, 97.5))*100


for which, mdl in [("qwen3", "Qwen3-VL-2B"), ("qwen2", "Qwen2-VL-7B")]:
    try:
        rows = [json.loads(l) for l in open(f"{D}/phase188_prune_{which}.jsonl")]
    except FileNotFoundError:
        print(f"\n{mdl}: not finished"); continue
    A = lambda k: np.array([int(np.argmax(r["probs"][k]) == r["label"]) for r in rows], float)
    none = A("none")
    print(f"\n{'='*92}\n{mdl}  n={len(rows)}   no-pruning upper bound {100*none.mean():.1f}%")
    for kf in KEEP:
        print(f"  --- keep {int(kf*100)}% of image tokens")
        print(f"      {'ranker':>12} {'single cut @L2':>16} {'progressive L2/L8/L16':>24}")
        for nm in ["rand", "layer2", "blockmean", "head"]:
            s = f"single_{nm}@{kf}"
            if s not in rows[0]["probs"]: continue
            acc_s = A(s)
            pg = f"prog_{nm}@{kf}"
            cell_p = f"{100*A(pg).mean():6.1f}%" if pg in rows[0]["probs"] else "      —"
            print(f"      {nm:>12} {100*acc_s.mean():15.1f}% {cell_p:>24}")
        bm, hd = A(f"single_blockmean@{kf}"), A(f"single_head@{kf}")
        rd = A(f"single_rand@{kf}")
        m, lo, hi = ci(hd-bm)
        print(f"      P1  head - blockmean (single) {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}]"
              f"{' CLEARS' if lo > 0 else ''}")
        if f"prog_head@{kf}" in rows[0]["probs"]:
            m2, lo2, hi2 = ci(A(f"prog_head@{kf}")-hd)
            print(f"      P2  progressive-head - single-head {m2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]"
                  f"{' CLEARS' if lo2 > 0 else ''}")
        g, glo, _ = ci(hd-rd)
        print(f"      GUARD head - rand {g:+5.1f} [{glo:+5.1f},..]{'  ok' if glo > 0 else '  BREACH'}")
        d, dlo, dhi = ci(hd-none)
        print(f"      cost check: head@{int(kf*100)}% - no-pruning {d:+5.1f} [{dlo:+5.1f},{dhi:+5.1f}]"
              f"{'  (lossless at this keep)' if dlo > -2.5 else ''}")
