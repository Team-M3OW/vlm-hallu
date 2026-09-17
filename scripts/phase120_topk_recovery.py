"""
Phase 120: how much of the head's miss is recoverable from its OWN ranking?

The far-miss diagnostic says misses are distractors (median 3-5 cells away), i.e. a RANKING failure.
If the covering cell sits in the head's top-2/3/5 on most misses, a cheap verification between the
head's top candidates could recover it. Multi-crop (4 crops in one pass) failed at -3.1pp, so any
verification must be a different mechanism: separate passes, matched budget, bar moves to
uniform@(300*passes).

Reports, both models, OOF head scores (3 seeds, GroupKFold by item, as phase 70):
  oracle-among-top-k coverage for k in {1,2,3,5,10}, overall and among top-1 misses.
  ALSO the oracle-among-top-k of the deployed block-mean argmax for reference (phase 70's 61.5% top-5).
Also: are the head's top-1 and top-2 in the same neighbourhood (<=1.5 cells) -- if so, a second crop
buys no new coverage and verification cannot help.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
import phase80a_qwen2vl_head as P80
P70.W = 0.25
ks = [1, 2, 3, 5, 10]
for name, build in [("Qwen3-VL", P70.build), ("Qwen2-VL", P80.build)]:
    r = build(); X, Y, G, DEP, rows = r[0], r[1], r[2], r[3], r[4]
    P = P70.oof(X, Y, G, seeds=3)
    res = {k: [] for k in ks}; miss = {k: [] for k in ks}; dres = {k: [] for k in ks}
    nb12 = []
    for gi, q in enumerate(rows):
        gh, gw = q["grid"]; m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        rm = m.ravel(); y = Y[G == gi]
        if y.max() < P70.COV_HIT: continue
        s = np.where(rm, P[G == gi], -1e9); order = np.argsort(-s)
        d = np.where(rm, DEP[gi], -1e9); dorder = np.argsort(-d)
        t1 = y[order[0]] >= P70.COV_HIT
        for k in ks:
            v = float(y[order[:k]].max() >= P70.COV_HIT); res[k].append(v)
            dres[k].append(float(y[dorder[:k]].max() >= P70.COV_HIT))
            if not t1: miss[k].append(v)
        c1, c2 = order[0], order[1]
        nb12.append(float(np.hypot(c1 // gw - c2 // gw, c1 % gw - c2 % gw) <= 1.5))
    print(f"\n{name}: items with a covering cell {len(res[1])}, top-1 misses {len(miss[1])}")
    print(f"  {'k':>3} {'head top-k':>11} {'argmax top-k':>13} {'misses recovered':>17}")
    for k in ks:
        print(f"  {k:>3} {100*np.mean(res[k]):10.1f}% {100*np.mean(dres[k]):12.1f}% {100*np.mean(miss[k]):16.1f}%")
    print(f"  head top-1 and top-2 within 1.5 cells: {100*np.mean(nb12):.1f}% of items")
