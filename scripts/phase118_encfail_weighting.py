"""
Phase 118: train the head for the items where placement MATTERS.

The head regresses coverage with every item weighted equally. But allocation only changes the answer
on items uniform@300 gets WRONG and the oracle crop gets RIGHT (certified encoding failures, ~40% of
items). On items uniform already answers, coverage is irrelevant; on items the oracle cannot fix,
nothing helps. Up-weighting encfail items in the training loss is free -- outcomes are on disk.

PRE-REGISTERED: primary weight w=3 on encfail items (others 1). w in {1,3,10} reported; w=1 is the
incumbent. Metrics: OOF top-1 coverage overall AND on encfail items. Noise floor is ~1pp (SS14Z);
a gain must clear it with a CI clear of zero, on BOTH models, or it is not claimed.
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/vlm-hallu/scripts")
import phase70_rerank_head as P70
import phase80a_qwen2vl_head as P80
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
P70.W = 0.25
D = "/home/kavinder/ARNABI_ARSH/vlm-hallu/data"

def oof_w(X, Y, G, wgt, seeds=3, neg_per_item=30):
    P = np.zeros(len(Y)); gs = np.unique(G)
    for s in range(seeds):
        rng = np.random.default_rng(700 + s)
        perm = {g: i for i, g in enumerate(rng.permutation(gs))}; Gp = np.vectorize(perm.get)(G)
        for tr, te in GroupKFold(5).split(X, Y, Gp):
            ytr = Y[tr]; pos = tr[ytr > 0]; negpool = tr[ytr <= 0]
            k = min(len(negpool), neg_per_item * len(np.unique(G[tr])))
            neg = rng.choice(negpool, size=k, replace=False); sub = np.concatenate([pos, neg])
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=150, learning_rate=0.10, random_state=s)
            m.fit(X[sub], Y[sub], sample_weight=wgt[sub]); P[te] += m.predict(X[te])
    return P / seeds

for name, build, attn, outf in [("Qwen3-VL", P70.build, f"{D}/phase30c_attn_maps_all.jsonl", f"{D}/phase78_w_sweep.jsonl"),
                                 ("Qwen2-VL", P80.build, f"{D}/phase74_Qwen2_VL_7B_Instruct.jsonl", f"{D}/phase97m_merged_qwen2vl.jsonl")]:
    r = build(); X, Y, G, DEP, rows = r[0], r[1], r[2], r[3], r[4]
    outs = {json.loads(l)["question_id_full"]: json.loads(l) for l in open(outf)}
    enc_item = np.array([ (np.argmax(outs[q["question_id_full"]]["probs"]["uniform@300"]) != outs[q["question_id_full"]]["label"]) and
                          (np.argmax(outs[q["question_id_full"]]["probs"]["oracle@0.25"]) == outs[q["question_id_full"]]["label"]) for q in rows])
    ring = []
    for q in rows:
        gh, gw = q["grid"]; m = np.zeros((gh, gw), bool)
        if gh > 2 and gw > 2: m[1:-1, 1:-1] = True
        else: m[:] = True
        ring.append(m.ravel())
    def top1(P):
        return np.array([float(Y[G == gi][int(np.argmax(np.where(ring[gi], P[G == gi], -1e9)))] >= P70.COV_HIT) for gi in np.unique(G)])
    res = {}
    for w in [1, 3, 10]:
        wgt = np.where(enc_item[G], float(w), 1.0)
        res[w] = top1(oof_w(X, Y, G, wgt))
    n = len(rows); rng = np.random.default_rng(118)
    def ci(d):
        m_ = len(d)   # FIX: bootstrap over the length of the vector passed, not the full n
        b = np.array([d[rng.integers(0, m_, m_)].mean() for _ in range(6000)])
        return d.mean()*100, np.percentile(b, 2.5)*100, np.percentile(b, 97.5)*100
    print(f"\n{name}: n={n}, encfail items {int(enc_item.sum())}")
    print(f"  {'weight':>7} {'cov all':>8} {'cov encfail':>12}   vs w=1 (all)        vs w=1 (encfail)")
    for w in [1, 3, 10]:
        a, lo, hi = ci(res[w] - res[1]); e = enc_item
        a2, lo2, hi2 = ci((res[w] - res[1])[e]) if e.sum() > 0 else (0, 0, 0)
        print(f"  {w:>7} {res[w].mean()*100:7.1f}% {res[w][e].mean()*100:11.1f}%   {a:+5.1f} [{lo:+5.1f},{hi:+5.1f}]   {a2:+5.1f} [{lo2:+5.1f},{hi2:+5.1f}]")
